import csv
import pandas as pd
import numpy as np
import ast
from itertools import product
from gurobipy import *
from network import Network


# Create Network
def create_network():
    datapath = "/Users/soni6/Box/Study/Project/Case Studies/dataset/"
    # initialise a network
    network = Network()

    # Add nodes to network
    nodes = pd.read_csv(datapath + "Node location.csv", usecols=['node name', 'location'])
    for index, row in nodes.iterrows():
        name = row['node name']
        lat = ast.literal_eval(row['location'])[0]
        lon = ast.literal_eval(row['location'])[1]
        network.add_node(name, lat, lon)

    # Add arcs to network
    # Origin-Trans
    for node1 in network.get_origin_nodes():
        for node2 in network.get_trans_nodes():
            network.add_arc(node1, node2)
    # Trans-Destination
    for node1 in network.get_trans_nodes():
        for node2 in network.get_dest_nodes():
            network.add_arc(node1, node2)
    # Origin-Destination
    for node1 in network.get_origin_nodes():
        for node2 in network.get_dest_nodes():
            network.add_arc(node1, node2)

    # Add commodities to network
    # Create set of orders
    orders = pd.read_csv(datapath + "DeliveryData/DeliveryData0.csv", index_col=0)
    customers = orders[['customer location', 'customer ZIP', 'quantity']]
    orders = orders.drop(['customer location', 'scenario'], axis=1)
    orders = orders.groupby(by=['origin facility', 'customer ZIP']).agg({'quantity': 'sum'})
    orders = orders.reset_index()

    # Separate customer coordinates into two columns (latitude and longitude)
    customers['latitude'] = np.nan
    customers['longitude'] = np.nan

    for index, row in customers.iterrows():
        customers.loc[index, 'latitude'] = ast.literal_eval(row['customer location'])[0]
        customers.loc[index, 'longitude'] = ast.literal_eval(row['customer location'])[1]

    customers = customers.drop(['customer location'], axis=1)

    customers['weighted_latitude'] = customers['quantity'] * customers['latitude']
    customers['weighted_longitude'] = customers['quantity'] * customers['longitude']

    # Calculating the numerator of the weighted average formula
    centroid_data = customers.groupby(by=['customer ZIP']).agg({'weighted_latitude': 'sum',
                                                                'weighted_longitude': 'sum'})

    # Quantities in orders table are separated by origin node, so group again by ZIP only
    zip_quantity = orders.groupby(by=['customer ZIP']).agg({'quantity': 'sum'})

    # Division by denominator (total quantity) to finish weighted average calculation
    for index, row in zip_quantity.iterrows():
        centroid_data.loc[index, 'weighted_latitude'] = \
            centroid_data.loc[index, 'weighted_latitude'].item() / row['quantity'].item()
        centroid_data.loc[index, 'weighted_longitude'] = \
            centroid_data.loc[index, 'weighted_longitude'].item() / row['quantity'].item()

    # Add zips to network
    with open(datapath + "ZIP CODE.csv", newline='') as f:
        reader = csv.reader(f)
        zips = list(reader)
    zips = zips[0]
    for zip_name in zips:
        # drop empty space: took 4 hours to find this bug
        zip_name = zip_name.replace(" ", "")
        network.add_zip(zip_name, centroid_data.loc[int(zip_name), 'weighted_latitude'],
                        centroid_data.loc[int(zip_name), 'weighted_longitude'])

    # add commodities
    for index, row in orders.iterrows():
        origin_node = network.get_node(row['origin facility'])
        cust_zip = network.get_zip(str(row['customer ZIP']))
        quantity = row['quantity']
        network.add_commodity(origin_node, cust_zip, quantity)

    # create possible paths for each commodity
    for commodity in network.get_commodities():
        origin = commodity.origin_node
        # O-D paths
        for dest in network.get_dest_nodes():
            arcs = []
            arcs.append(network.get_arc(origin.name + dest.name))
            network.add_path(arcs, commodity)
        # O-T-D paths
        for trans in network.get_trans_nodes():
            for dest in network.get_dest_nodes():
                arcs = []
                arcs.append(network.get_arc(origin.name + trans.name))
                arcs.append(network.get_arc(trans.name + dest.name))
                network.add_path(arcs, commodity)
    return network


def dist(lat1, lat2, lon1, lon2):
    return 69.5 * abs(lat1 - lat2) + 57.3 * abs(lon1 - lon2)


def create_determinsitic_model(network: Network):
    # create path variables for each commodity
    m = Model("deterministic")
    # Decision variables
    x = {}
    for commodity in network.get_commodities():
        x[commodity] = m.addVars(network.get_commodity_paths(commodity), vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                 name='CommodityPath')
    y = m.addVars(network.get_arcs(), vtype=GRB.INTEGER, lb=0, name='NumTrucks')
    u = m.addVars(network.get_zips(), network.get_dest_nodes(), vtype=GRB.BINARY, lb=0, name='ZipDestinationMatch')
    unfulfilled = m.addVars(network.get_commodities(), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='FractionUnfulfilled')
    r = m.addVars(network.get_zips(), vtype=GRB.CONTINUOUS, lb=0, name='RemainingDistanceZipToCustomer')
    max_load = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='MaxLoad')
    min_load = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='MinLoad')

    m.update()
    m.modelSense = GRB.MINIMIZE

    # constraints
    m.addConstrs(
        (x[k].sum('*') + unfulfilled[k] == 1 for k in network.get_commodities()), name='CommodityFulfillment')

    m.addConstrs(
        (x[k][p] <= y[a] for k in network.get_commodities() for p in network.get_commodity_paths(k)
         for a in p.arcs), name='PathOpen')

    m.addConstrs((quicksum(p.commodity.quantity * x[p.commodity][p] for p in network.get_arc_paths(a))
                  <= 1000 * y[a] for a in network.get_arcs()), name='ArcCapacity')
    m.addConstrs(
        (u.sum(z, '*') == 1 for z in network.get_zips()), name='DestNodeSelection')

    # slightly different from Kat's proposal: Kat's constraint makes model computationally very hard to solve
    m.addConstrs(
        (u[k.dest, d] >= sum(x[k][p] for p in network.get_commodity_dest_node_paths(k, d))
         for k in network.get_commodities() for d in network.get_dest_nodes()
         ), name='PathOpenZipDestination')

    # max and min load constraints
    m.addConstrs((min_load <= sum(u[k.dest, d] * k.quantity for k in network.get_commodities())
                  for d in network.get_dest_nodes()), name="MinLoad")
    m.addConstrs((max_load >= sum(u[k.dest, d] * k.quantity for k in network.get_commodities())
                  for d in network.get_dest_nodes()), name="MaxLoad")

    # I think next constraint has some ambiguity. What if a part of commodity is fulfilled by TP.
    # Why should we still penalise the distance from destination node? Should we instead define r on z?
    # I added an alternative to this constraint based on zips instead of commodities
    # m.addConstrs(
    #     (r[k] >= dist(k.dest.lat, d.lat, k.dest.lon, d.lon) * u[k.dest, d]
    #      for k in network.get_commodities() for d in network.get_dest_nodes()),
    #     name='DistanceDestinationNodeToZip')

    m.addConstrs(
        (r[z] >= dist(z.lat, d.lat, z.lon, d.lon) * u[z, d]
         for z in network.get_zips() for d in network.get_dest_nodes()),
        name='DistanceDestinationNodeToZip')

    # we probably need parameters from max_load-min_load and distance in order to convert them to cost scale
    # I am using the below parameters randomly
    lambda1 = 5
    lambda2 = 0.5
    m.setObjective(quicksum((100 + 2 * a.distance) * y[a] for a in network.get_arcs()) +
                   quicksum(1000 * k.quantity * unfulfilled[k] for k in network.get_commodities()) +
                   lambda1 * (max_load - min_load) +
                   lambda2 * quicksum(r[z] for z in network.get_zips())
                   )

    m.setParam("TimeLimit", 50)
    m.setParam("MIPGap", 0.01)
    m.update()
    m.optimize()

    # print output
    for a in network.get_arcs():
        if y[a].x > 0:
            print("Trucks on Arc", a.name, "=", y[a].x)

    for z in network.get_zips():
        for d in network.get_dest_nodes():
            if u[z, d].x > 0:
                print("Zip ", z.name, " connected to ", d.name)

    print("Max Load ", max_load.x)
    print("Min Load ", min_load.x)
    print("Missed Load ", sum(unfulfilled[k].x for k in network.get_commodities()))
    # cost
    total_cost= sum((100 + 2 * a.distance) * y[a].x for a in network.get_arcs()) + \
                sum(1000 * k.quantity * unfulfilled[k].x for k in network.get_commodities())
    tot_packages = sum(k.quantity for k in network.get_commodities())
    cost_per_package = total_cost/tot_packages
    print("Cost per package ", cost_per_package)
    # Katherine: can you calculate distance per package?


network0 = create_network()
create_determinsitic_model(network0)
