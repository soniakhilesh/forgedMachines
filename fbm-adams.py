import csv
import pandas as pd
import numpy as np
import ast
from itertools import product
from gurobipy import *
from network_adams import Network


# Create Network
def create_network():
    # datapath = "/Users/soni6/Box/Study/Project/Case Studies/dataset/"
    datapath = "C:/Users/kbada/PycharmProjects/forged-by-machines/"

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
        network.add_zip(zip_name, centroid_data.loc[int(zip_name), 'weighted_latitude'],
                        centroid_data.loc[int(zip_name), 'weighted_longitude'])

    # Create set of origin-destination pairs for orders and corresponding demands
    for index, row in orders.iterrows():
        origin_node = network.get_node(row['origin facility'])
        cust_zip = network.get_zip(row['customer ZIP'])
        quantity = row['quantity']
        network.add_commodity(origin_node, cust_zip, quantity)

    # Create possible paths for each commodity
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


def create_deterministic_model(network: Network):
    # Move this somewhere else!
    def dist(lat1, lat2, lon1, lon2):
        return 69.5 * abs(lat1 - lat2) + 57.3 * abs(lon1 - lon2)

    # Create path variables for each commodity
    m = Model("deterministic")
    # Decision variables
    x = {}
    for commodity in network.get_commodities():
        x[commodity] = m.addVars(network.get_commodity_paths(commodity), vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                 name='CommodityPath')
        # print(network.get_commodity_paths(commodity))
        # print(x[commodity][network.get_commodity_paths(commodity)[0]])
    y = m.addVars(network.get_arcs(), vtype=GRB.INTEGER, lb=0, name='NumTrucks')
    u = m.addVars(network.get_zips(), network.get_dest_nodes(), vtype=GRB.BINARY, lb=0, name='ZipDestinationMatch')
    unfulfilled = m.addVars(network.get_commodities(), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='FractionUnfulfilled')
    r = m.addVars(network.get_commodities(), vtype=GRB.CONTINUOUS, lb=0, name='RemainingDistanceZipToCustomer')

    m.update()
    m.modelSense = GRB.MINIMIZE

    # constraints
    m.addConstrs(
        (x[k].sum('*') + unfulfilled[k] == 1 for k in network.get_commodities()), name='CommodityFulfillment')

    m.addConstrs(
        (x[k][p] <= y[a] for k in network.get_commodities() for p in network.get_commodity_paths(k)
         for a in p.arcs), name='PathOpen')

    for a in network.get_arcs():
        m.addConstr((quicksum(p.commodity.quantity * x[p.commodity][p] for p in network.get_arc_paths(a))
                     <= 1000 * y[a]), name='ArcCapacity{a.name}')
    m.addConstrs(
        (u.sum(z, '*') == 1 for z in network.get_zips()), name='DestNodeSelection')

    m.addConstrs(
        (u[z, d] >= x[p.commodity][p] for z in network.get_zips() for d in network.get_dest_nodes()
         for p in network.get_dest_node_paths(d)), name='PathOpenZipDestination')

    m.addConstrs(
        (r[k] >= dist(k.dest.Lat, d.Lat, k.dest.Lon, d.Lon) * u[k.dest, d]
         for k in network.get_commodities() for d in network.get_dest_nodes()),
        name='DistanceDestinationNodeToZip')

    # m.addConstrs(
    #     (r[k] >= dist(network.get_zip_lat(network.get_commodity_dest(k)), network.get_lat(d),
    #                   network.get_zip_lon(network.get_commodity_dest(k)), network.get_lon(d)) * u[network.get_commodity_dest(k), d]
    #      for k in network.get_commodities() for d in network.get_dest_nodes()),
    #     name='DistanceDestinationNodeToZip')

    m.update()

    # Setting the objective function
    m.setObjective(quicksum((100 + 2*network.get_arc_dist(a))*y[a] for a in network.get_arcs()) +
                   quicksum(1000*k.quantity*unfulfilled[k] for k in network.get_commodities()) +
                   quicksum(r[k] for k in network.get_commodities()))
    m.update()

    m.optimize()

    for a in network.get_arcs():
        if y[a].x > 0:
            print("Trucks on Arc", a.name, "=", y[a].x)

    for z in network.get_zips():
        for d in network.get_dest_nodes():
            if u[z, d].x > 0:
                print("Zip ", z.name, " connected to ", d.name)



network0 = create_network()
create_deterministic_model(network0)
