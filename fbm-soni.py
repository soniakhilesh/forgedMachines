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

    # Add zips to network
    with open(datapath + "ZIP CODE.csv", newline='') as f:
        reader = csv.reader(f)
        zips = list(reader)
    zips = zips[0]
    for zip_name in zips:
        network.add_zip(zip_name, 0.0, 0.0)  # temporarily using 0 for latitude and longitude

    # Add commodities to network
    # Create set of orders
    orders = pd.read_csv(datapath + "DeliveryData/DeliveryData0.csv", index_col=0)
    orders = orders.drop(['customer location', 'scenario'], axis=1)
    orders = orders.groupby(by=['origin facility', 'customer ZIP']).agg({'quantity': 'sum'})
    orders = orders.reset_index()
    # Create set of origin-destination pairs for orders and corresponding demands
    for index, row in orders.iterrows():
        origin_node = network.get_node(row['origin facility'])
        cust_zip = network.get_zip(row['customer ZIP'])
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


def create_determinsitic_model(network: Network):
    # create path variables for each commodity
    m = Model("deterministic")
    # Decision variables
    x = {}
    for commodity in network.get_commodities():
        x[commodity] = m.addVars(network.get_commodity_paths(commodity), vtype=GRB.BINARY, lb=0, name='CommodityPath')
        # print(network.get_commodity_paths(commodity))
        # print(x[commodity][network.get_commodity_paths(commodity)[0]])
    y = m.addVars(network.get_arcs(), vtype=GRB.INTEGER, lb=0, name='NumTrucks')
    u = m.addVars(network.get_zips(), network.get_dest_nodes(), vtype=GRB.BINARY, lb=0, name='ZipDestinationMatch')
    m.update()
    m.modelSense = GRB.MINIMIZE

    # constraints
    m.addConstrs(
        (x[k].sum('*') == 1 for k in network.get_commodities()), name='OnePathPerCommodity')

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
    m.update()
    m.optimize()
    for v in m.getVars():
        if v.x>0:
            print('%s %g' % (v.varName, v.x))

network = create_network()
create_determinsitic_model(network)