import csv
import pandas as pd
import numpy as np
import ast
from itertools import product
from gurobipy import *

# Import data
nodes = pd.read_csv("Node location.csv", usecols=['node name', 'location'])

with open("ZIP CODE.csv", newline='') as f:
    reader = csv.reader(f)
    zips = list(reader)
zips = zips[0]

# Just using scenario 0 for now
orders = pd.read_csv("DeliveryData0.csv", index_col=0)


# Split nodes into 3 types
origin_nodes = nodes[nodes['node name'].str[0] == 'O']
transfer_nodes = nodes[nodes['node name'].str[0] == 'T']
destination_nodes = nodes[nodes['node name'].str[0] == 'D']

# Convert node data to dictionary
nodes_coordinates = {}
for index, row in nodes.iterrows():
    nodes_coordinates[row['node name']] = ast.literal_eval(row['location'])

# Create path combinations
list_2_arcs = tuple(product(tuple(origin_nodes['node name']), tuple(destination_nodes['node name'])))
list_3_arcs = tuple(product(tuple(origin_nodes['node name']), tuple(transfer_nodes['node name']),
                            tuple(destination_nodes['node name'])))

list_paths = list_2_arcs + list_3_arcs

# Create set of arcs
arcs1 = list(product(tuple(origin_nodes['node name']), tuple(destination_nodes['node name'])))
arcs2 = list(product(tuple(origin_nodes['node name']), tuple(transfer_nodes['node name'])))
arcs3 = list(product(tuple(transfer_nodes['node name']), tuple(destination_nodes['node name'])))

arcs = arcs1 + arcs2 + arcs3

# Create set of orders
orders = orders.drop(['customer location', 'scenario'], axis=1)
orders = orders.groupby(by=['origin facility', 'customer ZIP']).agg({'quantity': 'sum'})
orders = orders.reset_index()

# Create set of origin-destination pairs for orders and corresponding demands
od_pairs = {}
for index, row in orders.iterrows():
    od_pairs[(row['origin facility'], row['customer ZIP'])] = row['quantity']


# Calculate delivery distance for each path
def delivery_distance(*locs):
    location1 = locs[0]
    location2 = locs[1]
    if len(locs) == 2:
        loc1 = nodes_coordinates[location1]
        lat1 = loc1[0]
        lon1 = loc1[1]
        loc2 = nodes_coordinates[location2]
        lat2 = loc2[0]
        lon2 = loc2[1]
        dist = 69.5*abs(lat1 - lat2) + 57.3*abs(lon1 - lon2)
    else:
        location3 = locs[2]
        loc1 = nodes_coordinates[location1]
        lat1 = loc1[0]
        lon1 = loc1[1]
        loc2 = nodes_coordinates[location2]
        lat2 = loc2[0]
        lon2 = loc2[1]
        loc3 = nodes_coordinates[location3]
        lat3 = loc3[0]
        lon3 = loc3[1]
        dist = 69.5 * abs(lat1 - lat2) + 57.3 * abs(lon1 - lon2) + 69.5 * abs(lat2 - lat3) + 57.3 * abs(lon2 - lon3)
    return dist


paths = {}
for i in list_paths:
    paths[i] = delivery_distance(*i)

# Create set of commodities and corresponding paths
commodities = orders.drop(['quantity'], axis=1)

# It only matters where commodity originates, the connection from destination node to ZIP will be determined later
# based on DV
commodity_paths = {}
for index, row in commodities.iterrows():
    commodity_paths[row['origin facility']] = []
    for i in list_paths:
        if i[0] == row['origin facility']:
            commodity_paths[row['origin facility']].append(i)


# Last adjustments to sets
P = list(list_paths)
A = tuplelist(arcs)
Z = tuplelist(zips)
D = tuplelist(list(destination_nodes['node name']))
K, q = multidict(od_pairs)
end = {}
for i in list_paths:
    for d in D:
        if i[-1] == d:
            end[i] = d
# K = tuplelist(list(commodities.to_records(index=False)))

# Printing datasets to check if they look correct
print(Z)
print(D)
print(K)
print(P)
print(commodity_paths)

# Start building the Model

m = Model("network")

# Decision variables
# HOW CAN WE DO THE THING BELOW USING THE GUROBI DATA STRUCTURES?
# I tried this:     K, P = multidict(commodity_paths)       but it didn't work
x = m.addVars(K, commodity_paths[K], vtype=GRB.BINARY, lb=0, name='CommodityPath')
y = m.addVars(A, vtype=GRB.INTEGER, lb=0, name='NumTrucks')
u = m.addVars(Z, D, vtype=GRB.BINARY, lb=0, name='ZipDestinationMatch')

m.modelSense = GRB.MINIMIZE

m.update()

# Constraints
m.addConstrs(
    (x.sum(k, '*') == 1 for k in K), name='OnePathPerCommodity')

m.addConstrs(
    (x[k, p] <= y[a] for k in K for p in P[k] for a in A[p]), name='PathOpen')

m.addConstrs(
    (quicksum(q[k]*x[k, p] for k in K for p in P[k]: a in p) <= 1000*y[a] for a in A), name='ArcCapacity')

m.addConstrs(
    (u.sum(z, '*') == 1 for z in Z), name='DestNodeSelection')

m.addConstrs(
    (u[z, d] >= x[p] for z in Z for d in D for p in P: end[p] == d), name='PathOpenZipDestination')
