import csv
import pandas as pd
import numpy as np
import ast
from itertools import product
from gurobipy import *
from network import Network


# do I want a single network with all scenario commodities or a separate network for each scenario?
def create_stochastic_network():
    """
    no information specific to a scenario: a general network structure
    create a network with all scenario commodities
    essentially add multiple commodities for same o-z pair: might wanna test nefore creating a model
    :return:
    """

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
    # Add zips to network: zip long and zip lat should be weighted and fixed: impossible to vary in each scenario as
    # we are simulating demand zip directly: hence no customer information in scenarios
    # Maybe we can evaluate zip location in each scenario and take average
    with open(datapath + "ZIP CODE.csv", newline='') as f:
        reader = csv.reader(f)
        zips = list(reader)
    zips = zips[0]
    for zip_name in zips:
        # drop empty space: took 4 hours to find this bug
        zip_name = zip_name.replace(" ", "")
        network.add_zip(zip_name, 0.0, 0.0)  # temporarily using 0,0

    # add commodities: for stochastic -add all o-z pairs without setting quantity
    for origin_node in network.get_origin_nodes():
        for zip_name in network.get_zips():
            network.add_commodity(origin_node, zip_name)

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


def create_extensive_form_model(network: Network, scenarios: list, demand_data):
    """
     demand_data: dict
    :param scenarios: [1,2,....N]
    :param demand_data: dict with tuple keys of the form (scenario_num, commodity_name)
    :param network: Network object: comnsists of all commodities across all the scenarios
    :return:
    """
    # create path variables for each commodity
    m = Model("extensive")
    # Decision variables-first stage begin
    y = m.addVars(network.get_arcs(), vtype=GRB.INTEGER, lb=0, name='NumTrucks')
    u = m.addVars(network.get_zips(), network.get_dest_nodes(), vtype=GRB.BINARY, lb=0, name='ZipDestinationMatch')

    tuplelist_comm_path_scenario = [(k, p, s) for k in network.get_commodities()
                                    for p in network.get_commodity_paths(k) for s in scenarios]
    # Decision variables-first stage end
    # Decision variables-second stage begin
    x = m.addVars(tuplelist_comm_path_scenario, vtype=GRB.CONTINUOUS, lb=0, ub=1,
                  name='CommodityPathScenario')
    unfulfilled = m.addVars(network.get_commodities(), scenarios, vtype=GRB.CONTINUOUS,
                            lb=0, ub=1, name='FractionUnfulfilledScenario')
    r = m.addVars(network.get_zips(), vtype=GRB.CONTINUOUS, lb=0, name='RemainingDistanceZipToCustomer')
    max_load = m.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name='MaxLoad')
    min_load = m.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name='MinLoad')
    # Decision variables-second stage end

    m.update()
    m.modelSense = GRB.MINIMIZE
    #
    # first stage constraints--begin
    m.addConstrs(
        (u.sum(z, '*') == 1 for z in network.get_zips()), name='DestNodeSelection')
    m.addConstrs(
        (r[z] >= dist(z.lat, d.lat, z.lon, d.lon) * u[z, d]
         for z in network.get_zips() for d in network.get_dest_nodes()),
        name='DistanceDestinationNodeToZip')
    # first stage constraints--end
    # second stage constraints--begin
    m.addConstrs(
        (x.sum(k, '*', s) + unfulfilled[(k, s)] == 1 for k in network.get_commodities()
         for s in scenarios), name='CommodityFulfillment')
    m.addConstrs(
        (x[k, p, s] <= y[a] for s in scenarios for k in network.get_commodities()
         for p in network.get_commodity_paths(k) for a in p.arcs), name='PathOpen')

    # next constraint needs to be updated in each batch run: update just coefficients
    m.addConstrs((quicksum(demand_data[(s, p.commodity)] * x[p.commodity, p, s]
                           for p in network.get_arc_paths(a)) <= 1000 * y[a] for a in network.get_arcs() for s in
                  scenarios), name='ArcCapacity')

    m.addConstrs(
        (u[k.dest, d] >= sum(x[k, p, s] for p in network.get_commodity_dest_node_paths(k, d))
         for k in network.get_commodities() for d in network.get_dest_nodes() for s in scenarios
         ), name='PathOpenZipDestination')

    # next two constraint needs to be updated in each batch run: update just coefficients
    m.addConstrs((min_load[s] <= sum(u[k.dest, d] * demand_data[s, k] for k in network.get_commodities())
                  for d in network.get_dest_nodes() for s in scenarios), name="MinLoad")
    m.addConstrs((max_load[s] >= sum(u[k.dest, d] * demand_data[s, k] for k in network.get_commodities())
                  for d in network.get_dest_nodes() for s in scenarios), name="MaxLoad")
    # second stage constraints--end
    # add objective
    lambda1 = 1
    lambda2 = 2
    m.setObjective(quicksum((100 + 2 * a.distance) * y[a] for a in network.get_arcs()) +
                   quicksum(1000 * demand_data[s, k] * unfulfilled[k, s] for k in network.get_commodities() for s in
                            scenarios)
                   + quicksum(lambda1 * (max_load[s] - min_load[s]) for s in scenarios) +
                   lambda2 * quicksum(r[z] for z in network.get_zips())
                   )
    # m.setParam("TimeLimit", 50)
    # m.setParam("MIPGap", 0.01)
    m.update()

    """
    a=network.get_arcs()[0]
    s=1
    p=network.get_arc_paths(a)[0]
    m.chgCoeff(m.getConstrByName("ArcCapacity[{},{}]".format(a,s)), x[p.commodity, p, s], 60)
    Don't query again and again: query oonce and store the variables
    Modifying objective
    m.getVarByName("FractionUnfulfilledScenario[{},{}]".format(k,s)).setAttr("obj",1000*demand_data[s,k)
    """
    return m


def run_saa(network, batch_num, scen_num):
    scenario_list = [i for i in range(scen_num)]
    # declare dictionaries for storing necessary objects to update model
    unfulfilled_vars = {}
    x_vars = {}
    u_vars = {}
    arc_capacity_constraints = {}
    min_load_con = {}
    max_load_con = {}
    for i in range(batch_num):
        # generate demand data
        demand_data = {}
        for s in scenario_list:
            for k in n.get_commodities():
                pair = (s, k)
                demand_data[pair] = 100
        # build or update model
        if i == 0:
            # build model from scratch
            ext_model = create_extensive_form_model(network, scenario_list, demand_data)
            # get necessary variables and constraints which need to be updated
            for s in scenario_list:
                for k in network.get_commodities():
                    unfulfilled_vars[(k, s)] = ext_model.getVarByName("FractionUnfulfilledScenario[{},{}]".format(k, s))
                for a in network.get_arcs():
                    arc_capacity_constraints[(a, s)] = ext_model.getConstrByName("ArcCapacity[{},{}]".format(a, s))
                for k in network.get_commodities():
                    for p in network.get_commodity_paths(k):
                        x_vars[(k, p, s)] = ext_model.getVarByName("CommodityPathScenario[{},{},{}]".format(k, p, s))
                min_load_con[s] = ext_model.getConstrByName("MinLoad[{}]".format(s))
                max_load_con[s] = ext_model.getConstrByName("MaxLoad[{}]".format(s))

            for zip_name in network.get_zips():
                for destination_node in network.get_dest_nodes():
                    u_vars[(zip_name, destination_node)] = ext_model.getVarByName("ZipDestinationMatch[{},{}]".format(
                        zip_name, destination_node))

        else:
            # change coefficients of model with the new demand data
            # update arc capacity constraint
            for a in network.get_arcs():
                for p in network.get_arc_paths(a):
                    for s in scenario_list:
                        ext_model.chgCoeff(arc_capacity_constraints[a, s], x_vars[p.commodity, p, s],
                                           demand_data[s, p.k])
            # update load constraints
            for s in scenario_list:
                for d in network.get_dest_nodes():
                    for k in network.get_commodities():
                        ext_model.chgCoeff(min_load_con[s], u_vars[k.dest, d], demand_data[s, k])
                        ext_model.chgCoeff(max_load_con[s], u_vars[k.dest, d], demand_data[s, k])
            # update objective
            for k in network.get_commodities():
                for s in scenario_list:
                    unfulfilled_vars[k, s].setAttr("obj", 1000 * demand_data[s, k])
            ext_model.update()

        # optimize
        ext_model.optimize()
        # store solution and objective value

n = create_stochastic_network()
run_saa(n, 2, 2)
