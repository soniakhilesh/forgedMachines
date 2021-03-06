import csv
import pandas as pd
import numpy as np
import ast
from itertools import product
from gurobipy import *
from network import Network
from od_demand_generator import demand_generator
from weightedzips import ZipCoords
from network import dist
from output import write_arcs_to_csv, write_zd_to_csv

datapath = "C:/Users/kbada/PycharmProjects/forged-by-machines/"


def create_stochastic_network():
    """
    no information specific to a scenario: a general network structure
    create a network with all scenario commodities
    essentially add multiple commodities for same o-z pair: might wanna test nefore creating a model
    :return:
    """

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
    zip_coords = ZipCoords()
    for zip_name in zips:
        # drop empty space: took 4 hours to find this bug
        zip_name = zip_name.replace(" ", "")
        network.add_zip(zip_name, zip_coords.get_lat(zip_name), zip_coords.get_lon(zip_name))  # temporarily using 0,0

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


def warmstart_model(network, y, u):
    # Import solution from mean-value deterministic model
    start_trucks = pd.read_csv(datapath + "warm-start-arc-trucks.csv")
    dest_assignment = pd.read_csv(datapath + "warm-start-o-z.csv")
    # Set y values using trucks dataframe
    start_trucks.set_index(["Arc"], inplace=True)
    for a in network.get_arcs():
        y[a].start = start_trucks.loc[a.origin.name+"->" + a.dest.name, 'Number of Trucks']
    # Set u values using dest_assignment dataframe
    for index, row in dest_assignment.iterrows():
        z = str(row['ZIP'])
        d = row['Assigned Destination Node']
        u[network.get_zip(z), network.get_node(d)].start = 1


def multi_objective(objectives: list, ext_model):
    """

    :param objectives: list of gurobi linear expressions [obj_cost, obj_load, obj_distance]
    :param ext_model: gurobi model
    :return:
    """
    obj_cost, obj_distance, obj_load = objectives[0], objectives[1], objectives[2]
    ext_model.setParam("TimeLimit", 300)
    ext_model.setParam("MIPGap", 0.04)
    solution_values = {}
    epsilon_values = np.arange(start=0, stop=0.2, step=0.05)
    tolerance = 0.000001


    # Solve for cost without any additional constraint
    ext_model.setObjective(obj_cost)
    ext_model.optimize()
    # solution_values['cost'][0] = ext_model.getObjective().getValue()
    best_cost = ext_model.getObjective().getValue()
    # Get corresponding best (most equitable) load just so we have a reference
    # ext_model.addConstr(obj_cost <= best_cost + tolerance)
    # ext_model.setObjective(obj_load)
    # ext_model.optimize()
    # best_load_1 = ext_model.getObjective().getValue()


    # Generating Pareto solution "blanket" for cost as first objective
    for i in epsilon_values:
        solution_values[i] = {}
        # Optimize for distance allowing for some deterioration in cost
        ext_model.setObjective(obj_distance)
        cost_rhs = (1 + i) * best_cost + tolerance
        ext_model.addConstr(obj_cost <= cost_rhs)
        ext_model.optimize()
        best_dist = ext_model.getObjective().getValue()
        # Store distance obj. value for corresponding epsilon
        solution_values[i]['distance'] = best_dist
        # Get best corresponding cost (might be slightly different)
        ext_model.addConstr(obj_distance <= best_dist + tolerance)
        ext_model.setObjective(obj_cost)
        ext_model.optimize()
        best_cost = ext_model.getObjective().getValue()
        # Store cost obj. value for corresponding epsilon
        solution_values[i]['cost'] = best_cost
        # Optimize for load without allowing deterioration
        cost_rhs = best_cost + tolerance
        ext_model.addConstr(obj_cost <= cost_rhs)
        ext_model.setObjective(obj_load)
        ext_model.optimize()
        # Store load obj. value for corresponding epsilon
        solution_values[i]['load'] = ext_model.getObjective().getValue()
    print(solution_values)

        # for j in epsilon_values:
        #
        #     ext_model.setObjective(obj_load)
        #     dist_rhs = (1 + epsilon_values[i]) * best_dist + tolerance
        #     ext_model.addConstr(obj_distance <= dist_rhs)
        #     ext_model.optimize()
        #     best_load = ext_model.getObjective().getValue()
        #     solution_values['load'][index_i] = best_dist


    # fix epsilon for load
    # epsilon_values['load'] = 5000  # we can directly change it depending on how much difference we want to allow maybe?
    # ext_model.addConstr(obj_load<=epsilon_values['load'])



    # write solution to csvs
    # write_arcs_to_csv(y,network,"TruckAssignments.csv")
    # write_zd_to_csv(u,network,"CustomerAssignments.csv")





def create_extensive_form_model(network: Network, scenarios: list, demand_data):
    """
     demand_data: dict
    :param scenarios: [1,2,....N]
    :param demand_data: dict with tuple keys of the form (scenario_num, commodity_name)
    :param network: Network object: consists of all commodities across all the scenarios
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
    r = m.addVars(network.get_commodities(), scenarios, vtype=GRB.CONTINUOUS, lb=0, name='DistanceTraveledByCommodity')
    max_load = m.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name='MaxLoad')
    min_load = m.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name='MinLoad')
    # Decision variables-second stage end

    m.update()
    m.modelSense = GRB.MINIMIZE
    #
    # first stage constraints--begin
    m.addConstrs(
        (u.sum(z, '*') == 1 for z in network.get_zips()), name='DestNodeSelection')
    # first stage constraints--end
    # second stage constraints--begin
    m.addConstrs(
        (r[k, s] >= p.distance * x[k, p, s] +
         dist(k.dest.lat, d.lat, k.dest.lon, d.lon) * u[k.dest, d]
         for k in network.get_commodities() for d in network.get_dest_nodes()
         for p in network.get_commodity_dest_node_paths(k, d) for s in scenarios),
        name='DistanceDestinationNodeToZip')
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

    # m.addConstrs(
    #     (u[k.dest, d] >= sum(x[k, p, s] for p in network.get_commodity_dest_node_paths(k, d))
    #      for k in network.get_commodities() for d in network.get_dest_nodes() for s in scenarios
    #      ), name='PathOpenZipDestination')

    # m.addConstrs(
    #     (u[k.dest, d] >= x[k, p, s]for k in network.get_commodities() for d in network.get_dest_nodes()
    #      for p in network.get_commodity_dest_node_paths(k, d) for s in scenarios
    #      ), name='PathOpenZipDestination')

    m.addConstrs(u[z, d] >= x[p.commodity, p, s] for z in network.get_zips() for d in network.get_dest_nodes()
                 for p in network.get_dest_node_zip_paths(d, z) for s in scenarios)

    # next two constraint needs to be updated in each batch run: update just coefficients
    m.addConstrs((min_load[s] <= sum(u[k.dest, d] * demand_data[s, k] for k in network.get_commodities())
                  for d in network.get_dest_nodes() for s in scenarios), name="MinLoad")
    m.addConstrs((max_load[s] >= sum(u[k.dest, d] * demand_data[s, k] for k in network.get_commodities())
                  for d in network.get_dest_nodes() for s in scenarios), name="MaxLoad")

    # second stage constraints--end

    obj_cost = LinExpr(quicksum((100 + 2 * a.distance) * y[a] for a in network.get_arcs())) + (
                1 / len(scenarios)) * quicksum(
        1000 * demand_data[s, k] * unfulfilled[k, s] for k in network.get_commodities() for s in
        scenarios)
    obj_load = (1 / len(scenarios)) * quicksum(max_load[s] - min_load[s] for s in scenarios)
    obj_distance = quicksum((1 / len(scenarios)*r[k, s] for k in network.get_commodities() for s in scenarios))
    use_cost = False  # 11322, 12939
    use_load = False  # 100, 2000
    use_distance = False  # 570,690
    # if use_cost:
    #     m.setObjective(obj_cost)
    #     m.addConstr(obj_load<=2500)
    #     m.addConstr(obj_distance<=700)
    # elif use_load:
    #     m.setObjective(obj_load)
    #     m.addConstr(obj_cost<=2*11322)
    #     m.addConstr(obj_distance<=690)
    # elif use_distance:
    #     m.setObjective(obj_distance)
    #     m.addConstr(obj_cost<=1.2*12939) #2*11322
    #     m.addConstr(obj_load<=2000)
    # else:
    #     print("No objective specified")

    # m.setObjective(quicksum((100 + 2 * a.distance) * y[a] for a in network.get_arcs()) +
    #                (1 / len(scenarios)) * quicksum(
    #     1000 * demand_data[s, k] * unfulfilled[k, s] for k in network.get_commodities() for s in
    #     scenarios)
    #                + (1 / len(scenarios)) * quicksum(max_load[s] - min_load[s] for s in scenarios) +
    #                 quicksum(r[z] for z in network.get_zips())
    #                )
    # m.setParam("TimeLimit", 400)
    # m.setParam("MIPGap", 0.04)

    # adding some heuristics constraint

    # a zip should be connected to one of the closes 3 dest nodes
    # for z in network.get_zips():
    #     close_dest_nodes = network.get_closest_dest_nodes(z,4)
    #     for d in network.get_dest_nodes():
    #         if d not in close_dest_nodes:
    #             m.addConstr(u[z,d]==0)

    m.update()

    # Warmstart model
    warmstart_model(network, y, u)

    """
    a=network.get_arcs()[0]
    s=1
    p=network.get_arc_paths(a)[0]
    m.chgCoeff(m.getConstrByName("ArcCapacity[{},{}]".format(a,s)), x[p.commodity, p, s], 60)
    Don't query again and again: query oonce and store the variables
    Modifying objective
    m.getVarByName("FractionUnfulfilledScenario[{},{}]".format(k,s)).setAttr("obj",1000*demand_data[s,k)
    """
    return [obj_cost, obj_load, obj_distance], m


def run_saa(network, batch_num, scen_num):
    scenario_list = [i for i in range(scen_num)]
    # declare dictionaries for storing necessary objects to update model
    unfulfilled_vars = {}
    x_vars = {}
    u_vars = {}
    y_vars={}
    arc_capacity_constraints = {}
    min_load_con = {}
    max_load_con = {}
    for i in range(batch_num):
        # generate demand data
        demand_data = {}
        for k in network.get_commodities():
            for s in scenario_list:
                temp_demand = max(int(demand_generator(k.origin_node.name, int(k.dest.name))), 0)
                demand_data[(s, k)] = temp_demand
        # build or update model
        if i == 0:
            # build model from scratch
            objectives, ext_model = create_extensive_form_model(network, scenario_list, demand_data)
            # get necessary variables and constraints which need to be updated
            for s in scenario_list:
                for k in network.get_commodities():
                    unfulfilled_vars[(k, s)] = ext_model.getVarByName("FractionUnfulfilledScenario[{},{}]".format(k, s))
                for a in network.get_arcs():
                    arc_capacity_constraints[(a, s)] = ext_model.getConstrByName("ArcCapacity[{},{}]".format(a, s))
                for k in network.get_commodities():
                    for p in network.get_commodity_paths(k):
                        x_vars[(k, p, s)] = ext_model.getVarByName("CommodityPathScenario[{},{},{}]".format(k, p, s))
                for d in network.get_dest_nodes():
                    min_load_con[d, s] = ext_model.getConstrByName("MinLoad[{},{}]".format(d, s))
                    max_load_con[d, s] = ext_model.getConstrByName("MaxLoad[{},{}]".format(d, s))

            for zip_name in network.get_zips():
                for destination_node in network.get_dest_nodes():
                    u_vars[(zip_name, destination_node)] = ext_model.getVarByName("ZipDestinationMatch[{},{}]".format(
                        zip_name, destination_node))
            for arc in network.get_arcs():
                y_vars[arc] = ext_model.getVarByName("NumTrucks[{}]".format(arc))

            # use multi objective stuff
            multi_objective(objectives, ext_model)

        else:
            # change coefficients of model with the new demand data
            # update arc capacity constraint
            for a in network.get_arcs():
                for p in network.get_arc_paths(a):
                    for s in scenario_list:
                        ext_model.chgCoeff(arc_capacity_constraints[a, s], x_vars[p.commodity, p, s],
                                           demand_data[s, p.commodity])
            # update load constraints
            for s in scenario_list:
                for d in network.get_dest_nodes():
                    for k in network.get_commodities():
                        ext_model.chgCoeff(min_load_con[d, s], u_vars[k.dest, d], -demand_data[s, k])  # coeff are neg
                        ext_model.chgCoeff(max_load_con[d, s], u_vars[k.dest, d], -demand_data[s, k])
            # update objective
            for k in network.get_commodities():
                for s in scenario_list:
                    unfulfilled_vars[k, s].setAttr("obj", 1000 * demand_data[s, k]) # Change in linear expression
            ext_model.update()


network = create_stochastic_network()

run_saa(network, 1, 1)
