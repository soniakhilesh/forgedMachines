import ast
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon
from random import uniform

distributions = pd.read_csv("od_distributions.csv")
min_factor = 0.5
max_factor = 1.5


# Function to generate demand based on fitted probability distribution, parameters
# and given origin and zipcode
def demand_generator(origin, zipcode):
    filter = (distributions['Origin'] == origin) & (distributions['ZIP'] == zipcode)
    prob_dist_info = distributions[filter]
    prob_dist_info = prob_dist_info.reset_index()
    min_demand = prob_dist_info['Minimum'].item()
    max_demand = prob_dist_info['Maximum'].item()
    # Normal distribution
    if prob_dist_info['Distribution'].item() == 'norm':
        mean = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
        st_dev = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
        sample = norm.ppf(uniform(0, 1), mean, st_dev)
        while sample < min_demand*min_factor or sample > max_demand*max_factor:
            sample = norm.ppf(uniform(0, 1), mean, st_dev)
        return sample
    # elif prob_dist_info['Distribution'].item() == 'lognorm':
    #     shape = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
    #     loc = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
    #     scale = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[2]
    #     return lognorm.ppf(uniform(0, 1), shape, loc, scale)
    # Exponential distribution
    elif prob_dist_info['Distribution'].item() == 'expon':
        loc = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
        scale = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
        sample = expon.ppf(uniform(0, 1), loc, scale)
        while sample < min_demand*min_factor or sample > max_demand*max_factor:
            sample = expon.ppf(uniform(0, 1), loc, scale)
        return sample
    return


# Tests
# l = []
# for i in range(1000):
#     l.append(demand_generator('ONode2', 90032))
# print("Minimum:\t", min(l))
# print("Average:\t", sum(l)/len(l))
# print("Maximum:\t", max(l))
# print(demand_generator('ONode3', 91214))    # lognormal
# print(demand_generator('ONode2', 91214))    # exponential
# print(demand_generator('ONode3', 90710))    # norm
