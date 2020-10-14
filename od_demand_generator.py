import ast
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon
from random import uniform

distributions = pd.read_csv("od_distributions.csv")


# Function to generate demand based on fitted probability distribution, parameters
# and given origin and zipcode
def demand_generator(origin, zipcode):
    filter = (distributions['Origin'] == origin) & (distributions['ZIP'] == zipcode)
    prob_dist_info = distributions[filter]
    prob_dist_info = prob_dist_info.reset_index()
    if prob_dist_info['Distribution'].item() == 'norm':
        mean = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
        st_dev = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
        return norm.ppf(uniform(0, 1), mean, st_dev)
    elif prob_dist_info['Distribution'].item() == 'lognorm':
        shape = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
        loc = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
        scale = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[2]
        return lognorm.ppf(uniform(0, 1), shape, loc, scale)
    elif prob_dist_info['Distribution'].item() == 'expon':
        loc = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[0]
        scale = ast.literal_eval(prob_dist_info.loc[0, 'Parameters'])[1]
        return expon.ppf(uniform(0, 1), loc, scale)
    return


# Tests
# print(demand_generator('ONode3', 91214))    # lognormal
# print(demand_generator('ONode2', 91214))    # exponential
# print(demand_generator('ONode3', 90710))    # norm
