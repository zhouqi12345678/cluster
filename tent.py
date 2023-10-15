# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:36:29 2023

@author: Administrator
"""
import random
import numpy as np

def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)

def initialize_population(num_populations, num_variables, variable_ranges, r):
    populations = []
    for _ in range(num_populations):
        population = []
        for i in range(num_variables):
            var_min, var_max = variable_ranges[i]
            random_value = random.uniform(0, 1)
            x = tent_map(random_value, r)
            variable = var_min + (var_max - var_min) * x
            population.append(variable)
        populations.append(population)
    return populations

# 设置参数
num_populations = 5  # 种群数量
num_variables = 3  # 变量数量
variable_ranges = [[0.01, 0.1], [3, 10], [0.5, 1]]  # 变量取值范围
r = 2 # 控制参数

# 初始化种群
populations = initialize_population(num_populations, num_variables, variable_ranges, r)

print(np.array(populations))
