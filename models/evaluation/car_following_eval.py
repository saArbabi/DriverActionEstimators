"""
Collisions
Hard brakes
RWSE
"""
import os
from highway import EnvLaneKeepMC
import matplotlib.pyplot as plt
import copy

config = {'lanes_n':1,
        'lane_width':3.75, # m
        'lane_length':400 # m
        }

env = EnvLaneKeepMC(config)

real_veh_glob_x = {}
ima_veh_glob_x = {}

for i in range(30):

    print(env.time_step)
    for veh_real, veh_ima in zip(env.real_vehicles, env.ima_vehicles):
        if not veh_real.id in real_veh_glob_x:
            real_veh_glob_x[veh_real.id] = []
            ima_veh_glob_x[veh_ima.id] = []
        ima_veh_glob_x[veh_real.id].append

    env.step()
    # print(env.ima_vehicles[0].vehicle_type)
    # print(env.ima_vehicles[0].act_long)
    # print(env.ima_vehicles[0].speed)
    # print(env.ima_vehicles[0].id)
    # print(env.time_step)

# env.ima_vehicles[1].__dict__.items()
