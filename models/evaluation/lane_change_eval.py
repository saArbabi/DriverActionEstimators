"""
Scripts for MC simulations. The episode and trace seeds are logged for reproducibility:
Metrics are:
Collision counts

RWSE
"""
import os
import numpy as np

import pickle
# os.chdir('../../')
# print('directory: ' + os.getcwd())
# directory: C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim
from importlib import reload
from vehicle_handler import VehicleHandler
import tensorflow as tf
from vehicles import neural_vehicles
reload(neural_vehicles)
from vehicles.neural_vehicles import NeuralIDMVehicle, NeurLatentVehicle, NeurLatentShortVehicle

import highway
reload(highway)
from highway import EnvMC
import matplotlib.pyplot as plt
import copy
import time
config = {'lanes_n':6,
        'lane_width':3.75, # m
        'lane_length':800 # m
        }

trace_n = 1
ima_collection = {}
collision_log = []
time_start = time.time()
for trace in range(trace_n):
    env = EnvMC(config)
    env.metric_collection_mode = True

    # env.neural_vehicle = MLPVehicle()
    env.neural_vehicle = NeuralIDMVehicle()
    # env.neural_vehicle = NeurLatentVehicle()
    # env.neural_vehicle = NeurLatentShortVehicle()
    # env.neural_vehicle = LSTMVehicle()
    np.random.seed(0) # ensures environment remains the same
    tf.random.set_seed(trace) # each trace has a unique seed
    # tf.random.set_seed(2021)
    for i in range(420):
        env.step()

    for veh_ima in env.ima_vehicles:
        # print(env.time_step)
        if veh_ima.collision_detected and veh_ima.time_lapse < 200:
            print('Oh no, collision detected within 20s')
            info = [veh_ima.id, veh_ima.time_lapse, trace]
            collision_log.append(info)

    for veh_id, data_log in env.ima_mc_log.items():
        for step_log in data_log:
            step_log[1:1] = [veh_id, trace]
        if veh_id not in ima_collection:
            ima_collection[veh_id] = [data_log]
        else:
            ima_collection[veh_id].append(data_log)

real_collection = {}
for veh_id, data_log in env.real_mc_log.items():
    for step_log in data_log:
        step_log[1:1] = [veh_id, trace]
    if veh_id in real_collection:
        real_collection[veh_id].append(data_log)
    else:
        real_collection[veh_id] = data_log
time_end = time.time()


print((time_end-time_start)/60)


# %%
"""
Save recordings
"""
# model_name = 'h_lat_f_act'
model_name = 'h_lat_f_idm_act'
# model_name = 'h_lat_act'

directory = './publication_results/'+model_name
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists('./publication_results/'+model_name+'/real_collection.pickle'):
    with open('./publication_results/'+model_name+'/real_collection.pickle', 'wb') as handle:
        pickle.dump(real_collection, handle)

    with open('./publication_results/'+model_name+'/ima_collection.pickle', 'wb') as handle:
        pickle.dump(ima_collection, handle)
    if collision_log:
        with open('./publication_results/'+model_name+'/collision_log.pickle', 'wb') as handle:
            pickle.dump(collision_log, handle)
