"""
Scripts for MC simulations. The episode and trace seeds are logged for reproducibility:
Metrics are:
Collision counts
RWSE
"""
import os
import numpy as np
import pickle
import sys
sys.path.insert(0, './src')
# os.getcwd()
from importlib import reload
import tensorflow as tf
from envs import merge_mc
reload(merge_mc)
from envs.merge_mc import EnvMergeMC
import matplotlib.pyplot as plt
import copy
import time
config = {'lanes_n':2,
        'lane_width':3.75, # m
        'lane_length':300 # m
        }
trace_n = 1
ima_collection = {}
real_collection = {}
collision_log = []
time_start = time.time()
# trace = 0
# for episode_id in [6]:
env = EnvMergeMC(config)
env.metric_collection_mode = True
# model_name = 'neural_029'
# model_name = 'neural_idm_113'
model_name = 'latent_mlp_03'
data_id = '029'
episodes_n = 13
history_len = 50 # choose this based on the model with longest history
rollout_len = 50

model_vehicle_map = {'neural_idm_113': 'NeuralIDMVehicle',
                    'neural_029': 'NeuralVehicle',
                    'latent_mlp_03': 'LatentMLPVehicle'
                                                    }
if model_vehicle_map[model_name] == 'NeuralVehicle':
    ##############################################
    epoch_count = '20'
    from vehicles.neural import neural_vehicle
    reload(neural_vehicle)
    from vehicles.neural.neural_vehicle import NeuralVehicle
    env.neural_vehicle = NeuralVehicle()
elif model_vehicle_map[model_name] == 'NeuralIDMVehicle':
    ##############################################
    epoch_count = '20'
    from vehicles.neural import neural_idm_vehicle
    reload(neural_idm_vehicle)
    from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
    env.neural_vehicle = NeuralIDMVehicle()
elif model_vehicle_map[model_name] == 'LatentMLPVehicle':
    ##############################################
    epoch_count = '20'
    from vehicles.neural import latent_mlp_vehicle
    reload(latent_mlp_vehicle)
    from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
    env.neural_vehicle = LatentMLPVehicle()

env.neural_vehicle.initialize_agent(
                model_name, epoch_count, data_id)

for episode_id in range(501, 501+episodes_n):
# for episode_id in [6]:
    np.random.seed(episode_id)
    env.transition_time = np.random.randint(\
            history_len, history_len+50) # controller ==> 'neural'
    for trace in range(2):
        env.initialize_env(episode_id)
        tf.random.set_seed(trace) # each trace has a unique seed
        for i in range(0, env.transition_time+rollout_len):
            env.step()
            if env.collision_detected:
                collision_id = f'{episode_id}_{trace}'
                if collision_id not in collision_log:
                    collision_log.append(collision_id)
                    print('collision_detected')

        for veh_id, data_log in env.ima_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [episode_id, veh_id, trace]
            if not episode_id in ima_collection:
                ima_collection[episode_id] = {}
            if not veh_id in ima_collection[episode_id]:
                ima_collection[episode_id][veh_id] = [data_log]
            else:
                # in case there are multiple traces per episode
                ima_collection[episode_id][veh_id].append(data_log)

    for veh_id, data_log in env.real_mc_log.items():
        for step_log in data_log:
            step_log[1:1] = [episode_id, veh_id, trace]
        if not episode_id in real_collection:
            real_collection[episode_id] = {}
        if not veh_id in real_collection[episode_id]:
            real_collection[episode_id][veh_id] = [data_log]
        else:
            # in case there are multiple traces per episode
            real_collection[episode_id][veh_id].append(data_log)
time_end = time.time()

print((time_end-time_start)/60)

# %%
"""
Save recordings
"""
exp_dir = './src/models/experiments/'+model_name+'/eval'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

if not os.path.exists(exp_dir+'/real_collection.pickle'):
    with open(exp_dir+'/real_collection.pickle', 'wb') as handle:
        pickle.dump(real_collection, handle)

    with open(exp_dir+'/ima_collection.pickle', 'wb') as handle:
        pickle.dump(ima_collection, handle)
    if collision_log:
        with open(exp_dir+'/collision_log.pickle', 'wb') as handle:
            pickle.dump(collision_log, handle)
else:
    print('Eval folder already exists')
