"""
Collisions
Hard brakes
RWSE
"""
import os
import numpy as np

# os.chdir('../../')
# print('directory: ' + os.getcwd())
# directory: C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim
from importlib import reload
from vehicle_handler import VehicleHandler
from vehicles import neural_vehicles
import tensorflow as tf
reload(neural_vehicles)

import highway
reload(highway)
from highway import EnvMC
import matplotlib.pyplot as plt
import copy

config = {'lanes_n':6,
        'lane_width':3.75, # m
        'lane_length':400 # m
        }

trace_n = 10
glob_x_collection = np.zeros([1, trace_n, 50]) # [epis_n, trace_n, steps_n]
ima_act_collection = np.zeros([1, trace_n, 50])
real_act_collection = np.zeros([1, 1, 50])

for trace in range(trace_n):
    env = EnvMC(config)
    np.random.seed(2005)
    tf.random.set_seed(trace)
    for i in range(220):
        # print(env.time_step)
        env.step()
        if 19 in env.real_mc_log:
            if len(env.real_mc_log[19]['glob_x']) == 50:
                # forward simulation for 20s
                glob_x_collection[0, trace, :] = np.array(env.real_mc_log[19]['glob_x'])
                ima_act_collection[0, trace, :] = np.array(env.ima_mc_log[19]['action'])
                if trace == 0:
                    real_act_collection[0, trace, :] = np.array(env.real_mc_log[19]['action'])
                break
# %%
plt.plot(real_act_collection[0, 0, :], color='red')
for trace in range(trace_n):
    plt.plot(ima_act_collection[0, trace, :], color='grey')

# %%
delta_x = np.array(env.real_mc_log[19]['glob_x']) -\
                        np.array(env.ima_mc_log[19]['glob_x'])
plt.plot(delta_x)

# %%

plt.plot(env.ima_mc_log[19]['glob_x'])
plt.plot(env.real_mc_log[19]['glob_x'])
# %%
plt.plot(env.ima_mc_log[19]['action'])
plt.plot(env.real_mc_log[19]['action'])
# %%

env.ima_mc_log[19]

plt.plot(att_real, color='red')
mveh_ex = np.array(env.ima_mc_log[19]['m_veh_exists'][25:])
plt.plot(mveh_ex, color='blue', linestyle='--')
# %%
"""
TODO:
[] ensure history is diverse - state arrays rather than points
[] ensrure you do not val on training set
[] somehow detect collisions

Note: in all this plan a few steps ahead.
end episdoe if:
20s gathered for all cars
there is a collision
---
There are two loops (both needed for rwse):
One for generating traces for the same episodes
One for different episodes
"""

att_ima = np.array(env.ima_mc_log[19]['att_log'][25:])
att_real = np.array(env.real_mc_log[19]['att_log'][25:])
arrays = [att_ima[:, i, 0, 0] for i in range(2)]
att_ima = np.stack(arrays)
plt.plot(att_real, color='red')
plt.plot(att_ima[0, :], color='grey')
print(1, 2)

# %%
# plt.plot(env.real_mc_log[19]['att_log'])
# plt.plot(env.ima_mc_log[19]['att_log'])
samples_n = 20
att_ima = np.array(env.ima_mc_log[19]['m_veh_exists'][25:])
att_ima = np.array(env.ima_mc_log[19]['att_log'][25:])
att_real = np.array(env.real_mc_log[19]['att_log'][25:])
arrays = [att_ima[:, i, 0, 0] for i in range(samples_n)]
att_ima = np.stack(arrays)
plt.plot(att_real, color='red')
mveh_ex = np.array(env.ima_mc_log[19]['m_veh_exists'][25:])
plt.plot(mveh_ex, color='blue', linestyle='--')
for i in range(samples_n):
    plt.plot(att_ima[i, :], color='grey')
# %%
