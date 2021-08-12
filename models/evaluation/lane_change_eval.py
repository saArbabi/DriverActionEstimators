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
import highway
reload(highway)
from highway import EnvMC
import matplotlib.pyplot as plt
import copy

config = {'lanes_n':6,
        'lane_width':3.75, # m
        'lane_length':400 # m
        }
env = EnvMC(config)
for i in range(170):
    # print(env.time_step)
    env.step()

# %%


att_ima = np.array(env.ima_mc_log[19]['att_log'][25:])
att_real = np.array(env.real_mc_log[19]['att_log'][25:])
arrays = [att_ima[:, i, 0, 0] for i in range(2)]
att_ima = np.stack(arrays)
plt.plot(att_real, color='red')
plt.plot(att_ima[0, :], color='grey')

# %%
# plt.plot(env.real_mc_log[19]['att_log'])
# plt.plot(env.ima_mc_log[19]['att_log'])
samples_n = 20
att_ima = np.array(env.ima_mc_log[19]['att_log'][25:])
att_real = np.array(env.real_mc_log[19]['att_log'][25:])
arrays = [att_ima[:, i, 0, 0] for i in range(samples_n)]
att_ima = np.stack(arrays)
plt.plot(att_real, color='red')
for i in range(samples_n):
    plt.plot(att_ima[i, :], color='grey')
