# import os
# import pickle
import matplotlib.pyplot as plt
from importlib import reload
import sys
import data_generator
reload(data_generator)
from data_generator import DataGenerator
import numpy as np

import highway
reload(highway)
from highway import Env

import vehicle_handler
reload(vehicle_handler)

from viewer import Viewer

config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
# viewer = Viewer(config)


data_config = {
                'future_seq_length':20,
                'history_seq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
data_arrays = data_gen.prep_data()
future_s, history_s, future_idm_s, future_merger_a, future_follower_a = data_arrays
history_s.shape

# %%
future_merger_a.shape
history_seqs[0][-1]
future_seqs[0][0]

# %%


plt.plot(history_s[100, :, 6])
plt.plot(range(20, 40), future_s[100, :, 6])
# %%
index = 0
index_name = {}
feature_data[:, 4] < 20
names = ['episode_id', 'veh_id', 'elapsed_time', 'ego_decision', \
         'leader_speed', 'follower_speed', 'merger_speed', \
         'leader_action', 'follower_action', 'merger_action', \
         'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
         'lane_y', 'leader_exists', 'follower_id']
for item_name in names:
    index_name[item_name] = index
    index += 1

# %%
keep_these = ['episode_id', 'leader_speed', 'follower_speed', 'merger_speed', \
         'leader_action', 'follower_action', 'merger_action', \
         'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
         'lane_y', 'leader_exists', 'follower_id']


a = [index_name[item] for item in keep_these]

keep_these = ['episode_id', 'merger_action', 'lane_y']
# %%
columns_n = future_s.shape[-1]
future_s.shape
names = ['episode_id', 'follower_speed',
                'fl_delta_v', 'fl_delta_x',
                'fm_delta_v', 'fm_delta_x']
for column in range(columns_n):
    plt.figure()
    plt.title(names[column])
    _ = plt.hist(future_idm_s[:, 0, column], bins=150)

# %%

# %%
