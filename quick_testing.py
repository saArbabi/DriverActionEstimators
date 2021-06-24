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
viewer = Viewer(config)


data_config = {
                'future_seq_length':20,
                'history_seq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
history_seqs, future_seqs = data_gen.prep_data()

# %%
future_seqs.shape
history_seqs.shape
history_seqs[0][-1]
future_seqs[0][0]

# %%


plt.plot(history_seqs[100, :, 9])
plt.plot(range(20, 40), future_seqs[100, :, 9])
# %%
index = 0
index_name = {}
feature_data[:, 4] < 20
names = ['episode_id', 'veh_id', 'elapsed_time', 'ego_decision', \
         'leader_speed', 'follower_speed', 'merger_speed', \
         'leader_action ', 'follower_action ', 'merger_action ', \
         'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
         'lane_y', 'leader_exists', 'follower_id']
for item_name in names:
    index_name[index] = item_name
    index += 1

# %%
columns_n = feature_data.shape[-1]

for column in range(columns_n):
    plt.figure()
    plt.title(names[column])
    _ = plt.hist(feature_data[:, column], bins=150)

# %%
        self.indxs = {
                    'speeds':{'leader':0, 'follower':1, 'merger':2},
                    'actions':{'leader':3, 'follower':4, 'merger':5},
                    'relatives':{'follower_leader':[6, 7], 'follower_merger':[8, 9]},
                    'lane_y':10, 'leader_exists':11}
