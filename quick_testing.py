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
feature_data = data_gen.prep_data()
feature_data.shape


# %%
plt.plot(feature_data[:, 1])
plt.plot(feature_data[:, -1])
plt.plot(feature_data[:, 3])
plt.plot(feature_data[:, 6])
plt.plot(feature_data[:, 8])
plt.plot(feature_data[:, 9])
feature_data[feature_data[:, 0] == 0][:, ]
lengths = [len(feature_data[i]) for i in range(241)]
plt.plot(lengths)



# %%
feature_data[:, 4]
feature_data[:, 4]
#
np.count_nonzero(np.isnan(feature_data[:, :]))
# %%
sample_traj = feature_data[feature_data[:, 0] == 0]
feature_data[feature_data[:, 0] == 0].shape
plt.plot(sample_traj[:, 3])
plt.plot(sample_traj[:, -1])
plt.plot(sample_traj[:, 5])
plt.plot(sample_traj[:, 14])

plt.plot(sample_traj[:, -1])
plt.plot(sample_traj[:, 7])
plt.plot(sample_traj[:, 14])
plt.plot(sample_traj[:, 8])
plt.plot(sample_traj[:, 15])
# %%
columns_n = feature_data.shape[-1]
for column in range(columns_n):
    plt.figure()
    _ = plt.hist(feature_data[:, column], bins=150)

# %%
        self.indxs = {
                    'speeds':{'leader':0, 'follower':1, 'merger':2},
                    'actions':{'leader':3, 'follower':4, 'merger':5},
                    'relatives':{'follower_leader':[6, 7], 'follower_merger':[8, 9]},
                    'lane_y':10, 'leader_exists':11}


np.unique(feature_data[:, 0])
np.unique(feature_data[:, 0])
feature_data[:, -1].shape

feature_data[0].shape
feature_data[0]
feature_data[0]
feature_data[feature_data[:, 0]==0].shape
feature_data[feature_data[:, 0]==5][:, -1]


# %%
