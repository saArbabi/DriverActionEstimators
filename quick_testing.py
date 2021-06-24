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
from viewer import Viewer

config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
viewer = Viewer(config)
"""
test to see nicer dummy
also accelerations
bad accelerations. whats up there?
"""
"""
data_config = {
                'future_seq_length':20,
                'history_seq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
feature_data = data_gen.prep_data()
# raw_recordings['info'][6]
# np.array(env.recordings)[:, -1]

# %%
feature_data[:, 4]
feature_data[:, 4]
#
np.count_nonzero(np.isnan(feature_data[:, :]))
# %%
sample_traj = feature_data[feature_data[:, 0] == 0]
feature_data[feature_data[:, 0] == 0].shape
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


"""
# %%
def main():
    # for i in range(100):
    while True:
        decision = input()
        if decision == 'n':
            sys.exit()
        try:
            viewer.focus_on_this_vehicle = int(decision)
        except:
            pass

        env.step()
        viewer.render(env.vehicles)
        # print(env.elapsed_time)


if __name__=='__main__':
    main()

# %%
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers
    for i_ in range(1000):
        env.step()

    def animation_frame(i):
        viewer.render(env.vehicles)
        env.step()
        # return line,

    animation = FuncAnimation(viewer.fig, func=animation_frame, \
                              frames=range(100), interval=1000)


    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=25, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('sim_example.mp4', writer, dpi=250)




# get_animation()
# plt.show()
# %%
