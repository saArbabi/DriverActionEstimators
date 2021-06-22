# import os
# import pickle
# import sys
from highway import Env
from viewer import Viewer
# from factory.vehicles import NeurIDM
# from importlib import reload
#
# from factory import vehicles
# reload(vehicles)


config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
viewer = Viewer(config)
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
a = np.array([False, True])
np.count_nonzero(feature_data == np.nan)

# mask = np.isnan(feature_data[:, 4])
# np.where(mask)


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

a = 3 if 2 > 100 else 6
a
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
        if env.elapsed_time > 5:
            decision = input()
            if decision == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(decision)
            except:
                pass

            viewer.render(env.vehicles)
            # print(env.elapsed_time)
        env.step()


if __name__=='__main__':
    main()

# %%
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def animation_frame(i):
        viewer.render(env.vehicles)
        env.step()
        # return line,

    animation = FuncAnimation(viewer.fig, func=animation_frame, frames=range(1800), interval=1000)


    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=25, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('sim_example.mp4', writer, dpi=500)



(1 == 10 or 4 == 6) and 6 == 6

# get_animation()
# plt.show()
# %%
