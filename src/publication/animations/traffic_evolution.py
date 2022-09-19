"""
Reads trajectory files and turns them into animations.

Trajs are generated using the script in qualitative.py
"""
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from importlib import reload
import json
import pickle
from matplotlib.animation import FuncAnimation, writers

# from matplotlib.lines import Line2D

# %%
"""
Load logs
"""
_dir = './src/publication/animations'
# st = [
#      self.time_step,
#      vehicle.act_long_c,
#      vehicle.speed,
#      vehicle.glob_x,
#      vehicle.glob_y]
with open(_dir+'/collections.pickle', 'rb') as handle:
    collections = pickle.load(handle) # {trace: {vehicle:[[], ...]}}


def dic_to_arr(collection):
    collection_list = []
    for trace_val in collection.values():
        trace_list = []
        for veh_val in trace_val.values():
            trace_list.append(veh_val)
        collection_list.append(trace_list)
    return np.array(collection_list)

for key, value in collections.items():
    collections[key] = [dic_to_arr(value[0]), dic_to_arr(value[1])]

collections[key][1].shape

# %%
plt.plot(real_arr[0, -2, :, 1], color='red')

for i in range(20):
    plt.plot(ima_arr[i, -2, :, 1], color='grey')

# %%
"""
Plot scene
"""
MEDIUM_SIZE = 18
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': MEDIUM_SIZE,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
from src.publication.animations import scene_viewer
reload(scene_viewer)
from src.publication.animations.scene_viewer import Viewer

with open('./src/envs/config.json', 'rb') as handle:
    config = json.load(handle)

plot_viewer = Viewer(config)
collections_copy = collections.copy()
plot_viewer.set_up_fig()
plot_viewer.draw_road()
plot_viewer.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.3)

ax_datas = []
for ax, collection in zip(plot_viewer.axs, collections_copy.values()):
    ax.collision_count = 0
    ax.text(10, 5.3, 'Collision count: ' + str(ax.collision_count), size=20, color='black')
    real_arr, ima_arr = collection
    ima_scatters = []
    real_scatters = []
    lines = []

    i = 30
    for veh_i in range(1, 6):
        if veh_i == 5:
            ima_facecolors='red'
            ima_edgecolor='red'
            real_facecolors='red'
            real_edgecolor='red'

            ima_glob_x = ima_arr[0, veh_i-1, i, 3]
            ima_glob_y = ima_arr[0, veh_i-1, i, 4]
            real_glob_x = real_arr[0, veh_i-1, i, 3]
            real_glob_y = real_arr[0, veh_i-1, i, 4]
        else:
            ima_facecolors='none'
            ima_edgecolor='grey'
            real_facecolors='green'
            real_edgecolor='green'

            ima_glob_x = ima_arr[:, veh_i-1, i, 3]
            ima_glob_y = ima_arr[:, veh_i-1, i, 4]
            real_glob_x = real_arr[0, veh_i-1, i, 3]
            real_glob_y = real_arr[0, veh_i-1, i, 4]

        ima_scatter = ax.scatter(ima_glob_x, ima_glob_y, s=150, marker='>', \
                                             facecolors=ima_facecolors, edgecolor=ima_edgecolor,\
                                             linewidth=2)

        real_scatter = ax.scatter(real_glob_x, real_glob_y, s=150, marker='>', \
                                             facecolors=real_facecolors, edgecolor=real_edgecolor,\
                                             linewidth=2)

        line, = ax.plot([], [], color=real_facecolors, linestyle='-', linewidth=15, alpha=0.1)

        ima_scatters.append(ima_scatter)
        real_scatters.append(real_scatter)
        lines.append(line)

    ax_datas.append([ima_scatters, real_scatters, lines])


# c%%

def animation_frame(i):
    print(i)
    plot_viewer.axs[0].set_title('Simulation time: '+str(i/10))
    for ax_data, collection, ax in zip(ax_datas, collections_copy.values(), plot_viewer.axs):
        ima_scatters, real_scatters, lines = ax_data
        real_arr, ima_arr = collection

        for n, (ima_scatter, real_scatter, line) in enumerate(zip(ima_scatters, real_scatters, lines)):
            ima_scatter.set_offsets(ima_arr[:, n, i+30, 3:])
            real_scatter.set_offsets(real_arr[0, n, i+30, 3:])
            line.set_data(real_arr[0, n, i:i+30, 3], real_arr[0, n, i:i+30, 4])

            # check for collisions and remove collided cars
            for nn in range(5):
                if nn != n:
                    x_diffs = np.abs((real_arr[0, nn, i+30, 3] - ima_arr[:, n, i+30, 3]))
                    x_diff_min = x_diffs.min()
                    indx = np.where(x_diffs == x_diff_min)[0]
                    y_diff_min = np.abs((ima_arr[:, n, i+30, 4] - real_arr[0, nn, i+30, 4])).min()
                    if y_diff_min < 0.5 and x_diff_min < 3:
                        ax.collision_count += 1
                        ima_arr[indx, n, :, 3] = -10
                        for text in ax.texts:
                            text.remove()
                        ax.text(10, 5.3, 'Collision count: ' + str(ax.collision_count), size=20, color='black')



    #
    # for n, scatter in enumerate(ima_scatters):
    #     scatter.set_offsets(ima_arr[0, n, i, 3:])
    #
    # for n, scatter in enumerate(real_scatters):
    #     scatter.set_offsets(real_arr[0, n, i, 3:])

animation = FuncAnimation(plot_viewer.fig, func=animation_frame,
                          frames=range(200), interval=100)


# setting up wrtiers object
Writer = writers['ffmpeg']
writer = Writer(fps=10, bitrate=3000)
animation.save('sim_example.mp4', writer, dpi=250)
print('COMPETE')


# %%
a = [1, 2, 3]
b = [1, 2, 3]

scatter = plt.scatter(a, b)
scatter.remove()
