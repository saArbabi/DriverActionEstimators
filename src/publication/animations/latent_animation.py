"""
This script is used to visualisae NIDM latent space as the model is trained.
"""
import sys
sys.path.insert(0, './src')
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot

import numpy as np
np.set_printoptions(suppress=True)
from importlib import reload
import pickle
import os
import json
import time
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from models.core.neural_idm import NeurIDMModel

MEDIUM_SIZE = 18
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': MEDIUM_SIZE,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# %%
def get_animation(epochs_n):
    """To show case how the latent spaces changes with more training.
    """
    def latent_samples(model, sample_index):
        h_seq = history_sca[sample_index, :, 2:]
        enc_h = model.h_seq_encoder(h_seq)
        latent_dis_param = model.belief_net(enc_h, dis_type='prior')
        z_, _ = model.belief_net.sample_z(latent_dis_param)
        return z_

    model_name = 'neural_idm_367_latent_anim'
    exp_path = './src/models/experiments/'+model_name+'/model_epo0'
    exp_dir = os.path.dirname(exp_path)
    with open(exp_dir+'/'+'config.json', 'rb') as handle:
        config = json.load(handle)
    model = NeurIDMModel(config)

    zsamples_n = 5000
    tf.random.set_seed(2021)
    examples_to_vis = np.random.choice(val_samples, zsamples_n, replace=False)
    sampled_z = latent_samples(model, examples_to_vis).numpy()

    fig = pyplot.figure(figsize=(7, 4))
    ax = plt.axes(projection='3d')
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('$z_1$', rotation=0, fontsize=25)
    ax.set_ylabel('$z_2$', labelpad=10, rotation=0, fontsize=25)
    ax.set_zlabel('$z_3$', rotation=0, fontsize=25)
    ax.tick_params(axis='both', which='major', pad=1)

    # z1 ticks
    ax.set_xlim(-5, 5)
    ax.set_xticks([-4, 0, 4])

    # z2 ticks
    ax.set_ylim(-22, 5)
    ax.set_yticks([-20, -10, 0])

    # z3 ticks
    ax.set_zlim(-3.2, 5)
    ax.set_zticks([-3, 0, 3, 6])


    aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]
    color_shade = aggressiveness

    latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
              s=10, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    axins = inset_axes(ax,
                        width="5%",
                        height="90%",
                        loc='right',
                        borderpad=-3.5)
    fig.colorbar(latent_plot, cax=axins, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    plt.ylabel('$\psi$', fontsize=25, rotation=0, labelpad=12)

    def animation_frame(i):
        exp_path = './src/models/experiments/'+model_name+'/model_epo'+str(i)
        print(i)
        model.load_weights(exp_path).expect_partial()
        sampled_z = latent_samples(model, examples_to_vis).numpy()
        latent_plot._offsets3d = (sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2])
        ax.set_title('Iterations: '+ str(i))
        # ax.view_init(azim=i)

    animation = FuncAnimation(fig, func=animation_frame, \
                              frames=epochs_n, interval=1)

    Writer = writers['ffmpeg']
    writer = Writer(fps=10, bitrate=3000)
    animation.save('latent_evolution.mp4', writer, dpi=250)

# %%
"""
Load data
"""
data_id = '049'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=30, rollout_len=50)

data_files_dir = './src/datasets/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    data_arrays = pickle.load(handle)
history_future_usc, history_sca, _, _, _, _ = data_arrays
all_epis = np.unique(history_sca[:, 0, 0])

train_epis = all_epis[:int(len(all_epis)*0.7)]
val_epis = np.setdiff1d(all_epis, train_epis)
val_samples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]

# %%
hf_usc_indexs = {}

col_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'm_veh_exists', 'e_veh_att',
         'e_veh_action_p', 'f_veh_action_p', 'm_veh_action_p',
         'e_veh_action_c', 'f_veh_action_c', 'm_veh_action_c',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'aggressiveness',
         'desired_v','desired_tgap', 'min_jamx', 'max_act', 'min_act',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
         'em_delta_y', 'delta_x_to_merge']
for i, item_name in enumerate(col_names):
    hf_usc_indexs[item_name] = i

# plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
get_animation(epochs_n=162)

np.random.randint(140, 190, 10)
