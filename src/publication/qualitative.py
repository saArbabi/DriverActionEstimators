import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
import os
import pickle
import sys
import json
import tensorflow as tf
from importlib import reload
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

reload(pyplot)
reload(plt)
sys.path.insert(0, './src')

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

# x%%
"""
Needed methods
"""
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_samples(model, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    merger_cs = future_m_veh_c[sample_index, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
    z_, _ = model.belief_net.sample_z(latent_dis_param)
    return z_

def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    """ Returns the state sequence. It also deletes the middle index, which is
        the transition point from history to future.
    """
    # data shape: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], history_len-1, axis=1)
    return traj.flatten()
# %%
"""
Load data
"""
history_len = 30 # steps
rollout_len = 50
data_id = '049'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

data_files_dir = './src/datasets/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    data_arrays = pickle.load(handle)
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_c = np.float32(future_m_veh_c)
history_future_usc.shape

all_epis = np.unique(history_sca[:, 0, 0])
np.random.seed(2021)
np.random.shuffle(all_epis)
train_epis = all_epis[:int(len(all_epis)*0.7)]
val_epis = np.setdiff1d(all_epis, train_epis)
train_samples = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_samples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
# %%
"""
Load model (with config file)
"""
model_name = 'neural_idm_367'
# model_name = 'neural_idm_367_low_beta' # run with epoch 3
# model_name = 'neural_045'
epoch_count = '10'
# epoch_count = '3'
exp_path = './src/models/experiments/'+model_name+'/model_epo'+epoch_count
exp_dir = os.path.dirname(exp_path)
with open(exp_dir+'/'+'config.json', 'rb') as handle:
    config = json.load(handle)
    print(json.dumps(config, ensure_ascii=False, indent=4))

from models.core import neural_idm
reload(neural_idm)
from models.core.neural_idm import NeurIDMModel
model = NeurIDMModel(config)
model.forward_sim.rollout_len = 50

model.load_weights(exp_path).expect_partial()

with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
    model.forward_sim.env_scaler = pickle.load(handle)

with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
    model.forward_sim.dummy_value_set = pickle.load(handle)
# %%
"""
Checkout some sample predictions
"""
Example_pred = 0
i = 0
covered_episodes = []
model.forward_sim.attention_temp = 1
traces_n = 50
tf.random.set_seed(2021)

sepcific_samples = []
distribution_name = 'prior'

while Example_pred < 5:
    sample_index = [val_samples[i]]
    i += 1
    e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
    m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode not in covered_episodes and episode != -8 and \
    #             e_veh_att[40:].mean() == 1 and \
    #             e_veh_att[:30].mean() == 0 and aggressiveness > 0.8:
    if episode not in covered_episodes and episode != -8 and \
                e_veh_att.mean() == 0 and aggressiveness > 0.8:

        Example_pred += 1
        covered_episodes.append(episode)
        merger_cs = vectorise(future_m_veh_c[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model.h_seq_encoder(h_seq)
        if distribution_name == 'posterior':
            f_seq = vectorise(future_sca[sample_index, :, 2:], traces_n)
            enc_f = model.f_seq_encoder(f_seq)
            _, latent_dis_param = model.belief_net([enc_h, enc_f], dis_type='both')
        elif distribution_name == 'prior':
            latent_dis_param = model.belief_net(enc_h, dis_type='prior')
        z_idm, z_att = model.belief_net.sample_z(latent_dis_param)

        proj_idm = model.belief_net.z_proj_idm(z_idm)
        proj_att = model.belief_net.z_proj_att(z_att)
        idm_params = model.idm_layer(proj_idm)
        displacement_seq, act_seq, att_scores = \
                                            model.forward_sim.rollout([\
                                            idm_params, proj_att, enc_h,\
                                            future_idm_ss, merger_cs])
        f_att_seq, m_att_seq = att_scores[0].numpy(), att_scores[1].numpy()
        act_seq = act_seq.numpy()
        if np.isnan(act_seq).any():
            raise ValueError('There is nan in actions')
        plt.figure(figsize=(5, 4))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = int(history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0])

        time_steps = range(time_0, time_0+history_len+rollout_len-1)
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
        plt.text(0.1, 0.4,
                        'experiment_name: '+ model_name+'_'+epoch_count +' '+'\n'
                        'episode_id: '+ info[0] +
                        'time_0: '+ info[1] +
                        'e_veh_id: '+ info[2] +
                        'aggressiveness: '+ info[3] +
                        'step_20_speed: '+ str(future_idm_ss[0, 0, 0])
                            , fontsize=10)

        true_params = []
        for param_name in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
            true_pram_val = history_future_usc[sample_index, 0, hf_usc_indexs[param_name]][0]
            true_params.append(round(true_pram_val, 2))

        plt.text(0.1, 0.3, 'true: '+ str(true_params))
        plt.text(0.1, 0.1, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))
        plt.figure(figsize=(5, 3))
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action_c'])
        plt.plot(time_steps, traj, color='purple')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action_c'])
        plt.plot(time_steps, traj, color='black', linewidth=2)
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action_c'])
        plt.plot(time_steps, traj, color='red')
        plt.legend(['f_veh_action_c', 'e_veh_action_c', 'm_veh_action_c'])

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[history_len-1:], act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.4)
        plt.scatter(time_steps[history_len-1+20], 0, color='red')

        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        plt.plot(time_steps , e_veh_att, color='red')
        plt.plot([time_steps[history_len-1], time_steps[history_len-1]], [0, 1], color='black')

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[history_len-1:], m_att_seq[sample_trace_i, :].flatten(), color='grey')
        plt.title(str(sample_index[0]) + ' -- Attention on merger')

# %%
"""Prediction for an specific sample from the dataset
"""
traces_n = 100
model.forward_sim.attention_temp = 1
sample_index = [11540]
sample_index = [109772]
# sample_index = [6590]
tf.random.set_seed(2021)
time_steps = range(0, history_len+rollout_len-1)

distribution_name = 'prior'
# distribution_name = 'posterior'

e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
episode = future_idm_s[sample_index, 0, 0][0]

merger_cs = vectorise(future_m_veh_c[sample_index, :, 2:], traces_n)
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model.h_seq_encoder(h_seq)
if distribution_name == 'posterior':
    f_seq = vectorise(future_sca[sample_index, :, 2:], traces_n)
    enc_f = model.f_seq_encoder(f_seq)
    _, latent_dis_param = model.belief_net([enc_h, enc_f], dis_type='both')
elif distribution_name == 'prior':
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
z_idm, z_att = model.belief_net.sample_z(latent_dis_param)

proj_idm = model.belief_net.z_proj_idm(z_idm)
proj_att = model.belief_net.z_proj_att(z_att)
idm_params = model.idm_layer(proj_idm)
displacement_seq, act_seq, att_scores = \
                                    model.forward_sim.rollout([\
                                    idm_params, proj_att, enc_h,\
                                    future_idm_ss, merger_cs])
f_att_seq, m_att_seq = att_scores[0].numpy(), att_scores[1].numpy()
act_seq = act_seq.numpy()


plt.figure(figsize=(5, 3))
episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
plt.text(0.1, 0.5,
        'episode_id: '+ info[0] +
        'time_0: '+ info[1] +
        'e_veh_id: '+ info[2] +
        'aggressiveness: '+ info[3]
            , fontsize=10)

true_params = []
for param_name in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    true_pram_val = history_future_usc[sample_index, 0, hf_usc_indexs[param_name]][0]
    true_params.append(round(true_pram_val, 2))

plt.text(0.1, 0.3, 'true: '+ str(true_params))
plt.text(0.1, 0.1, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))
plt.figure(figsize=(5, 3))
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action_c'])
plt.plot(time_steps, traj, color='purple')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action_c'])
plt.plot(time_steps, traj, color='black', linewidth=2)
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action_c'])
plt.plot(time_steps, traj, color='red')
plt.legend(['f_veh_action_c', 'e_veh_action_c', 'm_veh_action_c'])

for sample_trace_i in range(traces_n):
   plt.plot(time_steps[history_len-1:], act_seq[sample_trace_i, :, :].flatten(),
                                color='grey', alpha=0.4)

plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

plt.figure(figsize=(5, 3))
# plt.plot(e_veh_att[:40] , color='black')
plt.plot(time_steps , e_veh_att, color='red')
plt.plot([time_steps[history_len-1], time_steps[history_len-1]], [0, 1], color='black')

for sample_trace_i in range(traces_n):
   plt.plot(time_steps[history_len-1:], m_att_seq[sample_trace_i, :].flatten(), color='grey')
plt.title(str(sample_index[0]) + ' -- Attention on merger')

# %%

""" plot setup
"""
MEDIUM_SIZE = 18
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 18,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# %%
"""
Latent visualisation - NIDM VS CVAE
"""
zsamples_n = 5000
tf.random.set_seed(2021)
examples_to_vis = np.random.choice(val_samples, zsamples_n, replace=False)
sampled_z = latent_samples(model, examples_to_vis).numpy()

# %%
fig = pyplot.figure()
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

# ax.view_init(10, 10)
# ax.grid(False)
axins = inset_axes(ax,
                    width="5%",
                    height="90%",
                    loc='right',
                    borderpad=-3.5)
fig.colorbar(latent_plot, cax=axins, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
plt.ylabel('$\psi$', fontsize=25, rotation=0, labelpad=12)

plt.savefig("NIDM_latent.jpg", dpi=500, bbox_inches='tight')


# %%
fig = pyplot.figure()
ax = plt.axes(projection='3d')
ax.zaxis.set_rotate_label(False)
ax.set_xlabel('$z_1$', fontsize=25)
ax.set_ylabel('$z_2$', rotation=0, fontsize=25)
ax.set_zlabel('$z_3$', fontsize=25)
ax.tick_params(axis='both', which='major', pad=1)
# z1 ticks
ax.set_xlim(-10.5, 10.5)

# z2 ticks
ax.set_ylim(-1.5, 1.5)

# z3 ticks
ax.set_zlim(-1.5, 1.5)

aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]
color_shade = aggressiveness
#
latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
          s=10, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)


plt.savefig("neural_latent.jpg", dpi=500, bbox_inches='tight')

# %%
"""
Action Figure
"""
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', lw=4, linestyle='-'),
                Line2D([0], [0], color='grey', lw=4),
                Line2D([0], [0], color='red', lw=4, linestyle='--'),
                ]



fig = plt.figure(figsize=(8, 4))
ax_1 = fig.add_subplot(111)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.28, hspace=0.1)


time_axis = np.linspace(0., 8., history_len+rollout_len-1)
for sample_trace_i in range(traces_n):
    label = '_nolegend_'
    if sample_trace_i == 0:
        label = 'NIDM'
    ax_1.plot(time_axis[history_len-1:], act_seq[sample_trace_i, :, :].flatten(), \
                    color='grey', alpha=0.4, linewidth=1, label=label, linestyle='-')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action_c'])

ax_1.plot(time_axis[:history_len], traj[:history_len], color='black', \
                                linestyle='-', label='Action history', linewidth=2)
ax_1.plot(time_axis[history_len:], traj[history_len:], color='red', \
                                    linestyle='--', label='True action', linewidth=2)


ax_1.set_xlabel('Time (s)')
ax_1.set_ylabel('Long. Accel ($\mathdefault{m/s^2}$)')
ax_1.set_xticks([0., 2, 4, 6, 8])
ax_1.set_yticks([3, 0, -3, -6,])
ax_1.set_ylim(-6.5, 4.1)
ax_1.set_xlim(0, 8.1)
ax_1.grid(alpha=0.2)
ax_1.minorticks_off()
ax_1.tick_params(top=False)
ax_1.legend(custom_lines, ['Action history', 'NIDM',\
                              'Ground-truth', 'Agent chosen plan'], ncol=1)


plt.savefig("action_fig.pdf", dpi=500, bbox_inches='tight')
# %%
"""
Attetnion Figure
"""
custom_lines = [Line2D([0], [0], color='blue', lw=3, linestyle='--'),
                Line2D([0], [0], color='red', lw=3, linestyle='--'),
                Line2D([0], [0], color='blue', lw=3, linestyle='-'),
                Line2D([0], [0], color='red', lw=3, linestyle='-')]

fig = plt.figure(figsize=(8, 4))
ax_2 = fig.add_subplot(111)
linewidth = 2.5
ax_2.plot(time_axis[:history_len-1], e_veh_att[:history_len-1], color='red', linestyle='-', label='$True \; w_m$', linewidth=linewidth)
ax_2.plot(time_axis[history_len:], e_veh_att[history_len:], color='red', linestyle='--', label='$True \; w_m$', linewidth=linewidth)


ax_2.plot(time_axis[:history_len-1], 1-e_veh_att[:history_len-1], color='blue', linestyle='-', label='$True \; w_f$', linewidth=linewidth)
ax_2.plot(time_axis[history_len:], 1-e_veh_att[history_len:], color='blue', linestyle='--', label='$True \; w_f$', linewidth=linewidth)


def get_att_space(att_seq):
    att_stdev = att_seq[:, :].std(axis=0)/2
    att_mean = att_seq[:, :].mean(axis=0)
    return att_mean, np.clip(att_mean+att_stdev, 0, 1), np.clip(att_mean-att_stdev, 0, 1)

att_mean, max_bound, min_bound = get_att_space(m_att_seq[:, :, 0])
ax_2.fill_between(time_axis[history_len-1:], max_bound, \
                min_bound, color='red', alpha=0.2, label='$ NIDM \; w_m$')

ax_2.plot(time_axis[history_len-1:], att_mean, color='red', alpha=0.5, linewidth=linewidth)

att_mean, max_bound, min_bound = get_att_space(f_att_seq[:, :, 0])
ax_2.fill_between(time_axis[history_len-1:], max_bound, \
                min_bound, color='blue', alpha=0.2, label='$NIDM \; w_f$')
ax_2.plot(time_axis[history_len-1:], att_mean, color='blue', alpha=0.5, linewidth = 3, linestyle='-')

ax_2.set_xlim(3, 8.1)
ax_2.set_xlabel('Time (s)')
ax_2.set_ylabel('Ego attention')
ax_2.set_yticks([0., 0.5, 1])
# ax_2.legend(custom_lines, ['True $w_l$', 'True $w_m$', 'NIDM $w_l$', 'NIDM $w_m$'],
#                   loc='upper center', bbox_to_anchor=(0.5, 1.18), edgecolor='black', ncol=4)

ax_2.legend(custom_lines, ['True $w_l$', 'True $w_m$', 'NIDM $w_l$', 'NIDM $w_m$'], edgecolor='black', ncol=1)
ax_2.minorticks_off()
plt.savefig("attention_fig.pdf", dpi=500, bbox_inches='tight')
# %%
"""
2D Latent figure
"""
zsamples_n = 5000
examples_to_vis = np.random.choice(val_samples, zsamples_n, replace=False)
sampled_z = latent_samples(model, examples_to_vis).numpy()
# %%
fig = plt.figure(figsize=(6.5, 4))
ax_3 = fig.add_subplot(111)
aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]

color_shade = aggressiveness
latent_plot = ax_3.scatter(sampled_z[:, 0], sampled_z[:, 1],
                s=10, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
ax_3.grid(False)
ax_3.scatter(z_idm[:, 0], z_idm[:, 1], s=20, marker="x", edgecolors='none', color='black')
ax_3.scatter(z_idm[:, 0].numpy().mean(), z_idm[:, 1].numpy().mean(), s=5000, marker="s", edgecolors='black', facecolors='none')

axins = inset_axes(ax_3,
                    width="5%",
                    height="90%",
                    loc='right',
                    borderpad=-2
                   )
fig.colorbar(latent_plot, cax=axins, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
plt.ylabel('$\psi$', fontsize=25, rotation=0, labelpad=12)
ax_3.set_xlabel('$z_1$', fontsize=35)
ax_3.set_ylabel('$z_2$', fontsize=35)
ax_3.set_xticks([-5, 0, 5])
# ax_3.set_xlim(-7, 5.5)
ax_3.set_ylim(-11, 11)


ax_3.minorticks_off()

plt.savefig("latent_fig.jpg", dpi=500, bbox_inches='tight')


# %%
subplot_xcount = 5
subplot_ycount = 1
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(15, 3))
fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

for i, ax in enumerate(axs):
    ax.set_ylim(-0.1,  12.1)
    if i > 0:
        # ax.set_yticks([])
        ax.set_yticklabels(['']*2)
        # ax.grid(alpha=0)
    if i == 0:
        ax.set_ylabel('Probability Density', fontsize=18)



# 2%%

def get_gmm_ll(data, test_x, n_classes=3):
    """
    Fit GMM to data with EM
    """
    estimator = GaussianMixture(n_components=n_classes, max_iter=3, random_state=0)
    data = tf.reshape(data, [-1, 1]).numpy()
    estimator.fit(data)
    test_x = np.reshape(test_x, [-1, 1])
    gmm_ll_data = np.exp(estimator.score_samples(data))
    gmm_ll_test = np.exp(estimator.score_samples(test_x))
    return estimator, gmm_ll_data, gmm_ll_test

"""desired_v"""
my_axis = 0

axs[my_axis].set_xlabel('$\mathdefault{v_{des} \; (m/s)}$', fontsize=20)
true_val = true_params[my_axis]
min_xlim = 23
max_xlim = 26

param_samples = idm_params[:, my_axis]
x = np.linspace(min_xlim , max_xlim, 100)
estimator, gmm_ll_data, gmm_ll_test = get_gmm_ll(param_samples, x)

axs[my_axis].plot(x, gmm_ll_test, linewidth=1, color='red')
axs[my_axis].plot([true_val, true_val], [-0.1, 20],
            linewidth=1.5, color='black', linestyle='--')
axs[my_axis].scatter(param_samples, gmm_ll_data, s=10, color='blue', alpha=0.5)

# axs[my_axis].set_xlim(min_xlim-0.1,  max_xlim+0.1)
# axs[my_axis].set_ylim(-0.1,  12.1)
# axs[my_axis].set_xticks([23, 24.5, 26])
# axs[my_axis].set_yticks([0, 1, 2, 3])

"""desired_tgap"""
my_axis = 1

axs[my_axis].set_xlabel('$\mathdefault{T_{des} \; (s)}$', fontsize=20)
true_val = true_params[my_axis]
min_xlim = 0.41
max_xlim = 1.1

param_samples = idm_params[:, my_axis]
x = np.linspace(min_xlim , max_xlim, 100)
estimator, gmm_ll_data, gmm_ll_test = get_gmm_ll(param_samples, x)

axs[my_axis].plot(x, gmm_ll_test, linewidth=1, color='red')
axs[my_axis].plot([true_val, true_val], [-0.1, 20],
            linewidth=1.5, color='black', linestyle='--')
axs[my_axis].scatter(param_samples, gmm_ll_data, s=10, color='blue', alpha=0.5)

"""min_jamx"""
my_axis = 2

axs[my_axis].set_xlabel('$\mathdefault{d_{min}  \; (m)}$', fontsize=20)
true_val = true_params[my_axis]
min_xlim = 0.5
max_xlim = 1.7

param_samples = idm_params[:, my_axis]
x = np.linspace(min_xlim , max_xlim, 100)
estimator, gmm_ll_data, gmm_ll_test = get_gmm_ll(param_samples, x)

axs[my_axis].plot(x, gmm_ll_test, linewidth=1, color='red')
axs[my_axis].plot([true_val, true_val], [-0.1, 20],
            linewidth=1.5, color='black', linestyle='--')
axs[my_axis].scatter(param_samples, gmm_ll_data, s=10, color='blue', alpha=0.5)

"""max_act"""
my_axis = 3

axs[my_axis].set_xlabel('$\mathdefault{a_{max} \; (m/s^2)}$', fontsize=20)
true_val = true_params[my_axis]
min_xlim = 2.3
max_xlim = 4.2

param_samples = idm_params[:, my_axis]
x = np.linspace(min_xlim , max_xlim, 100)
estimator, gmm_ll_data, gmm_ll_test = get_gmm_ll(param_samples, x)

axs[my_axis].plot(x, gmm_ll_test, linewidth=1, color='red')
axs[my_axis].plot([true_val, true_val], [-0.1, 20],
            linewidth=1.5, color='black', linestyle='--')
axs[my_axis].scatter(param_samples, gmm_ll_data, s=10, color='blue', alpha=0.5)

"""min_act"""
my_axis = 4

axs[my_axis].set_xlabel('$\mathdefault{b_{max}  \; (m/s^2)}$', fontsize=20)
true_val = true_params[my_axis]
min_xlim = 3.4
max_xlim = 4.2

param_samples = idm_params[:, my_axis]
x = np.linspace(min_xlim , max_xlim, 100)
estimator, gmm_ll_data, gmm_ll_test = get_gmm_ll(param_samples, x)

axs[my_axis].plot(x, gmm_ll_test, linewidth=1, color='red')
axs[my_axis].plot([true_val, true_val], [-0.1, 20],
            linewidth=1.5, color='black', linestyle='--')
axs[my_axis].scatter(param_samples, gmm_ll_data, s=10, color='blue', alpha=0.5)

plt.savefig("idm_params.pdf", dpi=500, bbox_inches='tight')
