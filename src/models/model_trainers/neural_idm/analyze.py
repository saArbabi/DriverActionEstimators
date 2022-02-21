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
reload(pyplot)
sys.path.insert(0, './src')

hf_usc_indexs = {}
col_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'm_veh_exists', 'e_veh_att',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'e_veh_action', 'f_veh_action', 'm_veh_action',
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

def latent_vis(zsamples_n):
    fig = pyplot.figure(figsize=(4, 4))
    examples_to_vis = np.random.choice(val_samples, zsamples_n, replace=False)
    sampled_z = latent_samples(model, examples_to_vis).numpy()
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 1, 1)
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.xaxis.set_tick_params(pad=1, which='both')
    # ax.yaxis.set_tick_params(pad=1, which='both')
    # ax.zaxis.set_tick_params(pad=1, which='both')

    # x ticks
    # ax.set_xticks([-2, 0, 2], minor=False)
    # ax.set_xlim(-2.5, 2.5)

    # y ticks
    # ax.set_yticks([-4, -2, 0, 2], minor=False)
    # ax.set_ylim(-4.5, 2.5)

    # z ticks
    # ax.set_zticks([-6, -3, 0, 3], minor=False)
    # ax.set_zlim(-6.5, 3.5)
    ax.minorticks_off()


    aggressiveness = history_future_usc[examples_to_vis, 0, hf_usc_indexs['aggressiveness']]
    color_shade = aggressiveness
    # latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
    #               s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
    latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
    # axins = inset_axes(ax,
    #                     width="5%",
    #                     height="90%",
    #                     loc='right',
    #                     borderpad=-2
    #                    )
    #
    # fig.colorbar(latent_plot, cax=axins, ticks=np.arange(0, 1.1, 0.2))

    ax.grid(False)
    # ax.view_init(30, 50)
    #===============
    #  Second subplot
    #===============
    # set up the axes for the second plot
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # latent_plot = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
    #               s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)
    #
    # axins = inset_axes(ax,
    #                     width="5%",
    #                     height="90%",
    #                     loc='right',
    #                     borderpad=-2
    #                    )
    #
    # fig.colorbar(latent_plot, cax=axins)
    # cbar = fig.colorbar(latent_plot, cax=axins)
    # ax.tick_params(pad=1)
    # ax.grid(False)
    # ax.view_init(30, 50)
    # # ax.set_xlabel('$z_{1}$', labelpad=1)
    # # ax.set_ylabel('$z_{2}$', labelpad=1)
    # # ax.set_zlabel('$z_{3}$', labelpad=1)
    # plt.subplots_adjust(wspace=0.2, hspace=None)
    return ax

def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    """ Returns the state sequence. It also deletes the middle index, which is
        the transition point from history to future.
    """
    # data shape: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 19, axis=1)
    return traj.flatten()
# %%
"""
Load data
"""
history_len = 20 # steps
rollout_len = 50
data_id = '047'
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
# np.where(train_epis == 64)
train_samples = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_samples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
# history_sca.shape
train_samples.shape
# %%
"""
Load model (with config file)
"""
model_name = 'neural_idm_306'
# model_name = 'neural_idm_test_48'
epoch_count = '10'
epoch_count = '5'
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
# model.forward_sim.attention_temp = 10
# %%
a = np.zeros([512, 1])
a[0:50, :] = 4
a.mean()
# %%

"""
Plot loss
"""
with open(exp_dir+'/'+'losses.pickle', 'rb') as handle:
    losses = pickle.load(handle)
plt.figure()
plt.plot(losses['test_mseloss'], label='test_mseloss')
plt.plot(losses['train_mseloss'], label='train_mseloss')
plt.grid()
plt.legend()
plt.figure()
plt.plot(losses['test_klloss'], label='test_klloss')
plt.plot(losses['train_klloss'], label='train_klloss')
plt.legend()
plt.grid()

# %%
"""
Compare losses
"""
losses = {}
for name in ['neural_idm_269']:
# for name in ['latent_mlp_09', 'latent_mlp_10']:
    with open('./src/models/experiments/'+name+'/'+'losses.pickle', 'rb') as handle:
        losses[name] = pickle.load(handle)

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_mseloss'], label='test_mseloss')
    plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_klloss'], label='test_klloss')
    plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()
# %%
"""
Find bad examples
"""
import tensorflow as tf
# examples_to_vis = val_samples
# val_samples.shape
def get_avg_loss_across_sim(examples_to_vis):
    merger_cs = future_m_veh_c[examples_to_vis, :, 2:]
    h_seq = history_sca[examples_to_vis, :, 2:]
    future_idm_ss = future_idm_s[examples_to_vis, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
    sampled_z = model.belief_net.sample_z(latent_dis_param)
    proj_latent = model.belief_net.belief_proj(sampled_z)
    idm_params = model.idm_layer(proj_latent)
    act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_latent, \
                                            future_idm_ss, merger_cs])
    true_actions = future_e_veh_a[examples_to_vis, :, 2:]
    loss = (tf.square(tf.subtract(act_seq, true_actions)))**0.5
    return tf.reduce_mean(loss, axis=1).numpy()
# loss = get_avg_loss_across_sim(train_samples[0:15])
loss = get_avg_loss_across_sim(val_samples[0:5000])
_ = plt.hist(loss, bins=150)
# _ = plt.hist(loss[loss<0.1], bins=150)
bad_samples = np.where(loss > 1)

# %%
def get_target_vals(arr):
    dxs = np.zeros([arr.shape[0], rollout_len+1, 1])

    for step in range(1, rollout_len+1):
        dxs[:, step, :] = dxs[:, step-1, :] + arr[:, step-1, 1:]*0.1 \
                                                    + 0.5*arr[:, step-1, 0:1]*0.1**2
    return dxs
data_arrays[-1].shape
dxs = get_target_vals(data_arrays[-1][:, :, 2:])
# %%
"""
Latent visualisation - aggressiveness used for color coding the latent samples
"""
latent_vis(zsamples_n=5000)
latent_vis(zsamples_n=5000)
# plt.savefig("nidm_latent.png", dpi=500)

# plt.savefig("latent.png", dpi=500)
# %%
# params = {
#           'font.size' : 12,
#           'font.family' : 'Palatino Linotype',
#           }
# plt.rcParams.update(params)
# %%
for i in range(10):
    plt.plot(displacement_seq[i, :, 0], color='gray')
plt.plot(dxs[sample_index[0], :, 0], color='red')


# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
Example_pred = 0
i = 0
covered_episodes = []
model.forward_sim.attention_temp = 5
traces_n = 50
tf.random.set_seed(2021)

# np.where((history_future_usc[:, 0, 0] == 26) & (history_future_usc[:, 0, 2] == 4))
sepcific_samples = []
distribution_name = 'prior'
# distribution_name = 'posterior'

while Example_pred < 10:
    sample_index = [val_samples[i]]
    # sample_index = [train_samples[i]]
    # sample_index = [i]
    i += 1
    e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
    m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode == 8 and \
    if episode not in covered_episodes and episode != -8 and \
                e_veh_att[20:35].mean() > 0 and e_veh_att[:20].mean() == 0:
                # e_veh_att.mean() == 0:
                # e_veh_att.mean() == 0:
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
        displacement_seq, act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_att, enc_h, \
                                                     future_idm_ss, merger_cs])
        f_att_seq, m_att_seq = att_scores[0].numpy(), att_scores[1].numpy()
        act_seq = act_seq.numpy()
        if np.isnan(act_seq).any():
            raise ValueError('There is nan in actions')
        plt.figure(figsize=(5, 4))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = int(history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0])

        time_steps = range(time_0, time_0+69)
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
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
        plt.plot(time_steps, traj, color='purple')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
        plt.plot(time_steps, traj, color='black', linewidth=2)
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
        plt.plot(time_steps, traj, color='red')
        plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[19:], act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.4)
        plt.scatter(time_steps[19+20], 0, color='red')
        # plt.text(30, -6, 'true: '+ str(true_params), color='red')
        # plt.text(30, -7, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))

        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        # plt.plot(e_veh_att[:40] , color='black')
        plt.plot(time_steps , e_veh_att, color='red')
        plt.plot([time_steps[19], time_steps[19]], [0, 1], color='black')

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[19:], m_att_seq[sample_trace_i, :].flatten(), color='grey')
        plt.title(str(sample_index[0]) + ' -- Attention on merger')

        plt.figure(figsize=(5, 3))
        plt.plot(time_steps , e_veh_att, color='red')
        plt.plot([time_steps[19], time_steps[19]], [0, 1], color='black')
        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[19:], f_att_seq[sample_trace_i, :].flatten(), color='grey')
        plt.title(str(sample_index[0]) + ' -- Attention on leader')

        ##########
        plt.figure(figsize=(5, 3))
        plt.plot(m_veh_exists, color='black')
        plt.title(str(sample_index[0]) + ' -- m_veh_exists')
        plt.grid()
        ######\######
        plt.figure(figsize=(5, 3))
        plt.plot(time_steps , em_delta_y, color='red')
        # plt.plot([0, 40], [-0.37, -0.37], color='green')
        # plt.plot([0, 40], [-1, -1], color='red')
        # plt.plot([0, 40], [-1.5, -1.5], color='red')
        plt.title(str(sample_index[0]) + ' -- em_delta_y')
        plt.grid()



# %%
"""
SINGLE VISUALISATION
"""
# %%
for i in range(10):
    plt.plot(displacement_seq[i, :, 0], color='gray')
plt.plot(dxs[sample_index[0], :, 0], color='red')


# %%
sample_index = [5755]
# sample_index = [train_samples[i]]
# sample_index = [i]
i += 1
e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
episode = future_idm_s[sample_index, 0, 0][0]

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
displacement_seq, act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_att, enc_h, \
                                             future_idm_ss, merger_cs])
f_att_seq, m_att_seq = att_scores[0].numpy(), att_scores[1].numpy()
act_seq = act_seq.numpy()
if np.isnan(act_seq).any():
    raise ValueError('There is nan in actions')
plt.figure(figsize=(5, 4))
episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
time_0 = int(history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0])

time_steps = range(time_0, time_0+69)
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
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
plt.plot(time_steps, traj, color='purple')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
plt.plot(time_steps, traj, color='black', linewidth=2)
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
plt.plot(time_steps, traj, color='red')
plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

for sample_trace_i in range(traces_n):
   plt.plot(time_steps[19:], act_seq[sample_trace_i, :, :].flatten(),
                                color='grey', alpha=0.4)
plt.scatter(time_steps[19+30], 0, color='red')
plt.text(30, -6, 'true: '+ str(true_params), color='red')
plt.text(30, -7, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))

plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

# %%









# %%
""" Set scientific plot format
"""
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])

MEDIUM_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

# %%
# %%
"""Prediction for an specific sample from the dataset
"""
traces_n = 50
model.forward_sim.attention_temp = 5
sample_index = [11540]
sample_index = [11086]
tf.random.set_seed(2021)

distribution_name = 'prior'
distribution_name = 'posterior'

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
displacement_seq, act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_att, enc_h, \
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
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
plt.plot(time_steps, traj, color='purple')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
plt.plot(time_steps, traj, color='black', linewidth=2)
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
plt.plot(time_steps, traj, color='red')
plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

for sample_trace_i in range(traces_n):
   plt.plot(time_steps[19:], act_seq[sample_trace_i, :, :].flatten(),
                                color='grey', alpha=0.4)
plt.scatter(time_steps[19+30], 0, color='red')
# plt.text(30, -6, 'true: '+ str(true_params), color='red')
# plt.text(30, -7, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))

plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

plt.figure(figsize=(5, 3))
# plt.plot(e_veh_att[:40] , color='black')
plt.plot(time_steps , e_veh_att, color='red')
plt.plot([time_steps[19], time_steps[19]], [0, 1], color='black')

for sample_trace_i in range(traces_n):
   plt.plot(time_steps[19:], m_att_seq[sample_trace_i, :].flatten(), color='grey')
plt.title(str(sample_index[0]) + ' -- Attention on merger')
# %%

"""
Visualisation of latent distribution for a given example in the dataset
"""
latent_dim = 3
from scipy.stats import norm
min_bound = z_idm.numpy().min()
max_bound = z_idm.numpy().max()
for dim in range(latent_dim):
    # plt.figure(figsize=(3, 2))
    datos = z_idm.numpy()[:, dim]
    (mu, sigma) = norm.fit(datos)
    x = np.linspace(min_bound, max_bound, 300)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, linewidth=2, label=sigma)
plt.legend()
    # plt.title('dim: '+str(dim) + ' sigma: '+str(round(sigma, 2)))
# %%
latent_dim = 3
from scipy.stats import norm
min_bound = z_att.numpy().min()
max_bound = z_att.numpy().max()
for dim in range(latent_dim):
    # plt.figure(figsize=(3, 2))
    datos = z_att.numpy()[:, dim]
    (mu, sigma) = norm.fit(datos)
    x = np.linspace(min_bound, max_bound, 300)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, linewidth=2, label=sigma)
plt.legend()
    # plt.title('dim: '+str(dim) + ' sigma: '+str(round(sigma, 2)))
# %%



fig, ax = plt.subplots(figsize=(5, 4))
time_axis = np.linspace(0., 7., 69)
for sample_trace_i in range(traces_n):
    label = '_nolegend_'
    if sample_trace_i == 0:
        label = 'NIDM'
    ax.plot(time_axis[19:], act_seq[sample_trace_i, :, :].flatten(), \
                    color='grey', alpha=0.4, linewidth=1, label=label, linestyle='-')

traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
ax.plot(time_axis, traj, color='red')
# traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
# ax.plot(time_axis, traj, linestyle='--', color='black')
# traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
# ax.plot(time_axis, traj, color='purple',linestyle='-')
ax.fill_between([0,3],[-9,-9], [5,5], color='lightgrey')
# ax.set_yticks([-6, -4, -2, 0, 2])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Long. Acceleration ($ms^{-2}$)')
ax.set_ylim(-6.5, 4.1)
ax.set_xlim(0, 8.1)
ax.grid(alpha=0.2)
ax.minorticks_off()
# ax.legend(['NIDM', 'Rear car', 'Merge car', 'Front car'], \
#            bbox_to_anchor=(0.25, 0.3))
ax.legend(['NIDM', 'True action'],  loc='lower left', framealpha=1, frameon=True)
ax.tick_params(top=False)
plt.savefig("example_actions.png", dpi=500)
# %%
fig, ax = plt.subplots(figsize=(5, 4))

for sample_trace_i in range(traces_n):
    label = '_nolegend_'
    if sample_trace_i == 0:
        label = 'NIDM'
    ax.plot(time_axis[19:], att_scores[sample_trace_i, :].flatten(), \
            color='grey', alpha=0.4, linewidth=1, label=label, linestyle='-')
ax.plot(time_axis, e_veh_att, color='red', linewidth=1.5, linestyle='-')
ax.fill_between([0,3],[-3,-3], [3,3], color='lightgrey')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, 8.1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('$w$')
ax.set_yticks([0., 0.5, 1])
ax.minorticks_off()
ax.grid(alpha=0.2)
ax.legend(['NIDM', 'True attentiveness'], framealpha=1, frameon=True)
plt.savefig("example_attention.png", dpi=500)


# %%

##########
# lATENT
ax= latent_vis()
ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, color='black')
idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, color='black')

ax.set_ylabel('$z_1$')
ax.set_xlabel('$z_2$')
# %%


# %%
"""
Visualisation of IDM param distribution for a given example in the dataset
"""
param_names = ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']
for i in range(5):
    true_val = true_params[i]
    param_name = param_names[i]

    datos = idm_params.numpy()[:, i]
    mean_val = datos.mean()
    (mu, sigma) = norm.fit(datos)
    x = np.linspace(mean_val-1, mean_val+1, 300)
    p = norm.pdf(x, mu, sigma)
    plt.figure(figsize=(3, 2))
    plt.plot(x, p, linewidth=2)
    plt.title(param_name + ' sigma: '+str(round(sigma, 2)) + ' mean: '+str(round(mu, 2)))
    # plt.axis('off')

    plt.plot([true_val, true_val], [0, p.max()], linewidth=2, color='red')
    # plt.plot([mean_bound, mean_bound], [0, p.max()], linewidth=2, color='black')

# %%
"""
Visualisation of IDM param distribution for a given example in the dataset

### quality figure ###
"""
param_names = ['desired_v', 'desired_tgap']

fig = plt.figure(figsize=(5, 4))
desired_v_axis = fig.add_subplot(211)
desired_tgap_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.4)

"""Desired speed"""
true_val = true_params[0]
param_samples = idm_params[:, 0]
(mu, sigma) = norm.fit(param_samples)
x = np.linspace(22, 26, 100)
p = norm.pdf(x, mu, sigma)

param_likelihoods = norm.pdf(idm_params[:, 0], mu, sigma)
desired_v_axis.plot(x, p, linewidth=1, color='red')
desired_v_axis.plot([true_val, true_val], [-0.1, 10],
            linewidth=1.5, color='black', linestyle='--')
desired_v_axis.scatter(param_samples, param_likelihoods, s=20, color='blue')
desired_v_axis.set_xticks([22, 23, 24, 25, 26])
desired_v_axis.set_ylim(-0.1, 5)
desired_v_axis.set_xlabel('$v_{des}$')
desired_v_axis.set_ylabel('Probability Density')
desired_v_axis.minorticks_off()
"""Desired Time Gap"""
true_val = true_params[1]
param_samples = idm_params[:, 1]
(mu, sigma) = norm.fit(param_samples)
x = np.linspace(0.25, 0.75, 100)
p = norm.pdf(x, mu, sigma)

param_likelihoods = norm.pdf(idm_params[:, 1], mu, sigma)
desired_tgap_axis.plot(x, p, linewidth=1, color='red')
desired_tgap_axis.plot([true_val, true_val], [-0.5, 35],
            linewidth=1.5, color='black', linestyle='--')
desired_tgap_axis.scatter(param_samples, param_likelihoods, s=20, color='blue')
desired_tgap_axis.set_xticks([0.3, 0.4, 0.5, 0.6])
desired_tgap_axis.set_ylim(-0.5, 30)
desired_tgap_axis.set_xlabel('$T_{des}$')
desired_tgap_axis.set_ylabel('Probability Density')
desired_tgap_axis.minorticks_off()
plt.savefig("example_params.png", dpi=500)



# %%
##########
# desired_vs = idm_params.numpy()[:, 0]
# desired_tgaps = idm_params.numpy()[:, 1]
# b_max = idm_params.numpy()[:, -1]
# fig = pyplot.figure(figsize=(3, 2))
# ax = Axes3D(fig)
#
# ax.scatter(19.2,  1., 2.6, color='red')
# ax.scatter(desired_vs, desired_tgaps, b_max, color='grey')
# ax.set_xlim(28, 30)
# ax.set_ylim(1, 2)
# ax.set_zlim(2, 3)
# ax.set_xticks([28., 19, 30])
# ax.set_yticks([1, 1.5, 2.])
# ax.set_zticks([2, 2.5, 3])
# # ax.set_title('Driver disposition')
# # ax.minorticks_off()
#
# ax.set_xlabel('$v_{des}$', labelpad=0)
# ax.set_ylabel('$T_{des}$', labelpad=1)
# ax.set_zlabel('$b_{max}$', labelpad=3)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.zaxis.set_rotate_label(False)
# ax.tick_params(axis='x', which='major', pad=0)
# ax.tick_params(axis='y', which='major', pad=0)
# ax.tick_params(axis='z', which='major', pad=0)
# ax.grid(False)
#
# plt.legend(['True parameter', 'Predicted parameters'])
# plt.savefig("example_params.png", dpi=500)
