import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
import os
import pickle
import sys
import json
from importlib import reload
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
sys.path.insert(0, './src')
# %%
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

# %%
"""
Needed methods
"""
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_samples(model, sample_index):
    """
    Unlike CVAE, here the latent represents the posterior distribution
    """
    h_seq = history_sca[sample_index, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    pos_params = model.belief_net(enc_h)
    sampled_z = model.belief_net.sample_z(pos_params)
    return sampled_z

def latent_vis(zsamples_n):
    fig = pyplot.figure(figsize=(4, 6))
    examples_to_vis = np.random.choice(val_examples, zsamples_n, replace=False)
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    sampled_z = latent_samples(model, examples_to_vis)
    aggressiveness = history_future_usc[examples_to_vis, 0, -7]
    color_shade = aggressiveness
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    ax.tick_params(pad=1)
    ax.grid(False)
    # ax.view_init(30, 50)
    #===============
    #  Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    axins = inset_axes(ax,
                        width="5%",
                        height="90%",
                        loc='right',
                        borderpad=-2
                       )

    fig.colorbar(att_sc, cax=axins)
    cbar = fig.colorbar(att_sc, cax=axins)
    ax.tick_params(pad=1)
    ax.grid(False)
    ax.view_init(30, 50)
    # ax.set_xlabel('$z_{1}$', labelpad=1)
    # ax.set_ylabel('$z_{2}$', labelpad=1)
    # ax.set_zlabel('$z_{3}$', labelpad=1)
    plt.subplots_adjust(wspace=0.2, hspace=None)

def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    """ Returns the state sequence. It also deletes the middle index, which is
        the transition point from history to future.
    """
    # data shape: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 49, axis=1)
    return traj.flatten()
# %%
"""
Load data
"""
history_len = 50 # steps
rollout_len = 50
data_id = '029'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'
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
train_examples = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
# history_sca.shape
train_examples.shape
# %%
"""
Load model (with config file)
"""
model_name = 'latent_mlp_03'
epoch_count = '20'
exp_path = './src/models/experiments/'+model_name+'/model_epo'+epoch_count
exp_dir = os.path.dirname(exp_path)
with open(exp_dir+'/'+'config.json', 'rb') as handle:
    config = json.load(handle)
    print(json.dumps(config, ensure_ascii=False, indent=4))

from models.core import latent_mlp
reload(latent_mlp)
from models.core.latent_mlp import  LatentMLP
model = LatentMLP(config)
model.load_weights(exp_path).expect_partial()

with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
    model.forward_sim.env_scaler = pickle.load(handle)

with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
    model.forward_sim.dummy_value_set = pickle.load(handle)
# model.forward_sim.attention_temp = 10
# %%
with open(exp_dir+'/'+'losses.pickle', 'rb') as handle:
    losses = pickle.load(handle)
plt.figure()
plt.plot(losses['test_llloss'], label='test_llloss')
plt.plot(losses['train_llloss'], label='train_llloss')
plt.grid()
plt.legend()
plt.figure()
plt.plot(losses['test_klloss'], label='test_klloss')
plt.plot(losses['train_klloss'], label='train_klloss')
plt.legend()
plt.grid()


# %%
"""
Latent visualisation - aggressiveness used for color coding the latent samples
"""
latent_vis(zsamples_n=3000)
# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
Example_pred = 0
i = 0
covered_episodes = []
traces_n = 2
sepcific_examples = []

"""
Posterior is used, as encoder here is an inference network
that estimates the latent vehicle state.
"""
distribution_name = 'posterior'
# distribution_name = 'prior'

# for i in bad_examples[0]:
# for i in sepcific_examples:
# for i in bad_zs:
# for i in bad_examples[0]:
while Example_pred < 30:
    "ENSURE ONLY VAL SAMPLES CONSIDERED"
    sample_index = [val_examples[i]]
    # sample_index = [train_examples[i]]
    # sample_index = [i]
    i += 1
    e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
    m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode not in covered_episodes:
    # if 4 == 4:
    if episode not in covered_episodes and e_veh_att[55:65].mean() > 0:
        covered_episodes.append(episode)
        merger_cs = vectorise(future_m_veh_c[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        if distribution_name == 'posterior':
            h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
            enc_h = model.h_seq_encoder(h_seq)
            pos_params = model.belief_net(enc_h)
            sampled_z = model.belief_net.sample_z(pos_params)

        elif distribution_name == 'prior':
            prior = tfd.Normal(loc=tf.zeros([traces_n, model.belief_net.latent_dim]), scale=1)
            sampled_z = prior.sample()

        sampled_z = tf.reshape(\
                        sampled_z, [traces_n, 1, model.belief_net.latent_dim])
        _, _, act_seq = model.forward_sim.rollout([sampled_z, \
                                        future_idm_ss, merger_cs], training=False)
        act_seq = act_seq.numpy()
        plt.figure(figsize=(5, 3))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = int(history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0])
        time_steps = range(time_0, time_0+99)
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
        plt.text(0.1, 0.5,
                        'episode_id: '+ info[0] +
                        'time_0: '+ info[1] +
                        'e_veh_id: '+ info[2] +
                        'aggressiveness: '+ info[3] +
                        'speed: '+ str(future_idm_ss[0, 0, 0])
                            , fontsize=10)

        plt.figure(figsize=(5, 3))
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
        plt.plot(time_steps, traj, color='purple')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
        plt.plot(time_steps, traj, color='black', linewidth=2)
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
        plt.plot(time_steps, traj, color='red')
        plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[49:], act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.4)
           # plt.plot(range(29, 59), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        Example_pred += 1
