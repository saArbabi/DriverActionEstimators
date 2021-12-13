import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
import os
import pickle
import sys
import json
from importlib import reload

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
    h_seq = history_sca[sample_index, :, 2:]
    merger_cs = future_m_veh_c[sample_index, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
    z_idm, z_att = model.belief_net.sample_z(latent_dis_param)
    return [z_idm, z_att]

def latent_vis(zsamples_n):
    examples_to_vis = np.random.choice(val_examples, zsamples_n, replace=False)
    aggressiveness = history_future_usc[examples_to_vis, 0, -7]
    color_shade = aggressiveness
    z_s = latent_samples(model, examples_to_vis)

    for z_ in z_s:
        fig = pyplot.figure(figsize=(4, 6))
        #===============
        #  First subplot
        #===============
        # set up the axes for the first plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        att_sc = ax.scatter(z_[:, 0], z_[:, 1], z_[:, 2],
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
        att_sc = ax.scatter(z_[:, 0], z_[:, 1], z_[:, 2],
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
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 29, axis=1)
    return traj.flatten()
# %%
"""
Load data
"""
history_len = 30 # steps
rollout_len = 30
data_id = '028'
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
model_name = 'neural_idm_107'
epoch_count = '20'
exp_path = './src/models/experiments/'+model_name+'/model_epo'+epoch_count
exp_dir = os.path.dirname(exp_path)
with open(exp_dir+'/'+'config.json', 'rb') as handle:
    config = json.load(handle)
    print(json.dumps(config, ensure_ascii=False, indent=4))

from models.core import neural_idm
reload(neural_idm)
from models.core.neural_idm import NeurIDMModel
model = NeurIDMModel(config)
model.load_weights(exp_path).expect_partial()

with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
    model.forward_sim.env_scaler = pickle.load(handle)

with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
    model.forward_sim.dummy_value_set = pickle.load(handle)
# model.forward_sim.attention_temp = 10
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
# for name in ['neural_idm_105', 'neural_idm_106']:
for name in ['neural_idm_107']:
    with open('./src/models/experiments/'+name+'/'+'losses.pickle', 'rb') as handle:
        losses[name] = pickle.load(handle)

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_mseloss'], label=name)
    # plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_klloss'], label=name)
    # plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()



# %%
"""
Find bad examples
"""
import tensorflow as tf
# examples_to_vis = val_examples
# val_examples.shape
def get_avg_loss_across_sim(examples_to_vis):
    merger_cs = future_m_veh_c[examples_to_vis, :, 2:]
    h_seq = history_sca[examples_to_vis, :, 2:]
    future_idm_ss = future_idm_s[examples_to_vis, :, 2:]
    enc_h = model.h_seq_encoder(h_seq)
    latent_dis_param = model.belief_net(enc_h, dis_type='prior')
    sampled_z = model.belief_net.sample_z(latent_dis_param)
    proj_belief = model.belief_net.belief_proj(sampled_z)
    idm_params = model.idm_layer(proj_belief)
    act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_belief, \
                                            future_idm_ss, merger_cs])
    true_actions = future_e_veh_a[examples_to_vis, :, 2:]
    loss = (tf.square(tf.subtract(act_seq, true_actions)))**0.5
    return tf.reduce_mean(loss, axis=1).numpy()
# loss = get_avg_loss_across_sim(train_examples[0:15])
loss = get_avg_loss_across_sim(val_examples[0:5000])
_ = plt.hist(loss, bins=150)
# _ = plt.hist(loss[loss<0.1], bins=150)
bad_examples = np.where(loss > 1)

# %%

# %%
"""
Latent visualisation - aggressiveness used for color coding the latent samples
"""
latent_vis(zsamples_n=3000)
latent_vis(zsamples_n=3000)
# plt.savefig("latent.png", dpi=500)
# %%
# import matplotlib.pyplot as plt
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# params = {
#           'font.size' : 20,
#           'font.family' : 'EB Garamond',
#           }
# plt.rcParams.update(params)
# plt.style.use(['science','ieee'])

# %%

# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
Example_pred = 0
i = 0
covered_episodes = []
model.forward_sim.attention_temp = 5
traces_n = 50
# np.where((history_future_usc[:, 0, 0] == 26) & (history_future_usc[:, 0, 2] == 4))
sepcific_examples = []
distribution_name = 'prior'
# distribution_name = 'posterior'
# for i in bad_examples[0]:
# for i in sepcific_examples:
# for i in [2815]:
# for i in bad_examples[0]:
while Example_pred < 20:
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
    # if episode not in covered_episodes and e_veh_att[0:3].mean() > 0 and e_veh_att[-5:].mean() == 0:
    if episode not in covered_episodes and e_veh_att[25:35].mean() > 0:
    # if episode not in covered_episodes and e_veh_att[25:35].mean() > 0:
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
        act_seq, att_scores = model.forward_sim.rollout([idm_params, proj_att, \
                                                     future_idm_ss, merger_cs])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

        plt.figure(figsize=(5, 4))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = int(history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0])
        time_steps = range(time_0, time_0+59)
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
           plt.plot(time_steps[29:], act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.4)
           # plt.plot(time_steps[29:], act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        # plt.plot(e_veh_att[:40] , color='black')
        plt.plot(time_steps , e_veh_att, color='red')

        plt.plot([time_steps[29], time_steps[29]], [0, 1], color='black')

        for sample_trace_i in range(traces_n):
           plt.plot(time_steps[29:], att_scores[sample_trace_i, :].flatten(), color='grey')
        plt.title(str(sample_index[0]) + ' -- Attention')

        ##########
        # m_veh_id = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_id'])
        # plt.figure(figsize=(5, 3))
        # plt.plot(m_veh_id, color='black')
        # plt.title(str(sample_index[0]) + ' -- m_veh_id')
        # plt.grid()
        ######\######

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
        ############
        ##########
        # lATENT
        # ax = latent_vis(2000)
        # ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, color='black')
        ##########

        """
        # plt.plot(desired_vs)
        # plt.grid()
        # plt.plot(desired_tgaps)
        # plt.grid()

        plt.figure(figsize=(5, 3))
        desired_vs = idm_params.numpy()[:, 0]
        desired_tgaps = idm_params.numpy()[:, 1]
        plt.scatter(desired_vs, desired_tgaps, color='grey')

        plt.scatter(24.7, 1.5, color='red')
        plt.xlim(15, 40)
        plt.ylim(0, 3)
        #
        # plt.scatter(30, 1, color='red')
        # plt.xlim(25, 35)
        # plt.ylim(0, 2)

        plt.title(str(sample_index[0]) + ' -- Param')
        plt.grid()
        """
        Example_pred += 1
# %%
"""Prediction for an specific sample from the dataset
"""
# model.arbiter.attention_temp = 5
traces_n = 50
model.forward_sim.attention_temp = 20
# sample_index = [12374]
sample_index = [12754]
e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
episode = future_idm_s[sample_index, 0, 0][0]
# merger_cs.shape
# plt.plot(history_future_usc[sample_index, 20: ,hf_usc_indexs['delta_x_to_merge']][0])
# plt.plot(merger_cs[0, :, -1])

merger_cs = vectorise(future_m_veh_c[sample_index, :, 2:], traces_n)
# merger_cs[:, 10, -1] = 0
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model.h_seq_encoder(h_seq)
latent_dis_param = model.belief_net(enc_h, dis_type='prior')
sampled_z = model.belief_net.sample_z(latent_dis_param)
proj_belief = model.belief_net.belief_proj(sampled_z)
act_seq, att_scores, idm_params_seq = model.forward_sim.rollout([proj_belief, \
                                             future_idm_ss, merger_cs])
#
# min_act = idm_params.numpy()[:, -2]
# plt.scatter(min_act, [0]*100)
# plt.scatter(1.43, 0, color='red')
# idm_params = tf.ones([100, 5])*[29.9, 1.01, 1.02, 3.83, 3.83]

# idm_params = tf.ones([100, 5])*[18., 1.11, 4, 1., 1]
# idm_params = idm_params.numpy()
# idm_params[:, -2] = 3.38
# idm_params[:, -1] = 3.97
# idm_params[:, 1] += 0.4
act_seq, att_scores, idm_params_seq = \
        act_seq.numpy(), att_scores.numpy(), idm_params_seq.numpy()
att_ = att_scores.flatten()

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
idm_names = ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']
for param_name in idm_names:
    true_pram_val = history_future_usc[sample_index, 0, hf_usc_indexs[param_name]][0]
    true_params.append(round(true_pram_val, 2))

plt.text(0.1, 0.3, 'true: '+ str(true_params)) #True
# plt.text(0.1, 0.1, 'pred: '+ str(idm_params[:, :].mean(axis=0).round(2)))
plt.text(0.1, 0.1, 'pred: '+ str(idm_params_seq[:, 0, :].mean(axis=0).round(2)))

##########
for i, param_name in enumerate(idm_names):
    plt.figure()
    for n in range(5):
        plt.plot(idm_params_seq[n, :, i])
    plt.title(param_name)
    plt.grid()

# %%

# plt.figure(figsize=(10, 10))
time_axis = np.linspace(0., 4., 59)
# plt.figure(figsize=(5, 3))
# plt.legend(['Leader', 'Follower', 'Merger'])

for sample_trace_i in range(traces_n):
   plt.plot(time_axis[29:], act_seq[sample_trace_i, :, :].flatten(), \
                    color='grey', alpha=0.5, linewidth=0.5, label='_nolegend_', linestyle='-')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
plt.plot(time_axis, traj, color='red')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
plt.plot(time_axis, traj, linestyle='--', color='black')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
plt.plot(time_axis, traj, color='purple',linestyle='-')
# plt.plot([3,3],[-2,1])
# plt.title('Vehicle actions')
# plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration ($ms^{-2}$)')
# plt.ylim(-0.1, 0.1)
# plt.yticks([1, 0., -1, -2])
# plt.xticks([0., 2, 4, 6])

plt.grid()
# plt.grid(alpha=0.1)
plt.legend(['Ego', 'Merger', 'Leader'])
# plt.savefig("example_actions.png", dpi=500)
# %%
# plt.figure(figsize=(10, 10))
plt.figure(figsize=(5, 4))
for sample_trace_i in range(traces_n):
   plt.plot(time_axis[29:], att_scores[sample_trace_i, :].flatten(), \
            color='grey', alpha=0.5, linewidth=0.5, label='_nolegend_', linestyle='-')
plt.plot(time_axis, e_veh_att, color='red', linewidth=1, linestyle='-')

plt.ylim(-0.05, 1.05)
plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('$w$')
# plt.title('Driver attentiveness')
plt.legend(['True attnetion'])
plt.xticks([0., 2, 4, 6])
plt.yticks([0., 0.5, 1])
# plt.xlim(0, 1)
plt.minorticks_off()
plt.grid(alpha=0.1)
# plt.savefig("example_attention.png", dpi=500)


# %%

##########
# lATENT
ax= latent_vis()
ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, color='black')
idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, color='black')

ax.set_ylabel('$z_1$')
ax.set_xlabel('$z_2$')

# %%
##########
# desired_vs = idm_params.numpy()[:, 0]
# desired_tgaps = idm_params.numpy()[:, 1]
# b_max = idm_params.numpy()[:, -1]
# fig = pyplot.figure(figsize=(3, 2))
# ax = Axes3D(fig)
#
# ax.scatter(29.2,  1., 2.6, color='red')
# ax.scatter(desired_vs, desired_tgaps, b_max, color='grey')
# ax.set_xlim(28, 30)
# ax.set_ylim(1, 2)
# ax.set_zlim(2, 3)
# ax.set_xticks([28., 29, 30])
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

# %%
# %%
"""
Visualisation of latent distribution for a given example in the dataset
"""
latent_dim = 3
from scipy.stats import norm
for dim in range(latent_dim):
    plt.figure(figsize=(3, 2))
    datos = sampled_z.numpy()[:, dim]
    (mu, sigma) = norm.fit(datos)
    datos.std()
    x = np.linspace(-1, 2, 300)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, linewidth=2)
    plt.title('dim: '+str(dim) + ' sigma: '+str(round(sigma, 2)))

# %%
"""
Visualisation of IDM param distribution for a given example in the dataset
"""
parameter_range = {'most_aggressive': {
                                'desired_v':25, # m/s
                                'desired_tgap': 0.5, # s
                                'min_jamx':1, # m
                                'max_act':4, # m/s^2
                                'min_act':4, # m/s^2
                                'politeness':0.,
                                'safe_braking':-5,
                                'act_threshold':0
                                },
                 'least_aggressvie': {
                                'desired_v':15, # m/s
                                'desired_tgap':2, # s
                                'min_jamx':5, # m
                                'max_act':2, # m/s^2
                                'min_act':2, # m/s^2
                                'politeness':1,
                                'safe_braking':-3,
                                'act_threshold':0.2
                                 }}
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