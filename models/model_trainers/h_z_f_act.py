import matplotlib.pyplot as plt
from importlib import reload
import pickle
import sys
import os

reload(plt)
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(suppress=True)

# %%
"""
Data prep
"""
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge()
with open('./models/experiments/sim_data_024.pickle', 'rb') as handle:
    features = pickle.load(handle)
features, dummy_value_set = data_gen.fill_missing_values(features)
features_scaled, env_scaler, m_scaler = data_gen.scale_data(features)

history_future_seqs = data_gen.sequence(features, 20, 20)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 20)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays

# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 512,
    },
    "exp_id": "NA",
    "Note": ""
}
class Trainer():
    def __init__(self, training_data, model_type, model_name):
        self.model = None
        self.model_type = model_type
        self.model_name = model_name

        self.train_mseloss = []
        self.train_klloss = []

        self.test_mseloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()
        self.prep_data(training_data)
#
    def initiate_model(self):
        if self.model_name == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import  NeurIDMModel
            self.model = NeurIDMModel(config)

        elif self.model_name == 'h_z_f_act':
            from models.core import h_z_f_act
            reload(h_z_f_act)
            from models.core.h_z_f_act import  NeurLatentModel
            self.model = NeurLatentModel(config)

        elif self.model_name == 'h_lat_act':
            from models.core import h_z_f_act
            reload(h_z_f_act)
            from models.core.h_z_f_act import NeurLatentModelOneStep
            self.model = NeurLatentModelOneStep(config)

        with open('./models/experiments/env_scaler_024.pickle', 'rb') as handle:
            self.model.forward_sim.env_scaler = pickle.load(handle)

        with open('./models/experiments/m_scaler_024.pickle', 'rb') as handle:
            self.model.forward_sim.m_scaler = pickle.load(handle)

    def prep_data(self, training_data):
        all_epis = np.unique(training_data[0][:, 0, 0])
        np.random.seed(2021)
        np.random.shuffle(all_epis)
        train_epis = all_epis[:int(len(all_epis)*0.7)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_indxs = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

        _, history_sca, future_sca, future_idm_s,\
                    future_m_veh_c, future_e_veh_a = training_data

        if self.model_type == 'cvae':
            self.train_input = [history_sca[train_indxs, :, 2:],
                        future_sca[train_indxs, :, 2:],
                        future_idm_s[train_indxs, :, 2:],
                        future_m_veh_c[train_indxs, :, 2:],
                        future_e_veh_a[train_indxs, :, 2:]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                        future_sca[val_indxs, :, 2:],
                        future_idm_s[val_indxs, :, 2:],
                        future_m_veh_c[val_indxs, :, 2:],
                        future_e_veh_a[val_indxs, :, 2:]]

        elif self.model_type == 'lstm_model':
            self.train_input = [history_sca[train_indxs, :, 2:],
                                        future_e_veh_a[train_indxs, 0, -1]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                                        future_e_veh_a[val_indxs, 0, -1]]

        elif self.model_type == 'mlp_model':
            history_sca = np.squeeze(history_sca)
            future_e_veh_a = np.squeeze(future_e_veh_a)

            self.train_input = [history_sca[train_indxs, 2:],
                                        future_e_veh_a[train_indxs, -1]]

            self.val_input = [history_sca[val_indxs, 2:],
                                        future_e_veh_a[val_indxs, -1]]

    def train(self, epochs):
        for epoch in range(epochs):
            self.epoch_count += 1
            self.model.train_loop(self.train_input)
            self.model.test_loop(self.val_input, epoch)
            if self.model_type == 'cvae':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))
            else:
                self.train_mseloss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_loss.result().numpy().item(), 2))
            print(self.epoch_count, 'epochs completed')

    def save_model(self, model_name, exp_id):
        model_name += exp_id + '_epo_'+str(self.epoch_count)
        print(model_name)
        exp_dir = './models/experiments/'+model_name+'/model'
        if not os.path.exists('./models/experiments/'+model_name):
            self.model.save_weights(exp_dir)
        else:
            print('This model is already saved')


model_trainer = Trainer(data_arrays, model_type='cvae', model_name='h_z_f_act')
# exp_dir = './models/experiments/'+'h_z_f_act009_epo_15'+'/model'
# model_trainer.model.load_weights(exp_dir).expect_partial()
# model_trainer.train(epochs=1)
# model_trainer.test_mseloss
# %%
#
# model_trainer.train(epochs=5)
#
# fig = plt.figure(figsize=(15, 5))
plt.style.use('default')
#
# mse_axis = fig.add_subplot(131)
# kl_axis = fig.add_subplot(132)
# idm_kl_axis = fig.add_subplot(133)
# mse_axis.plot(model_trainer.test_mseloss)
# mse_axis.plot(model_trainer.train_mseloss)
# %%
all_epis = np.unique(history_sca[:, 0, 0])
np.random.seed(2021)
np.random.shuffle(all_epis)
train_epis = all_epis[:int(len(all_epis)*0.7)]
val_epis = np.setdiff1d(all_epis, train_epis)

train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
history_sca.shape
train_indxs.shape
val_examples.shape
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_c = np.float32(future_m_veh_c)
# np.count_nonzero(np.isnan(history_sca))
# %%
model_trainer.model.vae_loss_weight = 0.1
################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(epochs=5)
################## ##### ##################
################## ##### ##################
################## ##### ##################
################## MSE LOSS ##################
fig = plt.figure(figsize=(15, 5))
# plt.style.use('default')

mse_axis = fig.add_subplot(121)
kl_axis = fig.add_subplot(122)
mse_axis.plot(model_trainer.test_mseloss)
mse_axis.plot(model_trainer.train_mseloss)

mse_axis.grid()
mse_axis.set_xlabel('epochs')
mse_axis.set_ylabel('loss (MSE)')
mse_axis.set_title('MSE')
mse_axis.legend(['test', 'train'])

################## kl LOSS ##################
kl_axis.plot(model_trainer.test_klloss)
kl_axis.plot(model_trainer.train_klloss)

kl_axis.grid()
kl_axis.set_xlabel('epochs')
kl_axis.set_ylabel('loss (kl)')
kl_axis.set_title('kl')
kl_axis.legend(['test', 'train'])
print(model_trainer.test_mseloss[-1])


#
# %%
model_trainer.save_model('h_z_f_act', '017')

# %%
"""
Find bad examples
"""

import tensorflow as tf
# examples_to_vis = val_examples
# val_examples.shape
def get_avg_loss_across_sim(examples_to_vis):
    val_input = [history_sca[examples_to_vis , :, 2:],
                future_sca[examples_to_vis, :, 2:],
                future_idm_s[examples_to_vis, :, 2:],
                future_m_veh_c[examples_to_vis, :, 2:]]
    act_pred, pri_params, pos_params = model_trainer.model(val_input)
    loss = (tf.square(tf.subtract(act_pred, future_e_veh_a[examples_to_vis, :, 2:])))
    # loss = (tf.abs(tf.subtract(act_pred, future_e_veh_a[examples_to_vis, :, 2:])))
    loss = tf.reduce_mean(loss, axis=1).numpy()
    return loss

loss = get_avg_loss_across_sim(val_examples[0:10000])
_ = plt.hist(loss, bins=150)
# _ = plt.hist(loss[loss<0.1], bins=150)
bad_examples = np.where(loss > 0.1)

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_samples(model_trainer, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    sdv_actions = future_m_veh_c[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
    sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_z

def latent_vis(n_z_samples):
    fig = pyplot.figure(figsize=(4, 6))
    examples_to_vis = np.random.choice(val_examples, n_z_samples, replace=False)

    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    sampled_z = latent_samples(model_trainer, examples_to_vis)
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
latent_vis(2000)

# %%
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])

# %%

# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 19, axis=1)
    return traj.flatten()

def get_e_veh_att(e_veh_id, e_veh_decision, e_veh_att):
    atten_on_ego = np.where(e_veh_att == e_veh_id)
    e_veh_changing_lane = np.where(e_veh_decision != 0)
    atten_on_e_veh_changing_lane = np.intersect1d(atten_on_ego, e_veh_changing_lane)
    e_veh_att = np.ones(40)
    e_veh_att[atten_on_e_veh_changing_lane] = 0
    return e_veh_att

hf_usc_indexs = {}
col_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'm_veh_exists', 'e_veh_att',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'e_veh_action', 'f_veh_action', 'm_veh_action',
         'aggressiveness',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
         'em_delta_y', 'delta_x_to_merge']

index = 0
for item_name in col_names:
    hf_usc_indexs[item_name] = index
    index += 1

# %%

Example_pred = 0
i = 0
covered_episodes = []
traces_n = 50
sepcific_examples = []

# for i in bad_examples[0]:
# for i in sepcific_examples:
# for i in bad_zs:
# for i in bad_examples[0][0:10]:
while Example_pred < 10:
    "ENSURE ONLY VAL SAMPLES CONSIDERED"
    sample_index = [val_examples[i]]
    # sample_index = [train_indxs[i]]
    # sample_index = [i]
    i += 1
    e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
    m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode not in covered_episodes:
    # if 4 == 4:
    if episode not in covered_episodes and e_veh_att[25:35].mean() > 0:
        covered_episodes.append(episode)
        sdv_actions = vectorise(future_m_veh_c[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
        sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
        proj_belief = model_trainer.model.belief_net.belief_proj(sampled_z)
        act_seq = model_trainer.model.forward_sim.rollout([proj_belief, \
                                                    future_idm_ss, sdv_actions])
        act_seq = act_seq.numpy()

        plt.figure(figsize=(5, 3))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
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
        plt.plot(traj, color='purple')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
        plt.plot(traj, color='black')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
        plt.plot(traj, color='red')
        plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.5)
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        Example_pred += 1

# %%
