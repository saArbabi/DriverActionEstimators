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
# viewer = Viewer(config)
data_config = {
                'future_scaeq_length':20,
                'history_scaeq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
data_arrays = data_gen.prep_data()
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_merger_a, future_follower_a = data_arrays

# %%


# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 256,
    },
    "exp_id": "NA",
    "Note": ""
}

class Trainer():
    def __init__(self, model_type):
        self.model = None
        self.model_type = model_type
        self.train_loss = []
        self.valid_loss = []

        self.train_mseloss = []
        self.train_klloss = []

        self.valid_mseloss = []
        self.valid_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self, model_type=None):
        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModel
        self.model = NeurIDMModel(config)

    def train(self, training_data, epochs):
        train_sample_index = int(len(training_data[0])*0.8)
        self.model.epochs_n = epochs

        _, history_sca, future_sca, future_idm_s,\
                future_merger_a, future_follower_a = training_data


        train_input = [history_sca[0:train_sample_index, :, 1:],
                    future_sca[0:train_sample_index, :, 1:],
                    future_idm_s[0:train_sample_index, :, 1:],
                    future_merger_a[0:train_sample_index, :, 1:],
                    future_follower_a[0:train_sample_index, :, 1:]]

        val_input = [history_sca[train_sample_index:, :, 1:],
                    future_sca[train_sample_index:, :, 1:],
                    future_idm_s[train_sample_index:, :, 1:],
                    future_merger_a[train_sample_index:, :, 1:],
                    future_follower_a[train_sample_index:, :, 1:]]

        for epoch in range(epochs):
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'vae_idm' or self.model_type == 'driver_model':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.valid_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.valid_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))
            else:
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
            print(self.epoch_count, 'epochs completed')
            self.epoch_count += 1

    def save_model(self, model_name):
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

model_trainer = Trainer(model_type='driver_model')

# %%
model_trainer.model.vae_loss_weight = 0.1
model_trainer.train(data_arrays, epochs=5)
plt.figure()
plt.plot(model_trainer.valid_mseloss)
plt.plot(model_trainer.train_mseloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
plt.title('MSE')

plt.figure()
plt.plot(model_trainer.valid_klloss)
plt.plot(model_trainer.train_klloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (KL)')
plt.title('KL')


# %%
# %%


np.random.seed(2020)
history_future_usc, history_sca, future_sca, future_idm_s, \
                                future_merger_a, future_follower_a = data_arrays
train_sample_index = int(len(history_sca)*0.8)
val_examples = range(train_sample_index, len(history_sca))


history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_merger_a = np.float32(future_merger_a)

timid_drivers = []
normal_drivers = []
aggressive_drivers = []
for sample_index in val_examples:
    if history_future_usc[sample_index, -1, -3] == 0:
       timid_drivers.append(sample_index)
    elif history_future_usc[sample_index, -1, -3] == 0.5:
       normal_drivers.append(sample_index)
    elif history_future_usc[sample_index, -1, -3] == 1:
       aggressive_drivers.append(sample_index)
history_sca.shape
len(timid_drivers)
len(normal_drivers)
len(aggressive_drivers)
# %%
def latent_samples(model_trainer, sample_index):
    sdv_actions = future_merger_a[sample_index, :, 1:]
    h_seq = history_sca[sample_index, :, 1:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    enc_acts = model_trainer.model.act_encoder(sdv_actions)
    prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
    sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_att_z, sampled_idm_z



fig = plt.figure(figsize=(7, 7))
att_axis = fig.add_subplot(211)
idm_axs = fig.add_subplot(212)
sampled_att_z, sampled_idm_z = latent_samples(model_trainer, aggressive_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='red')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='red')
#
sampled_att_z, sampled_idm_z = latent_samples(model_trainer, timid_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='green')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='green')

# plt.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='green')
sampled_att_z, sampled_idm_z = latent_samples(model_trainer, normal_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='orange')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='orange')
# plt.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='orange')
# # plt.scatter(z[:, 0], z[:, 1], s=20, color='blue')

# att_axis.set_ylabel('$z_1$')
# att_axis.set_xlabel('$z_2$')

# %%

# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def get_true_attention(ego_id, ego_decision, follower_atten):
    atten_on_ego = np.where(follower_atten == ego_id)
    ego_changing_lane = np.where(ego_decision != 0)
    atten_on_ego_changing_lane = np.intersect1d(atten_on_ego, ego_changing_lane)
    true_attention = np.ones(40)
    true_attention[atten_on_ego_changing_lane] = 0
    return true_attention

# model_trainer.model.idm_sim.arbiter.attention_temp = 20
model_trainer.model.arbiter.attention_temp = 20

Example_pred = 0
traces_n = 20
i = 0
covered_episodes = []
col_names = ['episode_id', 'time_step', 'ego_decision', \
        'leader_action', 'follower_action', 'merger_action', \
        'lane_y', 'follower_aggress', \
        'follower_atten', 'veh_id', 'follower_id']
indxs = {}
for i in range(len(col_names)):
    indxs[col_names[i]] = i

while Example_pred < 10:
    sample_index = [timid_drivers[i]]
    # sample_index = [normal_drivers[i]]
    # sample_index = [aggressive_drivers[i]]
    i += 1

    ego_id = history_future_usc[sample_index, 0, indxs['veh_id']]
    ego_decision = history_future_usc[sample_index, :, indxs['ego_decision']][0]
    follower_atten = history_future_usc[sample_index, :, indxs['follower_atten']][0]
    true_attention = get_true_attention(ego_id, ego_decision, follower_atten)
    # if true_attention.mean() != 0:
    # plt.figure()
    # plt.plot(ego_decision)
    # plt.plot(true_attention)
    # plt.plot(ego_decision)
    lane_y = history_future_usc[sample_index, :, indxs['lane_y']][0]
    episode = future_idm_s[sample_index, 0, 0][0]

    if episode not in covered_episodes and 0 < true_attention[:30].mean() < 1:
        # if episode not in covered_episodes and true_attention[30:].mean() == 0 and true_attention[:30].mean() == 1:
        covered_episodes.append(episode)

        true_action = history_future_usc[sample_index, :, indxs['follower_action']][0]

        sdv_actions = vectorise(future_merger_a[sample_index, :, 1:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 1:], traces_n)
        f_seq_unscaled = vectorise(future_idm_s[sample_index, :, 1:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        enc_acts = model_trainer.model.act_encoder(sdv_actions)
        prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
        sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
        att_scores =  model_trainer.model.arbiter(sampled_att_z)

        idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
        idm_params = tf.reshape(idm_params, [traces_n, 1, 5])
        idm_params = tf.repeat(idm_params, 20, axis=1)

        act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, f_seq_unscaled])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
        plt.figure()
        plt.plot(history_future_usc[sample_index, :, indxs['leader_action']][0], color='purple')
        plt.plot(history_future_usc[sample_index, :, indxs['follower_action']][0], color='red')
        plt.plot(history_future_usc[sample_index, :, indxs['merger_action']][0], color='orange')
        plt.legend(['leader_action', 'follower_action', 'merger_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), color='grey')
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure()
        plt.plot(true_attention[:20] , color='black')
        plt.plot(range(20, 40), true_attention[20:], color='red')

        for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
        plt.ylim(-0.1, 1.1)
        plt.title(str(sample_index[0]) + ' -- Attention')
        plt.grid()

        ##########

        # plt.plot(desired_vs)
        # plt.grid()
        # plt.plot(desired_tgaps)
        # plt.grid()

        plt.figure()
        desired_vs = idm_params.numpy()[:, 0, 0]
        desired_tgaps = idm_params.numpy()[:, 0, 1]
        plt.scatter(desired_vs, desired_tgaps, color='grey')

        plt.scatter(19.4, 2, color='green')
        # plt.scatter(25, 1.4, color='orange')
        # plt.scatter(30, 1, color='red')
        plt.xlim(15, 40)
        plt.ylim(0, 3)
        #
        # plt.scatter(30, 1, color='red')
        # plt.xlim(25, 35)
        # plt.ylim(0, 2)

        plt.title(str(sample_index[0]) + ' -- Param')
        plt.grid()

        ##########
        plt.figure()
        plt.plot(lane_y[:20], color='black')
        plt.plot(range(20, 40), lane_y[20:], color='red')
        # plt.plot([0, 40], [-0.37, -0.37], color='green')
        # plt.plot([0, 40], [-1, -1], color='red')
        plt.plot([0, 40], [-1.5, -1.5], color='red')
        plt.title(str(sample_index[0]) + ' -- lane_y')
        plt.grid()
        ############
        Example_pred += 1


# %%

# %%
"""Single sample Anticipation visualisation
"""
model_trainer.model.arbiter.attention_temp = 20
traces_n = 20
sample_index = [6014]

true_attention = future_follower_a[sample_index, :, -2].flatten()
m_y = future_idm_s[sample_index, :, -2].flatten()
episode = future_idm_s[sample_index, 0, 0][0]


true_action = future_follower_a[sample_index, :, -1].flatten()

sdv_actions = vectorise(future_merger_a[sample_index, :, 1:], traces_n)
h_seq = vectorise(history_sca[sample_index, :, 1:], traces_n)
f_seq_unscaled = vectorise(future_idm_s[sample_index, :, 1:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
enc_acts = model_trainer.model.act_encoder(sdv_actions)
prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
att_scores =  model_trainer.model.arbiter([sampled_att_z, enc_h])

idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
idm_params = tf.reshape(idm_params, [traces_n, 1, 5])
idm_params = tf.repeat(idm_params, 20, axis=1)

act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, f_seq_unscaled])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
plt.figure()
for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), color='grey')
   # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')
plt.plot(true_action[:20].flatten(), color='black')
plt.plot(range(20, 40), true_action[20:].flatten(), color='red')
plt.ylim(-3, 3)
plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

plt.figure()
plt.plot(true_attention[:20] , color='black')
plt.plot(range(20, 40), true_attention[20:], color='red')

for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
plt.ylim(-0.1, 1.1)
plt.title(str(sample_index[0]) + ' -- Attention')
plt.grid()

##########

# plt.plot(desired_vs)
# plt.grid()
# plt.plot(desired_tgaps)
# plt.grid()

plt.figure()
desired_vs = idm_param[0].numpy().flatten()
desired_tgaps = idm_param[1].numpy().flatten()
plt.scatter(desired_vs, desired_tgaps, color='grey', s=3)

# plt.scatter(19.4, 2, color='red')
# plt.xlim(15, 25)
# plt.ylim(1, 3)

# plt.scatter(25, 1.4, color='red')
# plt.xlim(20, 30)
# plt.ylim(0, 3)
#
plt.scatter(30, 1, color='red')
plt.xlim(25, 35)
plt.ylim(0, 2)

plt.title(str(sample_index[0]) + ' -- Param')
plt.grid()

##########
plt.figure()
plt.plot(m_y[:20], color='black')
plt.plot(range(20, 40), m_y[20:], color='red')
plt.plot([0, 40], [-1, -1])
plt.title(str(sample_index[0]) + ' -- m_y')
plt.grid()
############
