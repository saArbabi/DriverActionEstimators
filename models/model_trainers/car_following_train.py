# import os
import matplotlib.pyplot as plt
from importlib import reload
import sys
import numpy as np
np.set_printoptions(suppress=True)
import time

# %%



indxs = {}
feature_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id',
         'aggressiveness', 'f_veh_exists',
         'e_veh_glob_x', 'f_veh_glob_x',
         'e_veh_speed', 'f_veh_speed',
         'e_veh_action', 'f_veh_action',
         'el_delta_v', 'el_delta_x']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1
# %%
"""
PREPARE DATA
"""
config = {'lanes_n':1,
        'lane_width':3.75, # m
        'lane_length':400 # m
        }
data_config = {
                'history_seq_length':20,
                'env_steps_n':2000,
                }
import highway
reload(highway)
from highway import EnvLaneKeep
env = EnvLaneKeep(config)

import data_generator
reload(data_generator)
from data_generator import DataGeneratorLaneKeep
data_gen = DataGeneratorLaneKeep(env, data_config)
features_origin  = data_gen.prep_data()
# features_origin[6]
features = features_origin.copy()
features.shape
# features = data_gen.fill_missing_values(features)
features.shape


# %%
"""
Driver model - neural idm
"""

features_scaled, scaler = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, 20, 40)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 40)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_a, future_e_veh_a = data_arrays
future_m_veh_a[future_m_veh_a[:, :, 2] == 1]

# data_arrays = [np.nan_to_num(data_array, 0) for data_array in data_arrays]
future_idm_s[500, 0, :]
future_m_veh_a.shape
future_m_veh_a.shape
history_future_usc[0, 20, :]
history_sca[0, -1, :]
future_e_veh_a[0, 0, :]
# %%
"""
Driver model - lstm
"""
features_scaled, scaler = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, 20, 1)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 1)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_a, future_e_veh_a = data_arrays
future_m_veh_a[future_m_veh_a[:, :, 2] == 1]
future_e_veh_a[0]
history_future_usc[0]
history_sca[0]
# data_arrays = [np.nan_to_num(data_array, 0) for data_array in data_arrays]
future_idm_s[0:10, 0, :]
future_m_veh_a.shape
future_sca.shape
future_m_veh_a.shape
# %%
col_names = ['episode_id', 'time_step',
        'e_veh_speed', 'f_veh_speed',
        'el_delta_v', 'el_delta_x', 'f_veh_exists']

for i in range(future_sca.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_sca[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%

col_names = ['episode_id', 'time_step', 'e_veh_id',
        'e_veh_speed', 'f_veh_speed',
        'e_veh_action', 'f_veh_action',
        'el_delta_v', 'el_delta_x', 'aggressiveness']


# np.count_nonzero(history_future_usc[:, :, 6] == 0)

for i in range(history_future_usc.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_future_usc[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%


# %%

col_names = ['episode_id', 'time_step',
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x']
# type(future_idm_s[i, 0, -3])

for i in range(future_idm_s.shape[-1]):
    plt.figure(figsize=(3, 3))
    _ = plt.hist(future_idm_s[:, -1, i], bins=150)
    plt.title(col_names[i])
    plt.grid()
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
    def __init__(self, training_data, model_type):
        self.model = None
        self.model_type = model_type

        self.train_mseloss = []
        self.train_att_klloss = []
        self.train_idm_klloss = []

        self.test_mseloss = []
        self.test_att_klloss = []
        self.test_idm_klloss = []
        self.epoch_count = 0
        self.initiate_model()
        self.prep_data(training_data)

    def initiate_model(self):
        if self.model_type == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import NeurIDMModelLaneKeep
            self.model = NeurIDMModelLaneKeep(config)

        elif self.model_type == 'lstm_model':
            from models.core import lstm
            reload(lstm)
            from models.core.lstm import  Encoder
            self.model = Encoder(config)

    def prep_data(self, training_data):
        all_epis = np.unique(training_data[0][:, 0, 0])
        np.random.seed(2021)
        np.random.shuffle(all_epis)
        train_epis = all_epis[:int(len(all_epis)*0.8)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_indxs = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

        _, history_sca, future_sca, future_idm_s,\
                    future_m_veh_a, future_e_veh_a = training_data

        if self.model_type == 'driver_model':

            self.train_input = [history_sca[train_indxs, :, 2:],
                        future_sca[train_indxs, :, 2:],
                        future_idm_s[train_indxs, :, 2:],
                        future_m_veh_a[train_indxs, :, 2:],
                        future_e_veh_a[train_indxs, :, 2:]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                        future_sca[val_indxs, :, 2:],
                        future_idm_s[val_indxs, :, 2:],
                        future_m_veh_a[val_indxs, :, 2:],
                        future_e_veh_a[val_indxs, :, 2:]]

        elif self.model_type == 'lstm_model':
            self.train_input = [history_sca[train_indxs, :, 2:],
                                        future_e_veh_a[train_indxs, 0, -1]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                                        future_e_veh_a[val_indxs, 0, -1]]

    def train(self, epochs):
        for epoch in range(epochs):
            t0 = time.time()
            self.model.train_loop(self.train_input)
            self.model.test_loop(self.val_input, epoch)
            if self.model_type == 'vae_idm' or self.model_type == 'driver_model':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_att_klloss.append(round(self.model.train_att_klloss.result().numpy().item(), 2))
                self.train_idm_klloss.append(round(self.model.train_idm_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_att_klloss.append(round(self.model.test_att_klloss.result().numpy().item(), 2))
                self.test_idm_klloss.append(round(self.model.test_idm_klloss.result().numpy().item(), 2))
            else:
                self.train_mseloss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_loss.result().numpy().item(), 2))
            t1 = time.time()
            print(self.epoch_count, 'epochs completed')
            print('Epoch took: ', round(t1-t0), ' seconds')
            self.epoch_count += 1

    def save_model(self, model_name):
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

# model_trainer = Trainer(data_arrays, model_type='driver_model')

model_trainer = Trainer(data_arrays, model_type='lstm_model')
model_trainer.train(epochs=3)
model_trainer.train_mseloss
# %%


fig = plt.figure(figsize=(15, 5))
plt.style.use('default')

mse_axis = fig.add_subplot(131)
att_kl_axis = fig.add_subplot(132)
idm_kl_axis = fig.add_subplot(133)
mse_axis.plot(model_trainer.test_mseloss)
mse_axis.plot(model_trainer.train_mseloss)



# %%
all_epis = np.unique(history_sca[:, 0, 0])
np.random.seed(2021)
np.random.shuffle(all_epis)
train_epis = all_epis[:int(len(all_epis)*0.8)]

val_epis = np.setdiff1d(all_epis, train_epis)
train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
history_sca.shape
train_indxs.shape
val_examples.shape
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_a = np.float32(future_m_veh_a)

# %%
model_trainer.model.vae_loss_weight = 0.1
model_trainer.train(epochs=5)
################## MSE LOSS ##################
fig = plt.figure(figsize=(15, 5))
plt.style.use('default')

mse_axis = fig.add_subplot(131)
att_kl_axis = fig.add_subplot(132)
idm_kl_axis = fig.add_subplot(133)
mse_axis.plot(model_trainer.test_mseloss)
mse_axis.plot(model_trainer.train_mseloss)
#
mse_axis.grid()
mse_axis.set_xlabel('epochs')
mse_axis.set_ylabel('loss (MSE)')
mse_axis.set_title('MSE')
mse_axis.legend(['test', 'train'])

################## att_kl LOSS ##################
att_kl_axis.plot(model_trainer.test_att_klloss)
att_kl_axis.plot(model_trainer.train_att_klloss)

att_kl_axis.grid()
att_kl_axis.set_xlabel('epochs')
att_kl_axis.set_ylabel('loss (att_kl)')
att_kl_axis.set_title('att_kl')
att_kl_axis.legend(['test', 'trai n'])

################## idm_kl LOSS ##################
idm_kl_axis.plot(model_trainer.test_idm_klloss)
idm_kl_axis.plot(model_trainer.train_idm_klloss)
idm_kl_axis.grid()
idm_kl_axis.set_xlabel('epochs')
idm_kl_axis.set_ylabel('loss (idm_kl)')
idm_kl_axis.set_title('idm_kl')
idm_kl_axis.legend(['test', 'train'])
att_axis, idm_axis = latent_vis()

# %%
def latent_samples(model_trainer, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    sdv_actions = future_m_veh_a[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    enc_acts = model_trainer.model.act_encoder(sdv_actions)
    prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
    sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_att_z, sampled_idm_z

def latent_vis():
    fig = plt.figure(figsize=(4, 4))
    # plt.style.use('ggplot')
    plt.style.use('default')
    att_axis = fig.add_subplot(211)
    idm_axis = fig.add_subplot(212)
    sampled_att_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
    aggressiveness = history_future_usc[val_examples, 0, -1]
    color_shade = aggressiveness
    att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, \
                                                c=color_shade, cmap='rainbow')
    idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, \
                                    c=color_shade, cmap='rainbow')

    att_axis.set_ylabel('$z_{att_1}$')
    att_axis.set_xlabel('$z_{att_2}$')
    idm_axis.set_ylabel('$z_{idm_1}$')
    idm_axis.set_xlabel('$z_{idm_1}$')

    return att_axis, idm_axis
att_axis, idm_axis = latent_vis()

# %%
import pickle
# model_trainer.save_model('driver_model_kl')
model_trainer.save_model('lstm_keep_lane')
# with open('./models/experiments/scaler.pickle', 'wb') as handle:
#     pickle.dump(scaler, handle)
