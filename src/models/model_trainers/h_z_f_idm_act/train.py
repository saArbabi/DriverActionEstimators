from importlib import reload
import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(suppress=True)
import os

# %%
"""
Data prep
"""
history_len = 20 # steps
rollout_len = 20
import data_generator
reload(data_generator)
from data_generator import DataGenMerge
data_gen = DataGenMerge()
with open('./models/experiments/sim_data_024.pickle', 'rb') as handle:
    features = pickle.load(handle)
features, dummy_value_set = data_gen.fill_missing_values(features)
features_scaled, env_scaler, m_scaler = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, history_len, rollout_len)
history_future_seqs_scaled = data_gen.sequence(features_scaled, history_len, rollout_len)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
future_e_veh_a[:, :, -1].std()
future_e_veh_a[:, :, -1].mean()
# %%
# """
# BALANCE DATA
# """
# history_future_usc, history_sca, future_sca, future_idm_s, future_m_veh_c, future_e_veh_a = data_arrays
# balance_value = np.count_nonzero((history_future_usc[:, :, -14] == 1).all(axis=1))/\
#                         history_future_usc.shape[0]
# print(balance_value)
# cond = (history_future_usc[:, :, -14] == 1).all(axis=1)
# data_arrays = [np.append(data_array, data_array[cond], axis=0) for data_array in data_arrays]
# %%

data_id = '_024'
file_name = 'env_scaler'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(env_scaler, handle)
else:
    print('This data id exists')

file_name = 'm_scaler'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(m_scaler, handle)
else:
    print('This data id exists')

file_name = 'dummy_value_set'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(dummy_value_set, handle)
else:
    print('This data id exists')

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
    def __init__(self, model_type, model_name, experiment_name):
        self.model = None
        self.model_type = model_type
        self.model_name = model_name
        self.experiment_name = experiment_name
        print('This is experiment ' + self.experiment_name)

        self.train_mseloss = []
        self.train_klloss = []

        self.test_mseloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self):
        if self.model_name == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import  NeurIDMModel
            self.model = NeurIDMModel(config)

        with open('./models/experiments/env_scaler_024.pickle', 'rb') as handle:
            self.model.forward_sim.env_scaler = pickle.load(handle)

        with open('./models/experiments/m_scaler_024.pickle', 'rb') as handle:
            self.model.forward_sim.m_scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set_024.pickle', 'rb') as handle:
            self.model.forward_sim.dummy_value_set = pickle.load(handle)

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
            train_input = [history_sca[train_indxs, :, 2:],
                        future_sca[train_indxs, :, 2:],
                        future_idm_s[train_indxs, :, 2:],
                        future_m_veh_c[train_indxs, :, 2:],
                        future_e_veh_a[train_indxs, :, 2:]]

            val_input = [history_sca[val_indxs, :, 2:],
                        future_sca[val_indxs, :, 2:],
                        future_idm_s[val_indxs, :, 2:],
                        future_m_veh_c[val_indxs, :, 2:],
                        future_e_veh_a[val_indxs, :, 2:]]

        return train_input, val_input

    def train(self, train_input, val_input, epochs):
        for epoch in range(epochs):
            self.epoch_count += 1
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'cvae':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))

            print(self.epoch_count, 'epochs completed')
            # print('vae_loss_weight', self.model.vae_loss_weight)
            # self.model.vae_loss_weight += 0.03

    def save_model(self, experiment_name, exp_id):
        experiment_name += exp_id + '_epo_'+str(self.epoch_count)
        exp_dir = './models/experiments/'+experiment_name
        self.experiment_name = experiment_name
        print('This is experiment ' + self.experiment_name)

        if not os.path.exists('./models/experiments/'+experiment_name):
            self.model.save_weights(exp_dir+'/model')

        else:
            print('This model is already saved')

tf.random.set_seed(2021)
experiment_name = 'h_z_f_idm_act095_epo_25'
model_trainer = Trainer(model_type='cvae',
        model_name='driver_model', experiment_name=experiment_name)
train_input, val_input = model_trainer.prep_data(data_arrays)
exp_dir = './models/experiments/'+experiment_name+'/model'
model_trainer.model.load_weights(exp_dir).expect_partial()


# %%
model_trainer.save_model('h_z_f_idm_act', '095')

# %%
