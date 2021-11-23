from importlib import reload
import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(suppress=True)
from importlib import reload
import pickle
import os
import sys
import json

sys.path.insert(0, './src')
# %%

# %%
"""
Load data
"""
history_len = 30 # steps
rollout_len = 30
data_id = '_026'
dataset_name = 'sim_data'+data_id
data_arr_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len, rollout_len)

data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    data_arrays = pickle.load(handle)
len(data_arrays)
# %%
config = {
 "model_config": {
    "dataset_name": dataset_name,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "vae_loss_weight": 0.1,
    "attention_temp": 5,
    "latent_dim": 3,
    },
    "Note": "To check performance for different attention_temps"
}

class Trainer():
    def __init__(self, exp_id):
        self.model_name = 'h_z_f_idm_act_'+exp_id
        self.exp_dir = './src/models/experiments/'+self.model_name
        self.train_mseloss = []
        self.train_klloss = []

        self.test_mseloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        from models.core import neural_idm
        reload(neural_idm)
        from models.core.neural_idm import  NeurIDMModel
        self.model = NeurIDMModel(config)

        with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
            self.model.forward_sim.env_scaler = pickle.load(handle)

        with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
            self.model.forward_sim.dummy_value_set = pickle.load(handle)

    def load_pre_trained(self, epoch_count):
        exp_dir = self.exp_dir+'/model_epo'+epoch_count
        self.epoch_count = int(epoch_count)
        self.model.load_weights(exp_dir).expect_partial()

        with open(os.path.dirname(exp_dir)+'/'+'losses.pickle', 'rb') as handle:
            losses = pickle.load(handle)

        self.train_mseloss = losses['train_mseloss']
        self.train_klloss = losses['train_klloss']
        self.test_mseloss = losses['test_mseloss']
        self.test_klloss = losses['test_klloss']

    def update_config(self):
        config['train_info'] = {}
        config['train_info']['epoch_count'] = self.epoch_count

        with open(self.exp_dir+'/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def prep_data(self, training_data):
        all_epis = np.unique(training_data[0][:, 0, 0])
        np.random.seed(2021)
        np.random.shuffle(all_epis)

        train_epis = all_epis[:int(len(all_epis)*0.7)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_examples = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

        _, history_sca, future_sca, future_idm_s,\
                    future_m_veh_c, future_e_veh_a = training_data
        train_input = [history_sca[train_examples, :, 2:],
                    future_sca[train_examples, :, 2:],
                    future_idm_s[train_examples, :, 2:],
                    future_m_veh_c[train_examples, :, 2:],
                    future_e_veh_a[train_examples, :, 2:]]

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
            self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
            self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
            self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
            self.test_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))

            print(self.epoch_count, 'epochs completed')

    def save_model(self):
        self.update_config()
        check_point_dir = self.exp_dir+'/model_epo{epoch}'.format(\
                                                    epoch=self.epoch_count)
        if not os.path.exists(check_point_dir+'.index'):
            self.model.save_weights(check_point_dir)
        else:
            print('This checkpoint is already saved')


    def save_loss(self):
        losses = {'train_mseloss':self.train_mseloss,
                  'train_klloss':self.train_klloss,
                  'test_mseloss':self.test_mseloss,
                  'test_klloss':self.test_klloss}

        with open(self.exp_dir+'/losses.pickle', 'wb') as handle:
            pickle.dump(losses, handle)

tf.random.set_seed(2021)
exp_id = '098'
model_trainer = Trainer(exp_id)
train_input, val_input = model_trainer.prep_data(data_arrays)
# model_trainer.train(train_input, val_input, epochs=1)
# model_trainer.load_pre_trained(epoch_count='20')
# %%
# model_trainer.model.forward_sim.attention_temp

################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(train_input, val_input, epochs=5)
################## ##### ##################
################## ##### ##################
################## ##### ##################

################## MSE LOSS ###############
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

# %%
model_trainer.save_model()
model_trainer.save_loss()
