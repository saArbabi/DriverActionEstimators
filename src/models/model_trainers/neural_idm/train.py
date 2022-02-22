import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot

import numpy as np
np.set_printoptions(suppress=True)
from importlib import reload
import pickle
import os
import sys
import json
import time
sys.path.insert(0, './src')

# %%
"""
Load data
"""
history_len = 20 # steps
rollout_len = 50
data_id = '047'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'train_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

data_files_dir = './src/datasets/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    train_input = pickle.load(handle)
data_arr_name = 'test_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)
train_input[0].shape
data_files_dir = './src/datasets/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    test_input = pickle.load(handle)
train_input[-1].shape
# %%
train_input[-1][:, :, 0:1].mean()
train_input[-1][:, :, 0:1].std()

# %%
config = {
 "model_config": {
    "dataset_name": dataset_name,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "vae_loss_weight": 0.01,
    "attention_temp": 1,
    "latent_dim": 6,
    },
     "data": {
     "dataset_name": dataset_name,
     "history_len": history_len,
     "rollout_len": rollout_len,
     },
    "Note": ""
    }

class Trainer():
    def __init__(self, exp_id):
        self.epoch_count = 0
        self.losses = {'train_losses':{'displacement_loss':[], \
                               'action_loss':[], 'kl_loss':[], 'tot_loss':[]}, \
                       'test_losses':{'displacement_loss':[], \
                              'action_loss':[], 'kl_loss':[], 'tot_loss':[]}}
        self.initiate_model(exp_id)

    def initiate_model(self, exp_id):
        from models.core import neural_idm
        reload(neural_idm)
        from models.core.neural_idm import  NeurIDMModel
        self.model = NeurIDMModel(config, exp_id)
        self.model.make_event_files()

        self.model.forward_sim.rollout_len = rollout_len

        with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
            self.model.forward_sim.env_scaler = pickle.load(handle)

        with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
            self.model.forward_sim.dummy_value_set = pickle.load(handle)

        self.model.disx_means = train_input[-1][:, :, -1:].mean(axis=0)
        self.model.disx_std = train_input[-1][:, :, -1:].std(axis=0)
        self.model.action_means = train_input[-1][:, :, 0].mean()
        self.model.action_std = train_input[-1][:, :, 0].std()

    def load_pre_trained(self, epoch_count):
        exp_dir = self.exp_dir+'/model_epo'+epoch_count
        self.epoch_count = int(epoch_count)
        self.model.load_weights(exp_dir).expect_partial()
        self.read_losses()

    def update_config(self):
        config['train_info'] = {}
        config['train_info']['epoch_count'] = self.epoch_count

        with open(self.exp_dir+'/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def get_event_paths(self):
        event_items = os.listdir(self.exp_dir+'/logs/train')
        train_event_paths = []
        for event in event_items:
            path = os.path.join(self.exp_dir+'/logs/train', event)
            train_event_paths.append(path)

        event_items = os.listdir(self.exp_dir+'/logs/test')
        test_event_paths = []
        for event in event_items:
            path = os.path.join(self.exp_dir+'/logs/test', event)
            test_event_paths.append(path)
        return train_event_paths, test_event_paths

    def read_event_files(self):
        train_losses = {'displacement_loss':[], 'action_loss':[], \
                                                'kl_loss':[], 'tot_loss':[]}
        test_losses = {'displacement_loss':[], 'action_loss':[], \
                                                'kl_loss':[], 'tot_loss':[]}
        train_event_paths, test_event_paths = self.get_event_paths()

        for train_event_path in train_event_paths:
            for e in tf.compat.v1.train.summary_iterator(train_event_path):
                for value in e.summary.value:
                    if value.tag == 'displacement_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'action_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'kl_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'tot_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())

        for test_event_path in test_event_paths:
            for e in tf.compat.v1.train.summary_iterator(test_event_path):
                for value in e.summary.value:
                    if value.tag == 'displacement_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'action_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'kl_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'tot_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
        return train_losses, test_losses

    def read_losses(self):
        train_losses, test_losses = self.read_event_files()
        self.losses['train_losses']['displacement_loss'] = train_losses['displacement_loss']
        self.losses['test_losses']['displacement_loss'] = test_losses['displacement_loss']
        self.losses['train_losses']['action_loss'] = train_losses['action_loss']
        self.losses['test_losses']['action_loss'] = test_losses['action_loss']
        self.losses['train_losses']['kl_loss'] = train_losses['kl_loss']
        self.losses['test_losses']['kl_loss'] = test_losses['kl_loss']
        self.losses['train_losses']['tot_loss'] = train_losses['tot_loss']
        self.losses['test_losses']['tot_loss'] = test_losses['tot_loss']

    def train(self, train_input, test_input, epochs):
        for epoch in range(epochs):
            t0 = time.time()
            self.epoch_count += 1
            self.model.train_test_loop([train_input, test_input])
            print(self.epoch_count, 'epochs completed')
            print(round((time.time()-t0)), 'secs per epoch')
        self.read_losses()

    def save_model(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.update_config()
        check_point_dir = self.exp_dir+'/model_epo{epoch}'.format(\
                                                    epoch=self.epoch_count)
        if not os.path.exists(check_point_dir+'.index'):
            self.model.save_weights(check_point_dir)
            print('Saved at ', self.exp_dir)
        else:
            print('This checkpoint is already saved')


tf.random.set_seed(2021)
exp_id = '320'
# exp_id = 'test_15'
model_name = 'neural_idm_'+exp_id
model_trainer = Trainer(exp_id)
model_trainer.exp_dir = './src/models/experiments/' + model_name
# model_trainer.load_pre_trained(epoch_count='1')
# model_trainer.model.vae_loss_weight = 0.03
# model_trainer.model.make_event_files()
print(model_trainer.exp_dir)
# %%
# model_trainer.model.vae_loss_weight = 0.3
################## Train ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(train_input, test_input, epochs=2)
################## ##### ##################
################## ##### ########### #######
################## ##### ##################
################## ##### ####### ###########
fig = plt.figure(figsize=(15, 10))
# plt.style.use('default')

displacement_axis = fig.add_subplot(221)
action_axis = fig.add_subplot(223)
kl_axis = fig.add_subplot(222)
tot_axis = fig.add_subplot(224)
train_losses, test_losses = model_trainer.losses['train_losses'], model_trainer.losses['test_losses']
itr_step = np.linspace(0, len(train_losses['displacement_loss']), len(test_losses['displacement_loss']))

np.array(test_losses['action_loss']).mean()/np.array(test_losses['displacement_loss']).mean()
np.array(train_losses['action_loss']).mean()/np.array(train_losses['displacement_loss']).mean()
################## displacement_loss LOSS ####    ###########
displacement_axis.plot(itr_step, test_losses['displacement_loss'], color='blue')
displacement_axis.plot(train_losses['displacement_loss'], color='red')
displacement_axis.grid()
displacement_axis.set_xlabel('iterations')
displacement_axis.set_ylabel('displacement_loss')
displacement_axis.legend(['test', 'train'])
################## action_loss LOSS ####    ###########
action_axis.plot(itr_step, test_losses['action_loss'], color='blue')
action_axis.plot(train_losses['action_loss'], color='red')
action_axis.grid()
action_axis.set_xlabel('iterations')
action_axis.set_ylabel('action_loss')
action_axis.legend(['test', 'train'])
################## kl LOSS ##################
kl_axis.plot(itr_step, test_losses['kl_loss'], color='blue')
kl_axis.plot(train_losses['kl_loss'], color='red')
kl_axis.grid()
kl_axis.set_xlabel('iterations')
kl_axis.set_ylabel('loss (kl)')
kl_axis.legend(['test', 'train'])
################## Total LOSS ##################
tot_axis.plot(itr_step, test_losses['tot_loss'], color='blue')
tot_axis.plot(train_losses['tot_loss'], color='red')
tot_axis.grid()
tot_axis.set_xlabel('iterations')
tot_axis.set_ylabel('tot_loss')
tot_axis.legend(['test', 'train'])
print('train_losses displacement_loss ', train_losses['displacement_loss'][-1])

# %%
model_trainer.save_model()
# %%
x = np.linspace(-5, 5, 100)
min = 1
max = 5
temp = 4/(max-min)

y = min + (max-min)/(1 + np.exp(-temp*x))
# y = np.exp(x)
plt.plot(x, y)
print('grad at x=0: '+str((y[50]-y[49])/(x[50]-x[49])))
# %%
