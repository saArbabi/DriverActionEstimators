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
"""
Load data
Following protocol from paper
"Simultaneous Policy Learning and Latent State Inference
for Imitating Driver Behavior"
for setting history/rollout length
"""
history_len = 30 # steps
rollout_len = 30
data_id = '031'
dataset_name = 'sim_data_'+data_id
data_arr_name = 'train_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    train_input = pickle.load(handle)
data_arr_name = 'val_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)
train_input[0].shape
data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'
with open(data_files_dir+data_arr_name+'.pickle', 'rb') as handle:
    val_input = pickle.load(handle)
# %%
config = {
 "model_config": {
    "dataset_name": dataset_name,
    "learning_rate": 1e-3,
    "batch_size": 512,
    },
    "Note": ""
}

class Trainer():
    def __init__(self, model_type):
        self.model_type = model_type
        self.train_llloss = []
        self.train_klloss = []

        self.test_llloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self):
        if self.model_type == 'mlp':
            from models.core import mlp
            reload(mlp)
            from models.core.mlp import MLP
            self.model = MLP(config)

        if self.model_type == 'lstm':
            from models.core import lstm
            reload(lstm)
            from models.core.lstm import LSTMEncoder
            self.model = LSTMEncoder(config)

    def load_pre_trained(self, epoch_count):
        exp_dir = self.exp_dir+'/model_epo'+epoch_count
        self.epoch_count = int(epoch_count)
        self.model.load_weights(exp_dir).expect_partial()

        with open(os.path.dirname(exp_dir)+'/'+'losses.pickle', 'rb') as handle:
            losses = pickle.load(handle)

        self.train_llloss = losses['train_llloss']
        self.test_llloss = losses['test_llloss']

    def update_config(self):
        config['train_info'] = {}
        config['train_info']['epoch_count'] = self.epoch_count

        with open(self.exp_dir+'/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def train(self, train_input, val_input, epochs):
        for epoch in range(epochs):
            self.epoch_count += 1
            self.model.train_loop(train_input)
            self.model.test_loop(val_input)
            self.train_llloss.append(round(self.model.train_llloss.result().numpy().item(), 2))
            self.test_llloss.append(round(self.model.test_llloss.result().numpy().item(), 2))

            print(self.epoch_count, 'epochs completed')

    def save_model(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.update_config()
        check_point_dir = self.exp_dir+'/model_epo{epoch}'.format(\
                                                    epoch=self.epoch_count)
        if not os.path.exists(check_point_dir+'.index'):
            self.model.save_weights(check_point_dir)
        else:
            print('This checkpoint is already saved')


    def save_loss(self):
        losses = {'train_llloss':self.train_llloss,
                  'test_llloss':self.test_llloss}

        with open(self.exp_dir+'/losses.pickle', 'wb') as handle:
            pickle.dump(losses, handle)

tf.random.set_seed(2021)
# model_type = 'mlp'
model_type = 'lstm'
model_trainer = Trainer(model_type)
exp_id = '01'
model_name = model_type+'_'+exp_id
model_trainer.exp_dir = './src/models/experiments/'+model_name
# model_trainer.train(train_input, val_input, epochs=1)
# model_trainer.load_pre_trained(epoch_count='20')
# model_trainer.train(train_input, val_input, epochs=1)
# %%

if model_trainer.model_type == 'mlp':
    _train_input = [train_input[0][:,0,:], train_input[-1][:,0,:]]
    _val_input = [val_input[0][:,0,:], val_input[-1][:,0,:]]

if model_trainer.model_type == 'lstm':
    _train_input = [train_input[0], train_input[-1][:,0,:]]
    _val_input = [val_input[0], val_input[-1][:,0,:]]

# %%
# model_trainer.model.forward_sim.attention_temp

################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(_train_input, _val_input, epochs=5)
################## ##### ##################
################## ##### ##################
################## ##### ##################

################## ll LOSS ###############
fig = plt.figure(figsize=(15, 5))
# plt.style.use('default')

ll_axis = fig.add_subplot(111)
kl_axis = fig.add_subplot(111)
ll_axis.plot(model_trainer.test_llloss)
ll_axis.plot(model_trainer.train_llloss)
ll_axis.grid()
ll_axis.set_xlabel('epochs')
ll_axis.set_ylabel('loss (ll)')
ll_axis.set_title('ll')
ll_axis.legend(['test', 'train'])

print(model_trainer.test_llloss[-1])

# %%
model_trainer.save_model()
model_trainer.save_loss()