import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from model import Encoder
from importlib import reload
import tensorflow as tf
import time
from sklearn import preprocessing
import pickle
from collections import deque

# %%

normal_idm = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_act':1.4, # m/s^2
                'min_act':2, # m/s^2
                }

timid_idm = {
                'desired_v':19.4, # m/s
                'desired_tgap':2, # s
                'min_jamx':4, # m
                'max_act':0.8, # m/s^2
                'min_act':1, # m/s^2
                }

aggressive_idm = {
                'desired_v':30, # m/s
                'desired_tgap':1, # s
                'min_jamx':0, # m
                'max_act':2, # m/s^2
                'min_act':3, # m/s^2
                }

config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 256,
    },
    "exp_id": "NA",
    "Note": ""
}
# %%
# %%
"""
Synthetic data generation
"""
def get_idm_params(driver_type):
    if driver_type == 'normal':
        idm_param = normal_idm
    if driver_type == 'timid':
        idm_param = timid_idm
    if driver_type == 'aggressive':
        idm_param = aggressive_idm

    desired_v = idm_param['desired_v']
    desired_tgap = idm_param['desired_tgap']
    min_jamx = idm_param['min_jamx']
    max_act = idm_param['max_act']
    min_act = idm_param['min_act']

    return desired_v, desired_tgap, min_jamx, max_act, min_act

def data_generator():
    xs = []
    ys = []
    info = {}
    episode_steps_n = 500
    drivers = ['normal', 'timid', 'aggressive']
    # drivers = ['normal']
    # drivers = ['aggressive']
    episode_id = 0
    episode_n = 50

    while episode_id < episode_n:
        for driver in drivers:
            desired_v, desired_tgap, min_jamx, max_act, min_act = get_idm_params(driver)

            follower_x = np.random.choice(range(30, 50))
            lead_x = 100
            follower_v = 20 + np.random.choice(range(-3, 3))
            lead_v = 20 + np.random.choice(range(-3, 3))
            lead_acc_mag = np.random.normal(1.5, 0.5)

            for time_step in range(episode_steps_n):
                dv = follower_v-lead_v
                dx = lead_x-follower_x

                desired_gap = min_jamx + desired_tgap*follower_v+(follower_v*dv)/ \
                                                (2*np.sqrt(max_act*min_act))

                acc = max_act*(1-(follower_v/desired_v)**4-\
                                                    (desired_gap/dx)**2)

                follower_v = follower_v + acc * 0.1
                follower_x = follower_x + follower_v * 0.1 \
                                            + 0.5 * acc * 0.1 **2

                lead_v = lead_v + lead_acc_mag*np.sin(lead_x*0.04) * 0.1
                lead_x = lead_x + lead_v * 0.1
                xs.append([episode_id, follower_v, lead_v, dv, dx, acc])
                ys.append([episode_id, acc])
                info[episode_id] = driver

            episode_id += 1
    xs = np.array(xs)
    # scaler = preprocessing.StandardScaler().fit(xs[:, 2:])
    xs_scaled = xs.copy()
    # xs_scaled[:, 2:] = scaler.transform(xs[:, 2:]).tolist()

    return xs, xs_scaled, np.array(ys), info, scaler

def seqseq_sequence(training_states, h_len, f_len):
    states_h, states_f, actions = training_states
    xs_h = [] # history, scaled
    xs_f = [] # future, not scaled
    ys_f = [] # future, not scaled
    episode_steps_n = len(states_h)
    xs_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        xs_h_seq.append(states_h[i])
        if len(xs_h_seq) == h_len:
            indx = i + f_len
            if indx > episode_steps_n:
                break

            xs_h.append(list(xs_h_seq))
            # xs_h.append(np.array(xs_h_seq))
            xs_f.append(states_f[i:indx])
            ys_f.append(actions[i:indx])

    return xs_h, xs_f, ys_f

def seq_sequence(training_states, h_len):
    states_h, states_c, actions = training_states
    xs_h = []
    xs_c = []
    ys_c = []
    episode_steps_n = len(states_h)
    xs_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        xs_h_seq.append(states_h[i])
        if len(xs_h_seq) == h_len:
            xs_h.append(list(xs_h_seq))
            # xs_h.append(np.array(xs_h_seq))
            xs_c.append(states_c[i])
            ys_c.append(actions[i])

    return xs_h, xs_c, ys_c

def dnn_prep():
    _, xs_scaled, ys, scaler = data_generator()
    xs_scaled_id = xs_scaled[:,2:].tolist()
    ys_id = ys[:,2:].tolist()

    return xs_scaled_id, ys_id

def seq_prep(h_len):
    xs, xs_scaled, ys, scaler = data_generator()
    episode_ids = list(np.unique(xs[:, 0]))
    sequence_xs_h = []
    sequence_xs_c = []
    sequence_ys_f = []
    for episode_id in episode_ids:
        xs_id = xs[xs[:,0]==episode_id][:,2:].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id][:,2:].tolist()
        ys_id = ys[ys[:,0]==episode_id][:,2:].tolist()

        xs_h, xs_c, ys_c = seq_sequence([xs_scaled_id, xs_id, ys_id], h_len)
        sequence_xs_h.extend(xs_h)
        sequence_xs_c.extend(xs_c)
        sequence_ys_f.extend(ys_c)

    return [sequence_xs_h, sequence_xs_c, sequence_ys_f], scaler

def seqseq_prep(h_len, f_len):
    xs, xs_scaled, ys, info, scaler = data_generator()
    episode_ids = list(np.unique(xs[:, 0]))
    sequence_xs_h = []
    sequence_xs_f = []
    sequence_ys_f = []

    for episode_id in episode_ids:
        xs_id = xs[xs[:,0]==episode_id].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id].tolist()
        ys_id = ys[ys[:,0]==episode_id].tolist()
        xs_h, xs_f, ys_f = seqseq_sequence([xs_scaled_id, xs_id, ys_id], h_len, f_len)
        sequence_xs_h.extend(xs_h)
        sequence_xs_f.extend(xs_f)
        sequence_ys_f.extend(ys_f)

    return [np.array(sequence_xs_h), np.array(sequence_xs_f), np.array(sequence_ys_f)], info, scaler


# training_data, scaler = seq_prep(h_len=30)
# training_data, scaler = seq_prep(h_len=30)
training_data, info, scaler = seqseq_prep(h_len=100, f_len=100)



# %%
class Trainer():
    def __init__(self, model_type):
        self.model = None
        self.model_type = model_type
        self.train_loss = []
        self.valid_loss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self, model_type=None):
        if self.model_type == 'dnn':
            from exploratory.models import dnn
            reload(dnn)
            from exploratory.models.dnn import  Encoder
            self.model = Encoder(config)

        if self.model_type == 'lstm':
            from exploratory.models import lstm
            reload(lstm)
            from exploratory.models.lstm import  Encoder
            self.model = Encoder(config)

        elif self.model_type == 'lstm_idm':
            from exploratory.models import lstm_idm
            reload(lstm_idm)
            from exploratory.models.lstm_idm import  Encoder
            self.model = Encoder(config, model_use='training')

        elif self.model_type == 'lstm_seq_idm':
            from exploratory.models import lstm_seq_idm
            reload(lstm_seq_idm)
            from exploratory.models.lstm_seq_idm import  Encoder
            self.model = Encoder(config, model_use='training')

    def train(self, training_data, epochs):

        if self.model_type == 'dnn' or self.model_type == 'lstm':
            xs, ys = training_data
            train_indx = int(len(xs)*0.8)

            for epoch in range(epochs):
                self.model.train_loop([xs[0:train_indx], ys[0:train_indx]])
                self.model.test_loop([xs[train_indx:], ys[train_indx:]], epoch)
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
                print(epoch, 'epochs completed')
                self.epoch_count += 1

        elif self.model_type == 'lstm_idm':
            xs_h, xs_c, ys_c = training_data
            train_indx = int(len(xs_h)*0.8)
            for epoch in range(epochs):
                input = [xs_h[0:train_indx], xs_c[0:train_indx], ys_c[0:train_indx]]
                self.model.train_loop(input)
                input = [xs_h[train_indx:], xs_c[train_indx:], ys_c[train_indx:]]
                self.model.test_loop(input, epoch)
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
                print(self.epoch_count, 'epochs completed')
                self.epoch_count += 1

        elif self.model_type == 'lstm_seq_idm':
            xs_h, xs_f, ys_f = training_data
            train_indx = int(len(xs_h)*0.8)
            for epoch in range(epochs):
                input = [xs_h[0:train_indx, :, 1:], xs_f[0:train_indx, :, 1:], \
                                                ys_f[0:train_indx, :, 1:]]

                self.model.train_loop(input)
                input = [xs_h[train_indx:, :, 1:], xs_f[train_indx:, :, 1:], \
                                                ys_f[train_indx:, :, 1:]]

                self.model.test_loop(input, epoch)
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
                print(self.epoch_count, 'epochs completed')
                self.epoch_count += 1

# model_trainer = Trainer(model_type='dnn')
# model_trainer = Trainer(model_type='lstm')
# model_trainer = Trainer(model_type='lstm_idm')
model_trainer = Trainer(model_type='lstm_seq_idm')
training_data[0][:,:,-1].min()
# %%
model_trainer.train(training_data, epochs=5)
plt.plot(model_trainer.valid_loss)
plt.plot(model_trainer.train_loss)

plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
model_trainer.valid_loss

# %%

# %%
import pickle
import os
os.getcwd()
# %%

exp_dir = './models/experiments/dnn_01/model_dir'
exp_dir = './models/experiments/dnn_02/model_dir'
exp_dir = './models/experiments/dnn_03/model_dir'
exp_dir = './models/experiments/lstm_01/model_dir'
exp_dir = './models/experiments/lstm_idm_01/model_dir'
exp_dir = './models/experiments/lstm_idm_03/model_dir'
exp_dir = './models/experiments/lstm_seq_idm_03/model_dir'
exp_dir = './models/experiments/lstm_seq6s_idm_03/model_dir'
exp_dir = './models/experiments/try/model_dir'



exp_dir = './models/experiments/lstm_seq_idm/model'


# %%
model_trainer.model.save_weights(exp_dir)
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)

# %%

# %%
normal_idm = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_act':1.4, # m/s^2
                'min_act':2, # m/s^2
                }

model_trainer.model.model_use = 'inference'

# %%
model_trainer.model.model_use = 'debug'
xs_h, xs_f, ys_f = training_data
train_indx = int(len(xs_h)*0.8)
episode_ids = np.unique(xs_h[train_indx:, 0, 0])

xs_h = xs_h[train_indx:, :, :]
xs_f = xs_f[train_indx:, :, :]
ys_f = ys_f[train_indx:, :, :]

for _ in range(20):
    plt.figure()
    episode_id = np.random.choice(episode_ids)
    driver_type = info[episode_id]
    xs_h_epis = xs_h[xs_h[:, 0, 0] == episode_id]
    xs_f_epis = xs_f[xs_f[:, 0, 0] == episode_id]
    ys_f_epis = ys_f[ys_f[:, 0, 0] == episode_id]
    i_choices = range(len(xs_h_epis))
    i = np.random.choice(i_choices)
    xs_h_i = xs_h_epis[i:i+1, :, 1:]
    xs_f_i = xs_f_epis[i:i+1, :, 1:]
    ys_f_i = ys_f_epis[i:i+1, :, 1:]

    actions, param = model_trainer.model([xs_h_i, xs_f_i])
    actions, param = actions.numpy()[0], param.numpy()[0]
    # print('true: ', ys_f[i])
    plt.title(str(param)+' '+driver_type)
    plt.plot(range(99, 199), actions, color='grey')
    plt.plot(range(99, 199), ys_f_i[0, :, -1], color='red')
    plt.plot(xs_h_i[0, :, -1], color='purple')
    plt.ylim(-3, 3)
    plt.grid()
    plt.legend(['pred', 'true'])
# %%
