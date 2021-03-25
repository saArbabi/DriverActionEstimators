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


###
# %%

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

# %%
"""
Synthetic data specs
"""
def get_desired_gap(follower_v, dv):
    gap = min_jamx + desired_tgap*follower_v+(follower_v*dv)/ \
                                    (2*np.sqrt(max_act*min_act))
    return gap

def act(follower_v, obs):
    desired_gap = get_desired_gap(follower_v, dv)
    acc = max_act*(1-(follower_v/desired_v)**4-\
                                        (desired_gap/dx)**2)
    return sorted([-3, acc, 3])[1]

def data_generator():
    xs = []
    ys = []
    episode_steps_n = 500
    drivers = ['normal', 'timid', 'aggressive']
    # drivers = ['normal']
    # drivers = ['aggressive']
    training_size = 30000
    episode_n = int(training_size/(episode_steps_n*len(drivers)))
    episode_id = 0

    for driver in drivers:
        desired_v, desired_tgap, min_jamx, max_act, min_act = get_idm_params(driver)

        while episode_id < episode_n:
            follower_x = 40
            lead_x = 100
            follower_v = 20 + np.random.choice(range(-5, 5))
            lead_v = 20 + np.random.choice(range(-5, 5))

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

                acc_magnitude = np.random.normal(1.5, 0.5)
                lead_v = lead_v + 1.5*np.sin(lead_x*0.01) * 0.1
                lead_x = lead_x + lead_v * 0.1
                xs.append([episode_id, time_step, follower_v, lead_v, dv, dx])
                ys.append([episode_id, time_step, acc])

            episode_id += 1
                # ys.append([sorted([-3, acc, 3])[1]])
                # if acc < -5:
                #     print('shit', acc)
                #     print('xs: ', xs[-1])
    xs = np.array(xs)
    scaler = preprocessing.StandardScaler().fit(xs[:, 2:])
    xs_scaled = xs.copy()
    xs_scaled[:, 2:] = scaler.transform(xs[:, 2:]).tolist()

    return xs, xs_scaled, np.array(ys), scaler

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
    xs, xs_scaled, ys, scaler = data_generator()
    episode_ids = list(np.unique(xs[:, 0]))
    sequence_xs_h = []
    sequence_xs_f = []
    sequence_ys_f = []
    for episode_id in episode_ids:
        xs_id = xs[xs[:,0]==episode_id][:,2:].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id][:,2:].tolist()
        ys_id = ys[ys[:,0]==episode_id][:,2:].tolist()

        xs_h, xs_f, ys_f = seqseq_sequence([xs_scaled_id, xs_id, ys_id], h_len, f_len)
        sequence_xs_h.extend(xs_h)
        sequence_xs_f.extend(xs_f)
        sequence_ys_f.extend(ys_f)

    return [sequence_xs_h, sequence_xs_f, sequence_ys_f], scaler


# training_data, scaler = seq_prep(h_len=30)
# training_data, scaler = seq_prep(h_len=30)
training_data, scaler = seqseq_prep(h_len=100, f_len=100)
np.array(training_data[0]).shape

# %%
np.array(training_data[0]).shape
ys = np.array(training_data[2])
np.array(training_data[2])
plt.hist(ys.flatten(), bins=150)
# %%

plt.plot(np.array(training_data[0])[-100, :, -1])
plt.plot(range(29, 59), np.array(training_data[1])[-100, :, -1])
plt.grid()
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
                input = [xs_h[0:train_indx], xs_f[0:train_indx], ys_f[0:train_indx]]
                self.model.train_loop(input)
                input = [xs_h[train_indx:], xs_f[train_indx:], ys_f[train_indx:]]
                self.model.test_loop(input, epoch)
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
                print(self.epoch_count, 'epochs completed')
                self.epoch_count += 1


# model_trainer = Trainer(model_type='dnn')
# model_trainer = Trainer(model_type='lstm')
# model_trainer = Trainer(model_type='lstm_idm')
model_trainer = Trainer(model_type='lstm_seq_idm')
#4 %%
model_trainer.train(training_data, epochs=5)
plt.plot(model_trainer.valid_loss)
plt.plot(model_trainer.train_loss)

plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')

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



exp_dir = './models/experiments/lstm_seq_idm_01/model_dir'


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
model_trainer.model.model_use = 'debug'
# model.model_use = 'training'
# from exploratory.models import idm_neural
# reload(idm_neural)
# from exploratory.models.idm_neural import  Encoder
# model = Encoder(config, model_use='training')
xs_h, xs_f, ys_f = training_data
xs_f[0][0]
choices = range(7000)
# %%
pos = 5
plt.plot(range(29,59), np.array(xs_f[i])[:, pos])
plt.plot(np.array(xs_h[i])[:, pos])
# %%
for _ in range(20):
    plt.figure()
    i = np.random.choice(choices)
    x = np.array(xs_f[i])
    x_scaled = np.array(xs_h[i])
    x_true = np.array(xs_h[i])
    x_scaled.shape = (1, 30, 5)
    x.shape = (1, 30, 5)
    actions, param = model_trainer.model([x_scaled, x])
    actions, param = actions.numpy()[0], param.numpy()[0]
    # print('true: ', ys_f[i])
    plt.title(str(param)+str(i))

    plt.plot(range(29, 59), actions, color='grey')
    plt.plot(range(29, 59), ys_f[i], color='red')
    plt.plot(x_true[:, -1], color='purple')
    plt.grid()
    plt.legend(['pred', 'true'])
# %%
desired_v = 25.
desired_tgap =  1.5
# min_jamx = self.neu_min_jamx(h_t)
min_jamx =  2.
max_act = 1.4
min_act = 2.
# tf.print(min_jamx)
xs_f[5445][0]


vel =  22.2
dv = -4
dx =  73.7

mult_1 = tf.multiply(max_act, min_act)
mult_2 = tf.multiply(2., tf.sqrt(mult_1))
mult_3 = tf.multiply(vel, dv)
div_1 = tf.divide(mult_3, mult_2)
mult_4 = tf.multiply(desired_tgap, vel)

desired_gap = tf.add_n([min_jamx, mult_4, div_1])
pow_1 = tf.pow(tf.divide(desired_gap, dx), 2.)
pow_2 = tf.pow(tf.divide(vel, desired_v), 4.)
subtract_1 = tf.add(pow_2, pow_1)
subtract_2 = tf.subtract(1., subtract_1)

tf.multiply(max_act, subtract_2)

# %%
x = np.linspace(-5, 20, 100)

y = np.sin(0.05*x)
plt.plot(x, y)
plt.grid()
# %%
model_name = 'lstm_01'
model_type = model_name[:-3]
a = tf.constant([1.,2,3])
tf.multiply(2., a)
