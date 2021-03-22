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
# %%

normal_idm = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_acc':1.4, # m/s^2
                'max_decc':2, # m/s^2
                }

timid_idm = {
                'desired_v':19.4, # m/s
                'desired_tgap':2, # s
                'min_jamx':4, # m
                'max_acc':0.8, # m/s^2
                'max_decc':1, # m/s^2
                }

aggressive_idm = {
                'desired_v':30, # m/s
                'desired_tgap':1, # s
                'min_jamx':0, # m
                'max_acc':2, # m/s^2
                'max_decc':3, # m/s^2
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

    # return sorted([-3, acc, 3])[1]

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
    max_acc = idm_param['max_acc']
    max_decc = idm_param['max_decc']

    return desired_v, desired_tgap, min_jamx, max_acc, max_decc

# %%
"""
Synthetic data specs
"""
def get_desired_gap(follower_v, dv):
    gap = min_jamx + desired_tgap*follower_v+(follower_v*dv)/ \
                                    (2*np.sqrt(max_acc*max_decc))
    return gap

def act(follower_v, obs):
    desired_gap = get_desired_gap(follower_v, dv)
    acc = max_acc*(1-(follower_v/desired_v)**4-\
                                        (desired_gap/dx)**2)
    return sorted([-3, acc, 3])[1]

def data_generator(model_type):
    xs = []
    ys = []
    episode_len = 60
    lead_v = 20

    drivers = ['normal', 'timid', 'aggressive']
    data_size = 20000
    training_size = int(data_size/len(drivers))

    for driver in drivers:
        desired_v, desired_tgap, min_jamx, max_acc, max_decc = get_idm_params(driver)

        for sample_i in range(training_size):
            follower_x = np.random.choice(range(40, 80))
            follower_v = np.random.choice(range(15, 25))
            lead_v = np.random.choice(range(15, 30))
            lead_x = 100

            for i in range(episode_len):
                dv = follower_v-lead_v
                dx = lead_x-follower_x
                xs.append([follower_v, lead_v, dv, dx])

                desired_gap = min_jamx + desired_tgap*follower_v+(follower_v*dv)/ \
                                                (2*np.sqrt(max_acc*max_decc))

                acc = max_acc*(1-(follower_v/desired_v)**4-\
                                                    (desired_gap/dx)**2)

                follower_v = follower_v + acc * 0.1
                follower_x = follower_x + follower_v * 0.1 \
                                            + 0.5 * acc * 0.1 **2
                lead_x = lead_x + lead_v * 0.1
                ys.append([acc])

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs).tolist()

    if model_type == 'dnn':
        return xs_scaled, xs, ys, scaler

    if model_type == 'lstm' or model_type == 'lstmidm':
        seq_xs_scaled = []
        seq_xs = []
        seq_ys = []
        obs_history_scaled = []
        obs_history = []

        for i in range(len(xs)):
            obs_history_scaled.append(xs_scaled[i])
            obs_history.append(xs[i])
            if len(obs_history) % episode_len == 0:
                seq_ys.append(ys[i])
                seq_xs_scaled.append(obs_history_scaled)
                seq_xs.append(obs_history)
                obs_history_scaled = []
                obs_history = []

        return seq_xs_scaled, seq_xs, seq_ys, scaler


xs_scaled, xs, ys, scaler = data_generator(model_type='lstm')

# xs_scaled, xs, ys, scaler = data_generator(model_type='dnn')
# max(ys)
xs[0]
xs_scaled[0]
# %%
from exploratory.models import dnn
reload(dnn)
from exploratory.models.dnn import  Encoder

def train_exp(xs_scaled, xs, ys, model_type):
    train_indx = int(len(xs)*0.8)
    train_loss = []
    valid_loss = []
    epoch_n = 30

    if model_type == 'dnn':
        from exploratory.models.dnn import  Encoder
        model = Encoder(config)

        for epoch in range(epoch_n):
            model.train_loop([xs[0:train_indx], ys[0:train_indx]])
            model.test_loop([xs[train_indx:], ys[train_indx:]], epoch)
            train_loss.append(round(model.train_loss.result().numpy().item(), 2))
            valid_loss.append(round(model.test_loss.result().numpy().item(), 2))

            print(epoch, 'epochs completed')

    if model_type == 'lstm':
        from exploratory.models.lstm import  Encoder
        model = Encoder(config)

        for epoch in range(epoch_n):
            model.train_loop([xs[0:train_indx], ys[0:train_indx]])
            model.test_loop([xs[train_indx:], ys[train_indx:]], epoch)
            train_loss.append(round(model.train_loss.result().numpy().item(), 2))
            valid_loss.append(round(model.test_loss.result().numpy().item(), 2))

            print(epoch, 'epochs completed')

    if model_type == 'lstmidm':
        from exploratory.models import idm_neural
        reload(idm_neural)
        from exploratory.models.idm_neural import  Encoder
        model = Encoder(config, model_use='training')

        for epoch in range(epoch_n):
            input = [xs_scaled[0:train_indx], xs[0:train_indx], ys[0:train_indx]]
            model.train_loop(input)
            input = [xs_scaled[train_indx:], xs[train_indx:], ys[train_indx:]]
            model.test_loop(input, epoch)
            train_loss.append(round(model.train_loss.result().numpy().item(), 2))
            valid_loss.append(round(model.test_loss.result().numpy().item(), 2))

            print(epoch, 'epochs completed')

    return model, train_loss, valid_loss

# train_debugger()
# model = Encoder(config, model_use='training')

# model, train_loss, valid_loss = train_exp(model, xs, ys, 'dnn')
# model, train_loss, valid_loss = train_exp(model, xs, ys, 'lstm')
model, train_loss, valid_loss = train_exp(xs_scaled, xs, ys, 'lstmidm')

# %%
plt.plot(valid_loss)
plt.plot(train_loss)

plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
# %%
import pickle
import os
os.getcwd()
# %%

exp_dir = './models/experiments/dnn_01/model_dir'
exp_dir = './models/experiments/dnn_02/model_dir'
exp_dir = './models/experiments/dnn_03/model_dir'
exp_dir = './models/experiments/lstm_01/model_dir'
exp_dir = './models/experiments/lstmidm_01/model_dir'
exp_dir = './models/experiments/lstmidm_03/model_dir'
exp_dir = './models/experiments/lstmidm_03_6s/model_dir'
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)

# %%
model.save_weights(exp_dir)
model.load_weights(exp_dir)

# %%

# %%
normal_idm = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_acc':1.4, # m/s^2
                'max_decc':2, # m/s^2
                }

model.model_use = 'inference'
# model.model_use = 'training'
# from exploratory.models import idm_neural
# reload(idm_neural)
# from exploratory.models.idm_neural import  Encoder
# model = Encoder(config, model_use='training')

for i in range(10):
    input = np.array([[xs_scaled[i]], [xs[i]]])
    out = model(input)
    # print('true: ', ys[i])
    print('pred: ', out)

# %%
x = np.linspace(-5, 20, 100)

y = np.sin(0.05*x)
plt.plot(x, y)
plt.grid()
# %%
model_name = 'lstm_01'
model_type = model_name[:-3]
