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

    drivers = ['normal', 'timid', 'aggressive']
    # drivers = ['normal']
    training_size = 3000
    veh_training_size = int(training_size/len(drivers))

    for driver in drivers:
        desired_v, desired_tgap, min_jamx, max_acc, max_decc = get_idm_params(driver)

        for _ in range(veh_training_size):
            # follower_x = np.random.choice(range(40, 80))
            # follower_v = np.random.choice(range(15, 30))
            # lead_v = np.random.choice(range(15, 30))
            lead_x = 100
            follower_x = 40
            # follower_x = np.random.choice(range(30, 95))
            follower_v = 20
            lead_v = 20

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

                lead_v = lead_v + 1.5*np.sin(lead_x*0.01) * 0.1
                lead_x = lead_x + lead_v * 0.1
                # ys.append([acc])
                ys.append([sorted([-3, acc, 3])[1]])
                # if acc < -5:
                #     print('shit', acc)
                #     print('xs: ', xs[-1])

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs).tolist()

    if model_type == 'dnn':
        return [xs_scaled, ys], scaler

    if model_type == 'lstm':
        xs_h = [] # history, scaled
        ys_c = [] # current, not scaled
        xs_h_seq = []

        for i in range(len(xs)):
            xs_h_seq.append(xs_scaled[i])
            if len(xs_h_seq) % episode_len/2 == 0:
                xs_h.append(xs_h_seq)
                ys_c.append(ys[i])
                xs_h_seq = []

        return [xs_h, ys_c], scaler

    if model_type == 'lstm_idm':
        xs_h = [] # history, scaled
        ys_c = [] # current, not scaled
        xs_c = [] # current, not scaled
        xs_h_seq = []

        for i in range(len(xs)):
            xs_h_seq.append(xs_scaled[i])
            if len(xs_h_seq) % episode_len/2 == 0:
                xs_h.append(xs_h_seq)
                ys_c.append(ys[i])
                xs_c.append(xs[i])
                xs_h_seq = []

        return [xs_h, xs_c, ys_c], scaler

    if model_type == 'lstm_seq_idm':
        xs_h = [] # history, scaled
        xs_f = [] # future, not scaled
        ys_f = [] # future, not scaled

        xs_h_seq = []
        xs_f_seq = []
        ys_f_seq = []
        for i in range(len(xs)+1):
            if len(xs_h_seq) == len(ys_f_seq) == episode_len/2:
                xs_h.append(xs_h_seq)
                xs_f.append(xs_f_seq)
                ys_f.append(ys_f_seq)
                xs_h_seq = []
                xs_f_seq = []
                ys_f_seq = []

            if i < len(xs):
                if len(xs_h_seq) < episode_len/2:
                    # history
                    xs_h_seq.append(xs_scaled[i])

                elif len(ys_f_seq) < episode_len/2:
                    # future
                    xs_f_seq.append(xs[i])
                    ys_f_seq.append(ys[i])

        return [xs_h, xs_f, ys_f], scaler



# training_data, scaler = data_generator(model_type='dnn')
# training_data, scaler = data_generator(model_type='lstm_idm')
training_data, scaler = data_generator(model_type='lstm_seq_idm')
len(training_data[0])
# plt.hist(np.array(ys_f)[:,:,0], bins=150)
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
# %%
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
                'max_acc':1.4, # m/s^2
                'max_decc':2, # m/s^2
                }

model_trainer.model.model_use = 'inference'
# model.model_use = 'training'
# from exploratory.models import idm_neural
# reload(idm_neural)
# from exploratory.models.idm_neural import  Encoder
# model = Encoder(config, model_use='training')
xs_h, xs_f, ys_f = training_data
xs_f[0][0]
# %%
start = 2450
end = 2500
for i in range(start, end):
    x = np.array(xs_h[i])
    x_scaled = np.array(xs_h[i])

    x_scaled.shape = (1, 30, 4)
    x.shape = (1, 30, 4)
    param = model_trainer.model([x_scaled, x]).numpy()[0]

    print('pred: ', param)

# %%
x = np.linspace(-5, 20, 100)

y = np.sin(0.05*x)
plt.plot(x, y)
plt.grid()
# %%
model_name = 'lstm_01'
model_type = model_name[:-3]
