import numpy as np
import matplotlib.pyplot as plt
from model import Encoder
from importlib import reload
import pickle

# %%
"""
Generate training data
"""
from factory import data_generator
reload(data_generator)
from factory.data_generator import *
# training_data, info, scaler = seqseq_prep(h_len=100, f_len=100)
training_samples_n = 5000
# training_data = dnn_prep(training_samples_n)
# training_data = seq_prep(30, training_samples_n=training_samples_n)
training_data, info, scaler = seqseq_prep(h_len=40, f_len=40, training_samples_n=training_samples_n)
training_data[0].shape

# %%
feature = training_data[0][0:10000, -1]
feature
_ = plt.hist(feature, bins=150)

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
            from models.core import dnn
            reload(dnn)
            from models.core.dnn import  Encoder
            self.model = Encoder(config)

        if self.model_type == 'lstm':
            from models.core import lstm
            reload(lstm)
            from models.core.lstm import  Encoder
            self.model = Encoder(config)

        elif self.model_type == 'lstm_idm':
            from models.core import lstm_idm
            reload(lstm_idm)
            from models.core.lstm_idm import  Encoder
            self.model = Encoder(config, model_use='training')

        elif self.model_type == 'lstm_seq_idm':
            from models.core import lstm_seq_idm
            reload(lstm_seq_idm)
            from models.core.lstm_seq_idm import  Encoder
            self.model = Encoder(config, model_use='training')

    def train(self, training_data, epochs):
        train_indx = int(len(training_data[0])*0.8)
        if self.model_type == 'dnn':
            xs_c, ys_c = training_data
            train_input = [xs_c[0:train_indx, 1:], ys_c[0:train_indx, 1:]]
            val_input = [xs_c[train_indx:, 1:], ys_c[train_indx:, 1:]]

        elif self.model_type == 'lstm':
            xs_h, _, ys_c = training_data
            train_input = [xs_h[0:train_indx, :, 1:], ys_c[0:train_indx, 1:]]
            val_input = [xs_h[train_indx:, :, 1:], ys_c[train_indx:, 1:]]

        elif self.model_type == 'lstm_idm':
            xs_h, xs_c, ys_c = training_data
            train_input = [xs_h[0:train_indx, :, 1:], xs_c[0:train_indx, 1:], \
                                            ys_c[0:train_indx, 1:]]
            val_input = [xs_h[train_indx:, :, 1:], xs_c[train_indx:, 1:], \
                                            ys_c[train_indx:, 1:]]

        elif self.model_type == 'lstm_seq_idm':
            xs_h, xs_f, ys_f = training_data
            train_input = [xs_h[0:train_indx, :, 1:], xs_f[0:train_indx, :, 1:], \
                                            ys_f[0:train_indx, :, 1:]]
            val_input = [xs_h[train_indx:, :, 1:], xs_f[train_indx:, :, 1:], \
                                            ys_f[train_indx:, :, 1:]]

        for epoch in range(epochs):
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
            self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
            print(self.epoch_count, 'epochs completed')
            self.epoch_count += 1

    def save_model(self, model_name):
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

# model_trainer = Trainer(model_type='dnn')
# model_trainer = Trainer(model_type='lstm')
# model_trainer = Trainer(model_type='lstm_idm')
model_trainer = Trainer(model_type='lstm_seq_idm')
# training_data[0][:,:,-1].min()
# %%
model_trainer.train(training_data, epochs=10)
plt.plot(model_trainer.valid_loss)
plt.plot(model_trainer.train_loss)

plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
print(model_trainer.valid_loss[-1])

# %%
# %%
model_trainer.save_model(model_name ='lstm_seq4s_idm')
# model_trainer.save_model(model_name = model_trainer.model_type)
# %%
exp_dir = './models/experiments/lstm_seq_idm/model'
exp_dir = './models/experiments/dnn/model'


# %%
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
