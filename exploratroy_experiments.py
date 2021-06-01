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
training_data, info, scaler = seqseq_prep(h_len=20, f_len=20, training_samples_n=training_samples_n)
print(training_data[0].shape)
# scaler.mean_
# scaler.var_
# dir(scaler)

# training_data[3][0, -1, :]
training_data[0][50, 1, :]

# %%
7 == 7 and not (3 == 3 and 1 == 3)
# %%

for i in range(1, 9):
    plt.figure()
    feature = training_data[2][0:100000, -1, i]
    feature.max()
    _ = plt.hist(feature, bins=150)
# %%

for i in range(1, 10):
    plt.figure()
    feature = training_data[0][0:10000, -1, i]
    feature.max()
    _ = plt.hist(feature, bins=150)

# %%
feature = training_data[0][0:10000, -1, -3]
feature.max()
_ = plt.hist(feature, bins=150)
# %%
# feat = training_data[0][0:10000, -1, :]
to_plot = feat[feat[:, -2] == 1][:, -1]
_ = plt.hist(to_plot, bins=150)
# %%
to_plot = feat[feat[:, -2] == 1][:, -7]
_ = plt.hist(to_plot, bins=150)
plt.xlabel('Hist')
plt.ylabel('Relative x - merger car')
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
    def __init__(self, model_type):
        self.model = None
        self.model_type = model_type
        self.train_loss = []
        self.valid_loss = []

        self.train_mseloss = []
        self.train_klloss = []

        self.valid_mseloss = []
        self.valid_klloss = []
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

        elif self.model_type == 'vae_idm':
            from models.core import vae
            reload(vae)
            from models.core.vae import  VAEIDM
            self.model = VAEIDM(config, model_use='training')

        elif self.model_type == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import  NeurIDMModel
            self.model = NeurIDMModel(config, model_use='training')

    def train(self, training_data, epochs):
        train_indx = int(len(training_data[0])*0.8)
        self.model.epochs_n = epochs

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

        elif self.model_type == 'lstm_seq_idm' or self.model_type == 'vae_idm':
            xs_h, xs_f, ys_f = training_data
            train_input = [xs_h[0:train_indx, :, 1:], xs_f[0:train_indx, :, 1:], \
                                            ys_f[0:train_indx, :, 1:]]
            val_input = [xs_h[train_indx:, :, 1:], xs_f[train_indx:, :, 1:], \
                                            ys_f[train_indx:, :, 1:]]

        elif self.model_type == 'driver_model':
            xs_h, scaled_xs_f, unscaled_xs_f, ys_f = training_data
            train_input = [xs_h[0:train_indx, :, 1:],
                        scaled_xs_f[0:train_indx, :, 1:],
                        unscaled_xs_f[0:train_indx, :, 1:],
                        ys_f[0:train_indx, :, 1:]]

            val_input = [xs_h[train_indx:, :, 1:],
                        scaled_xs_f[train_indx:, :, 1:],
                        unscaled_xs_f[train_indx:, :, 1:],
                        ys_f[train_indx:, :, 1:]]


        for epoch in range(epochs):
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'vae_idm' or self.model_type == 'driver_model':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.valid_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.valid_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))
            else:
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
# model_trainer = Trainer(model_type='lstm_seq_idm')
# model_trainer = Trainer(model_type='vae_idm')
model_trainer = Trainer(model_type='driver_model')
# training_data[0][:,:,-1].min()

# %%
model_trainer.train(training_data, epochs=5)
plt.figure()
plt.plot(model_trainer.valid_mseloss)
plt.plot(model_trainer.train_mseloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
plt.title('MSE')

plt.figure()
plt.plot(model_trainer.valid_klloss)
plt.plot(model_trainer.train_klloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (KL)')
plt.title('KL')
# model_trainer.train(training_data, epochs=10)
# loss_view_lim = 0
#
# train_loss = model_trainer.train_loss[loss_view_lim:]
# valid_loss = model_trainer.valid_loss[loss_view_lim:]
# plt.plot(valid_loss)
# plt.plot(train_loss)
# plt.legend(['val', 'train'])
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss (MSE)')
# # model_trainer.model.sigma
# print(model_trainer.valid_loss[-1])
# %%
# model_name ='lstm_seq2s_idm'
model_name ='testing_car'
model_trainer.save_model(model_name =model_name)
# model_trainer.save_model(model_name = model_trainer.model_type)
# %%
model_trainer.

t = tf.constant([[-10., -1., 0.], [0.5, 2., 10.]])
t2 = tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1)
t2.numpy()
# %%
# val_compare = {}
# val_compare['2'] = valid_loss
val_compare['good'] = valid_loss

#- %%
for f, loss_val in val_compare.items():
    plt.plot(loss_val)
plt.legend(val_compare.keys())
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
plt.grid()
# %%
from scipy.stats import norm
for i in [0.5, 1, 2, 5]:
    x = np.linspace(-5, 5, 1000)
    # y = -5*(abs(np.tanh(5*(x-0.5))) - 1)
    # y = (tf.tanh(x))**2
    # y = np.exp(x)
    y = 1/(1+np.exp(-i*x))
    # y = (x-1)**2
    # plt.plot(x, y)
    # y = (x+0)**2
    plt.plot(x, y)
plt.grid()
# %%
idm_param = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_act':1.4, # m/s^2
                'min_act':2, # m/s^2
                }

desired_v = idm_param['desired_v']
desired_tgap = idm_param['desired_tgap']
min_jamx = idm_param['min_jamx']
max_act = idm_param['max_act']
min_act = idm_param['min_act']
follower_v = 20
lead_v = 20
dx = 20
dv = 0.1

des_options = np.linspace(0.5, 3, 100)
actions = []
for des in des_options:
    min_act = des
    desired_gap = min_jamx + desired_tgap*follower_v+(follower_v*dv)/ \
                                    (2*np.sqrt(max_act*min_act))

    acc = max_act*(1-(follower_v/desired_v)**4-\
                                        (desired_gap/dx)**2)

    actions.append(acc)
plt.grid()
plt.xlabel('Param range (min_act)')
plt.ylabel('Action value')
plt.plot(des_options, actions)

# %%


# %%
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)

# %%
"""visualse latent vector.
"""
# model_trainer.model.model_use = 'inference'

xs_h, xs_f, xs_f, ys_f = training_data
train_indx = int(len(xs_h)*0.8)

# %%
import tensorflow as tf
for i in indxs[0:10]:
    # encoder_states = model_trainer.model.encoder(np.zeros([1, :, 1:]))
    encoder_states = model_trainer.model.encoder(xs_h[i:i+1, 0:1, 1:])
    z_mean, z_log_sigma = model_trainer.model.belief_estimator(encoder_states[0])
    tf.print(np.exp(z_log_sigma.numpy()))
encoder_states[0].shape

# %%

xs_h = xs_h[train_indx:, :, :]
xs_f = xs_f[train_indx:, :, :]
# xs_f = xs_f[train_indx:, :, :]
ys_f = ys_f[train_indx:, :, :]

indxs = np.random.choice(range(len(xs_h)), 500, replace=False)
episodes = xs_h[indxs, 0, 0]
tim = []
norm = []
agg = []

for indx, epis in zip(indxs.tolist(), episodes.tolist()):
    if info[epis] == 'timid':
        tim.append(indx)
    elif info[epis] == 'normal':
        norm.append(indx)
    elif info[epis] == 'aggressive':
        agg.append(indx)
#
# for epis in episodes:
#     if info[epis]

# %%
def latent_samples(model_trainer, indx):
    encoder_states = model_trainer.model.encoder(xs_h[indx, :, 1:])
    mean, logvar = model_trainer.model.belief_estimator(encoder_states[0])
    samples = model_trainer.model.belief_estimator.sample_z([mean, logvar]).numpy()

    return samples

samples = latent_samples(model_trainer, agg)
plt.scatter(samples[:, 0], samples[:, 1], color='red')
samples = latent_samples(model_trainer, tim)
plt.scatter(samples[:, 0], samples[:, 1], color='green')
samples = latent_samples(model_trainer, norm)
plt.scatter(samples[:, 0], samples[:, 1], color='orange')
plt.ylabel('$z_1$')
plt.xlabel('$z_2$')


# %%
a = np.array([1,2,3,4])
a[[1,2]]
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
