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
training_samples_n = 15000
# training_data = dnn_prep(training_samples_n)
# training_data = seq_prep(30, training_samples_n=training_samples_n)
training_data, info, scaler = seqseq_prep(h_len=20, f_len=20, training_samples_n=training_samples_n)


print(training_data[-2].shape)
# scaler.mean_
# %%
# scaler.var_
# dir(scaler)
# len(info)
plt.plot(training_data[0][0, :, -2])
plt.plot(training_data[2][0, :, 8])


training_data[2][0, 0, :]

training_data[-1][0, 0, :]

# %%
def get_random_vals(mean_vel):
    init_v = 20 + np.random.choice(range(-3, 3))
    action_magnitute = np.random.uniform(-3, 3)
    action_freq = np.random.uniform(0.02, 0.06)
    return init_v, action_magnitute, action_freq

x = np.linspace(0, 200, 1000)
for i in range(29):
    l_v, m_act_mag, m_sin_freq = get_random_vals(mean_vel=20)
    a = m_act_mag*np.sin(x*m_sin_freq)
    plt.plot(x, a)
# %%
def get_episode_data(training_data, episode_id):
    eps_data = []
    for set in training_data:
        set_ = set[set[:, 0, 0] == episode_id]
        eps_data.append(set_)
    return eps_data
# training_data[0][:, :, 0:1]
eps_data = get_episode_data(training_data, 1)
eps_data[-2].shape

set = training_data[0]
set[set[:, 0, 0] == 1]
set[set[:, 0, 0] == 1].shape

# %%

plt.plot(set[0][,: , -1])
plt.plot(range(19, 39), set[0][:, -1])
# %%
"""
Data imbalance
"""
att_l = 0
att_m = 0
att_l += np.sum( training_data[-1][:, :, -2] == 1)
att_m += np.sum( training_data[-1][:, :, -2]  == 0)

plt.bar([1, 2], [att_l, att_m])
att_l/(att_l+att_m)

# %%
""" Addressing data imbalance
"""
xs_h, scaled_xs_f, unscaled_xs_f, merger_xas, ys_f = training_data
balance_data = True

if balance_data:
    train_input = [xs_h[:, :, 1:],
                scaled_xs_f[:, :, 1:],
                unscaled_xs_f[:, :, 1:],
                merger_xas[:, :, 1:],
                ys_f[:, :, 1:]]

    balanced_training_data = []
    axis_0, axis_1 = np.where(train_input[0][:, :, -1] == 0)
    lc_samples = np.unique(axis_0).astype(int)

    set_i = 0
    for set_ in train_input:
        set_ = np.append(set_, np.repeat(set_[lc_samples, :, :], 10, axis=0), axis=0)
        balanced_training_data.append(set_)
        set_i += 1

    train_input = balanced_training_data

att_l = 0
att_m = 0
for set in balanced_training_data[0:2]:
    att_l += np.sum(set[:, 0:10, -1] == 1)
    att_m += np.sum(set[:, 0:10, -1]  == 0)

plt.bar([1, 2], [att_l, att_m])
att_l/(att_l+att_m)

# %%

# for i in range(1, 10):
plt.figure()
# feature = training_data[0][:, -1, -1]
feature = balanced_training_data[1][0:10000, -1, i]
feature.max()
_ = plt.hist(feature, bins=150)

# %%
for i in range(1, 11):
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
        train_sample_index = int(len(training_data[0])*0.8)
        self.model.epochs_n = epochs

        if self.model_type == 'dnn':
            xs_c, ys_c = training_data
            train_input = [xs_c[0:train_sample_index, 1:], ys_c[0:train_sample_index, 1:]]
            val_input = [xs_c[train_sample_index:, 1:], ys_c[train_sample_index:, 1:]]

        elif self.model_type == 'lstm':
            xs_h, _, ys_c = training_data
            train_input = [xs_h[0:train_sample_index, :, 1:], ys_c[0:train_sample_index, 1:]]
            val_input = [xs_h[train_sample_index:, :, 1:], ys_c[train_sample_index:, 1:]]

        elif self.model_type == 'lstm_idm':
            xs_h, xs_c, ys_c = training_data
            train_input = [xs_h[0:train_sample_index, :, 1:], xs_c[0:train_sample_index, 1:], \
                                            ys_c[0:train_sample_index, 1:]]
            val_input = [xs_h[train_sample_index:, :, 1:], xs_c[train_sample_index:, 1:], \
                                            ys_c[train_sample_index:, 1:]]

        elif self.model_type == 'lstm_seq_idm' or self.model_type == 'vae_idm':
            xs_h, xs_f, ys_f = training_data
            train_input = [xs_h[0:train_sample_index, :, 1:], xs_f[0:train_sample_index, :, 1:], \
                                            ys_f[0:train_sample_index, :, 1:]]
            val_input = [xs_h[train_sample_index:, :, 1:], xs_f[train_sample_index:, :, 1:], \
                                            ys_f[train_sample_index:, :, 1:]]

        elif self.model_type == 'driver_model':
            xs_h, scaled_xs_f, unscaled_xs_f, merger_xas, ys_f = training_data
            balance_data = False

            if balance_data:
                train_input = [xs_h[0:train_sample_index, :, 1:],
                            scaled_xs_f[0:train_sample_index, :, 1:],
                            unscaled_xs_f[0:train_sample_index, :, 1:],
                            merger_xas[0:train_sample_index, :, 1:],
                            ys_f[0:train_sample_index, :, 1:]]

                balanced_training_data = []
                axis_0, axis_1 = np.where(train_input[0][:, :, -1] == 0)
                lc_samples = np.unique(axis_0).astype(int)

                set_i = 0
                for set in train_input:
                    if set_i >= 3:
                        set_ = set[:, :, :]
                    else:
                        set_ = set[:, :, :-1]

                    set_ = np.append(set_, np.repeat(set_[lc_samples, :, :], 10, axis=0), axis=0)
                    balanced_training_data.append(set_)
                    set_i += 1

                train_input = balanced_training_data

            else:

                train_input = [xs_h[0:train_sample_index, :, 1:],
                            scaled_xs_f[0:train_sample_index, :, 1:],
                            unscaled_xs_f[0:train_sample_index, 20:, 1:],
                            merger_xas[0:train_sample_index, :, 1:],
                            ys_f[0:train_sample_index, 20:, 2:]]

            val_input = [xs_h[train_sample_index:, :, 1:],
                        scaled_xs_f[train_sample_index:, :, 1:],
                        unscaled_xs_f[train_sample_index:, 20:, 1:],
                        merger_xas[train_sample_index:, :, 1:],
                        ys_f[train_sample_index:, 20:, 2:]]


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
model_trainer.model.vae_loss_weight = 0.1
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
import tensorflow_probability as tfp
tfd = tfp.distributions
norm1 = tfd.Normal(loc=2., scale=3.)
norm2 = tfd.Normal(loc=0., scale=-1)
tfp.distributions.kl_divergence(norm1, norm2)

# %%
# model_name ='lstm_seq2s_idm'
model_name ='testing_car'
model_trainer.save_model(model_name =model_name)
# model_trainer.save_model(model_name = model_trainer.model_type)
# %%

# %%
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)

# %%

# %%
np.random.seed(2020)
s_h_scaled, s_f_scaled, s_hf_unscaled, merger_act, y_hf = training_data
train_sample_index = int(len(s_h_scaled)*0.8)
val_examples = range(train_sample_index, len(s_h_scaled))


s_h_scaled = np.float32(s_h_scaled)
s_hf_unscaled = np.float32(s_hf_unscaled)
merger_act = np.float32(merger_act)

timid_drivers = []
normal_drivers = []
aggressive_drivers = []

for sample_index in val_examples:
    epis = s_h_scaled[sample_index, 0, 0]
    if info[epis] == 'timid':
       timid_drivers.append(sample_index)
    elif info[epis] == 'normal':
       normal_drivers.append(sample_index)
    elif info[epis] == 'aggressive':
       aggressive_drivers.append(sample_index)
s_h_scaled.shape
len(timid_drivers)
len(normal_drivers)
len(aggressive_drivers)

# %%
def latent_samples(model_trainer, sample_index):
   enc_h = model_trainer.model.h_seq_encoder(s_h_scaled[sample_index, :, 1:])
   print(s_h_scaled.shape)
   enc_f_acts = model_trainer.model.act_encoder(merger_act[sample_index, :, 1:])
   prior_param = model_trainer.model.belief_net([enc_h, enc_f_acts], dis_type='prior')
   sampled_z = model_trainer.model.belief_net.sample_z(prior_param).numpy()

   return sampled_z

samples = latent_samples(model_trainer, aggressive_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='red')
samples = latent_samples(model_trainer, timid_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='green')
samples = latent_samples(model_trainer, normal_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='orange')
# plt.scatter(z[:, 0], z[:, 1], s=20, color='blue')

plt.ylabel('$z_1$')
plt.xlabel('$z_2$')

# %%
def latent_samples(model_trainer, sample_index):
   enc_h = model_trainer.model.h_seq_encoder(s_h_scaled[sample_index, :, 1:])
   f_enc_state = model_trainer.model.f_seq_encoder(s_f_scaled[sample_index, :, 1:])

   enc_f_acts = model_trainer.model.act_encoder(merger_act[sample_index, :, 1:])
   prior_param, posterior_param = model_trainer.model.belief_net([enc_h, f_enc_state, enc_f_acts], dis_type='both')
   sampled_z = model_trainer.model.belief_net.sample_z(posterior_param).numpy()

   return sampled_z

samples = latent_samples(model_trainer, aggressive_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='red')
samples = latent_samples(model_trainer, timid_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='green')
samples = latent_samples(model_trainer, normal_drivers)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='orange')
# plt.scatter(z[:, 0], z[:, 1], s=20, color='blue')

plt.ylabel('$z_1$')
plt.xlabel('$z_2$')

# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

# model_trainer.model.idm_sim.arbiter.attention_temp = 20
model_trainer.model.arbiter.attention_temp = 20

Example_pred = 0
traces_n = 20
i = 0
covered_episodes = []

while Example_pred < 20:
   # sample_index = [timid_drivers[i]]
   sample_index = [normal_drivers[i]]
   # sample_index = [aggressive_drivers[i]]
   i += 1
   true_attention = y_hf[sample_index, :, -2].flatten()
   m_y = s_hf_unscaled[sample_index, :, -1].flatten()
   episode = s_hf_unscaled[sample_index, 0, 0][0]

   # if episode not in covered_episodes and true_attention[:30].mean() == 1:
   if episode not in covered_episodes and true_attention[30:].mean() == 0 and true_attention[:10].mean() == 1:
       Example_pred += 1
       covered_episodes.append(episode)

       true_action = y_hf[sample_index, :, -1].flatten()

       sdv_actions = vectorise(merger_act[sample_index, :, 1:], traces_n)
       h_seq = vectorise(s_h_scaled[sample_index, :, 1:], traces_n)
       hf_seq_unscaled = vectorise(s_hf_unscaled[sample_index, 20:, 1:], traces_n)
       enc_h = model_trainer.model.h_seq_encoder(h_seq)
       enc_f_acts = model_trainer.model.act_encoder(sdv_actions)
       prior_param = model_trainer.model.belief_net([enc_h, enc_f_acts], dis_type='prior')
       sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
       att_scores = model_trainer.model.arbiter(sampled_att_z)
       # att_scores =  model_trainer.model.arbiter(sampled_z)

       idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
       idm_params = tf.reshape(idm_params, [traces_n, 1, 5])
       idm_params = tf.repeat(idm_params, 20, axis=1)

       act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, hf_seq_unscaled])
       act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
       plt.figure()
       for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), color='grey')
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')
       plt.plot(true_action[:20].flatten(), color='black', linestyle='--')
       plt.plot(range(20, 40), true_action[20:].flatten(), color='red', linestyle='--')
       plt.ylim(-3, 3)
       plt.title(str(sample_index[0]) + ' -- Action')
       plt.grid()

       plt.figure()
       plt.plot(true_attention[:20] , color='black', linestyle='--')
       plt.plot(range(20, 40), true_attention[20:], color='red', linestyle='--')

       for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
       plt.ylim(-0.1, 1.1)
       plt.title(str(sample_index[0]) + ' -- Attention')
       plt.grid()

       ##########

       # plt.plot(desired_vs)
       # plt.grid()
       # plt.plot(desired_tgaps)
       # plt.grid()

       plt.figure()
       desired_vs = idm_params.numpy()[:, 0, 0]
       desired_tgaps = idm_params.numpy()[:, 0, 1]
       plt.scatter(desired_vs, desired_tgaps, color='grey')

       plt.scatter(25, 1.4, color='red')
       # plt.scatter(30, 1.4, color='red')
       plt.xlim(15, 40)
       plt.ylim(0, 3)
       #
       # plt.scatter(30, 1, color='red')
       # plt.xlim(25, 35)
       # plt.ylim(0, 2)

       plt.title(str(sample_index[0]) + ' -- Param')
       plt.grid()

       ##########
       plt.figure()
       plt.plot(m_y[:20], color='black', linestyle='--')
       plt.plot(range(20, 40), m_y[20:], color='red', linestyle='--')
       plt.plot([0, 40], [-1, -1])
       plt.title(str(sample_index[0]) + ' -- m_y')
       plt.grid()
       ############

# %%

# %%
"""Single sample Anticipation visualisation
"""
model_trainer.model.arbiter.attention_temp = 20
traces_n = 20
sample_index = [5741]

true_attention = y_hf[sample_index, :, -2].flatten()
m_y = s_hf_unscaled[sample_index, :, -1].flatten()
episode = s_hf_unscaled[sample_index, 0, 0][0]


true_action = y_hf[sample_index, :, -1].flatten()

sdv_actions = vectorise(merger_act[sample_index, :, 1:], traces_n)
h_seq = vectorise(s_h_scaled[sample_index, :, 1:], traces_n)
hf_seq_unscaled = vectorise(s_hf_unscaled[sample_index, 20:, 1:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
enc_f_acts = model_trainer.model.act_encoder(sdv_actions)
prior_param = model_trainer.model.belief_net([enc_h, enc_f_acts], dis_type='prior')
sampled_z = model_trainer.model.belief_net.sample_z(prior_param).numpy()
att_scores =  model_trainer.model.arbiter(sampled_z)
# att_scores =  model_trainer.model.arbiter(sampled_z)

idm_params = tf.repeat(tf.constant([[25, 1.5, 2, 1.4, 2]]), 20, axis=0)
idm_params = tf.reshape(idm_params, [1, 20, 5])
idm_params = tf.repeat(idm_params, traces_n, axis=0)

act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, hf_seq_unscaled])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
plt.figure()
for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), color='grey')
   # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')
plt.plot(true_action[:20].flatten(), color='black', linestyle='--')
plt.plot(range(20, 40), true_action[20:].flatten(), color='red', linestyle='--')
plt.ylim(-3, 3)
plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

plt.figure()
plt.plot(true_attention[:20] , color='black', linestyle='--')
plt.plot(range(20, 40), true_attention[20:], color='red', linestyle='--')

for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
plt.ylim(-0.1, 1.1)
plt.title(str(sample_index[0]) + ' -- Attention')
plt.grid()

##########

# plt.plot(desired_vs)
# plt.grid()
# plt.plot(desired_tgaps)
# plt.grid()

plt.figure()
desired_vs = idm_param[0].numpy().flatten()
desired_tgaps = idm_param[1].numpy().flatten()
plt.scatter(desired_vs, desired_tgaps, color='grey', s=3)

# plt.scatter(19.4, 2, color='red')
# plt.xlim(15, 25)
# plt.ylim(1, 3)

# plt.scatter(25, 1.4, color='red')
# plt.xlim(20, 30)
# plt.ylim(0, 3)
#
plt.scatter(30, 1, color='red')
plt.xlim(25, 35)
plt.ylim(0, 2)

plt.title(str(sample_index[0]) + ' -- Param')
plt.grid()

##########
plt.figure()
plt.plot(m_y[:20], color='black', linestyle='--')
plt.plot(range(20, 40), m_y[20:], color='red', linestyle='--')
plt.plot([0, 40], [-1, -1])
plt.title(str(sample_index[0]) + ' -- m_y')
plt.grid()
############


# %%
k = 0.3
w = 1.85
x = np.linspace(0, w, 100)
color = ['green', 'orange', 'red']
driver_types = ['Timid', 'Normal', 'Aggressive']
i=0
for k in [0.1, 1.5, 6]:
   y = (np.exp(k*x)-1)/(np.exp(k*w)-1)
   plt.plot(x, y, color=color[i])
   i+=1
plt.legend(driver_types)
plt.xlabel('Lateral position (m)')
plt.ylabel('Attention probability (%)')
# %%
count = 0
counts = []
p = 0
for i in range(2000):
   if np.random.random() < p or p >= 1:
       counts.append(count)
       count = 0
       p = 0

   count += 1
   p += .04

counts
# _ = plt.hist(counts, bins=200)
# %%

from scipy.stats import beta, gamma, norm
x = np.linspace(0, 1, 100)
p = beta.pdf(x, 10, 2)
plt.plot(x, p, color='red')
p = beta.pdf(x, 2, 10)
plt.plot(x, p, color='green')
p = beta.pdf(x,  3, 3)
plt.plot(x, p)
mean, var, skew, kurt = beta.stats(2, 10, moments='mvsk')
mean
# %%
x = np.linspace(15, 35, 100)
scale = 1
vel_mean = 20
p = normal_drivers.pdf(x, vel_mean, scale)
plt.plot(x, p, color='red')

vel_mean = 25
p = normal_drivers.pdf(x, vel_mean, scale)
plt.plot(x, p, color='red')

vel_mean = 30
p = normal_drivers.pdf(x, vel_mean, scale)
plt.plot(x, p, color='red')

# %%

samples = np.random.beta(1, 1, 80)
plt.scatter(samples*1.85, [0]*len(samples), s=5)
# plt.scatter(samples, [0]*len(samples))
plt.xlim(0, 1.85)

# %%
x = np.linspace(0, 10, 100)
y = np.random.gamma()

rv = binom(n, p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
       label='frozen pmf')
ax.legend(loc='best', frameon=False)
plt.show()


# %%
x = np.linspace(-5, 5, 1000)

for i in [5, 10, 20]:
   y = 1/(1+np.exp(-i*x))
   plt.plot(x, y)
plt.grid()


# %%

x = np.linspace(-3, 3 , 100)
scale = 10
min = 10
y = np.tanh(x)*scale + 10 + scale
plt.plot(x, y)
plt.plot([-3, 3], [29.5, 29.5])
plt.grid()
