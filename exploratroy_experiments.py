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


print(training_data[2].shape)
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
for set in training_data[0:2]:
    att_l += np.sum(set[:, 0:10, -1] == 1)
    att_m += np.sum(set[:, 0:10, -1]  == 0)

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
        set_ = np.append(set_, np.repeat(set_[lc_samples, :, :], 4, axis=0), axis=0)
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
            xs_h, scaled_xs_f, unscaled_xs_f, merger_xas, ys_f = training_data
            balance_data = False

            if balance_data:
                train_input = [xs_h[0:train_indx, :, 1:],
                            scaled_xs_f[0:train_indx, :, 1:],
                            unscaled_xs_f[0:train_indx, :, 1:],
                            merger_xas[0:train_indx, :, 1:],
                            ys_f[0:train_indx, :, 1:]]

                balanced_training_data = []
                axis_0, axis_1 = np.where(train_input[0][:, :, -1] == 0)
                lc_samples = np.unique(axis_0).astype(int)

                set_i = 0
                for set in train_input:
                    if set_i >= 3:
                        set_ = set[:, :, :]
                    else:
                        set_ = set[:, :, :-1]

                    set_ = np.append(set_, np.repeat(set_[lc_samples, :, :], 4, axis=0), axis=0)
                    balanced_training_data.append(set_)
                    set_i += 1

                train_input = balanced_training_data

            else:

                train_input = [xs_h[0:train_indx, :, 1:-1],
                            scaled_xs_f[0:train_indx, :, 1:-1],
                            unscaled_xs_f[0:train_indx, :, 1:-1],
                            merger_xas[0:train_indx, :, 1:],
                            ys_f[0:train_indx, :, 1:]]

            val_input = [xs_h[train_indx:, :, 1:-1],
                        scaled_xs_f[train_indx:, :, 1:-1],
                        unscaled_xs_f[train_indx:, :, 1:-1],
                        merger_xas[train_indx:, :, 1:],
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
model_trainer.model.vae_loss_weight = 0.2
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
import tensorflow as tf
for i in indxs[0:10]:
    # enc_h = model_trainer.model.encoder(np.zeros([1, :, 1:]))
    enc_h = model_trainer.model.encoder(xs_h[i:i+1, 0:1, 1:])
    z_mean, z_log_sigma = model_trainer.model.belief_estimator(enc_h)
    tf.print(np.exp(z_log_sigma.numpy()))
enc_h.shape

# %%

np.random.seed(2020)

xs_h, xs_f_scaled, xs_f, merger_xas, ys_f = training_data
train_indx = int(len(xs_h)*0.8)
xs_h = xs_h[train_indx:, :, :]
xs_f = xs_f[train_indx:, :, :]
xs_f_scaled = xs_f_scaled[train_indx:, :, :]
merger_xas = merger_xas[train_indx:, :, :]
ys_f = ys_f[train_indx:, :, :]
xs_h = np.float32(xs_h)
xs_f = np.float32(xs_f)
merger_xas = np.float32(merger_xas)

indxs = np.random.choice(range(len(xs_h)), 500, replace=False)
episodes = xs_h[indxs, 0, 0]
# np.unique(indxs).shape
timid_drivers = []
normal_drivers = []
aggressive_drivers = []

for indx, epis in zip(indxs.tolist(), episodes.tolist()):
    if info[epis] == 'timid':
        timid_drivers.append(indx)
    elif info[epis] == 'normal':
        normal_drivers.append(indx)
    elif info[epis] == 'aggressive':
        aggressive_drivers.append(indx)
xs_f.shape
len(timid_drivers)
len(normal_drivers)
len(aggressive_drivers)
# %%
def latent_samples(model_trainer, indx):
    enc_h = model_trainer.model.history_state_enc(xs_h[indx, :, 1:-1])
    enc_f_acts = model_trainer.model.future_action_enc(merger_xas[indx, :, 1:])
    prior_param = model_trainer.model.belief_estimator([enc_h, enc_f_acts], dis_type='prior')
    sampled_z = model_trainer.model.belief_estimator.sample_z(prior_param).numpy()

    return z

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
def latent_samples(model_trainer, indx):
    enc_h = model_trainer.model.history_state_enc(xs_h[indx, :, 1:-1])
    f_enc_state = model_trainer.model.future_state_enc(xs_f_scaled[indx, :, 1:-1])

    enc_f_acts = model_trainer.model.future_action_enc(merger_xas[indx, :, 1:])
    prior_param, posterior_param = model_trainer.model.belief_estimator([enc_h, f_enc_state[0], enc_f_acts[0]], dis_type='both')
    sampled_z = model_trainer.model.belief_estimator.sample_z(posterior_param).numpy()

    return z

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
model_trainer.model.idm_sim.arbiter.attention_temp = 5
Example_pred = 0
traces_n = 20
i = 0
covered_episodes = []
while Example_pred < 20:
    indx = [timid_drivers[i]]
    # indx = [normal_drivers[i]]
    # indx = [aggressive_drivers[i]]
    i += 1
    data_sample_h = np.repeat(xs_h[indx, :, 1:-1], traces_n, axis=0)
    data_sample_f_scaled = np.repeat(xs_f_scaled[indx, :, 1:-1], traces_n, axis=0)
    data_sample_f = np.repeat(xs_f[indx, :, 1:-1], traces_n, axis=0)
    data_sample_merger_xas = np.repeat(merger_xas[indx, :, 1:], traces_n, axis=0)
    # avg_att_h = abs(data_sample_h[:, :, -2]).min()
    # avg_att_f = abs(data_sample_f[:, :, -2]).min()
    # avg_att_1 = xs_h[indx, :, -1].mean()
    avg_att_1 =  xs_h[indx, :, -1].mean()
    avg_att_2 = xs_f_scaled[indx, 5:, -1].mean()
    # if avg_att == 1 or avg_att == 0:
    # if avg_att_1 == 1 and avg_att_2 != 1:
    episode = xs_h[indx, :, 0][0][0]
    # if episode not in covered_episodes:
    if episode not in covered_episodes and avg_att_1 != 1 or avg_att_2 != 1:
    # if episode not in covered_episodes and abs(data_sample_f[:, :, -1]).max() > 1:
        covered_episodes.append(episode)
        enc_h = model_trainer.model.history_state_enc(data_sample_h)
        enc_f_acts = model_trainer.model.future_action_enc(data_sample_merger_xas)
        prior_param = model_trainer.model.belief_estimator([enc_h, enc_f_acts], dis_type='prior')
        sampled_z = model_trainer.model.belief_estimator.sample_z(prior_param).numpy()

        # context = z
        # context = tf.concat([z, enc_h], axis=1)
        decoder_output = model_trainer.model.decoder(sampled_z)
        # decoder_output[0, :]

        idm_param = model_trainer.model.idm_layer(enc_h)
        # ones = np.ones([traces_n, 1], dtype='float32')
        # idm_param = [ones*25, ones*1.5, ones*2, ones*1.4, ones*2]

        act_seq, att_scores = model_trainer.model.idm_sim.rollout([data_sample_f, idm_param, decoder_output])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
        plt.figure()
        for sample_trace_i in range(traces_n):
            plt.plot(act_seq[sample_trace_i, :, :].flatten(), color='grey')
            # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')
        plt.plot(ys_f[indx, :20, -1].flatten(), color='black', linestyle='--')
        plt.plot(range(19, 40), ys_f[indx, 19:, -1].flatten(), color='red', linestyle='--')
        # plt.plot(xs_h[indx, :, -2].flatten(), color='black', linestyle='--')
        # plt.plot(att_scores[0, :, :].flatten())
        # plt.ylim(act_seq.mean()-2, act_seq.mean()+2)
        # ys_f[indx, :, -1]
        # act_seq.mean()
        plt.ylim(-3, 3)
        plt.title(str(indx[0]) + ' -- Action')
        plt.grid()

        plt.figure()
        plt.plot(xs_f[indx, :20, -1].flatten(), color='black', linestyle='--')
        plt.plot(range(19, 40), xs_f[indx, 19:, -1].flatten(), color='red', linestyle='--')

        for sample_trace_i in range(traces_n):
            plt.plot(att_scores[sample_trace_i, :, :].flatten(), color='grey')
        plt.ylim(-0.1, 1.1)
        plt.title(str(indx[0]) + ' -- Attention')
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

        plt.title(str(indx[0]) + ' -- Param')
        plt.grid()

        ##########
        state_indx = -3
        plt.figure()
        plt.plot(xs_f[indx, :20, state_indx].flatten(), color='black', linestyle='--')
        plt.plot(range(19, 40), xs_f[indx, 19:, state_indx].flatten(), color='red', linestyle='--')
        plt.title(str(indx[0]) + ' -- m_y')
        plt.grid()
        ############

        Example_pred += 1
# %%


plt.plot(ys_f[indx, :20, -1].flatten(), color='black', linestyle='--')
plt.plot(range(19, 40), ys_f[indx, 19:, -1].flatten(), color='red', linestyle='--')

# %%


# indx = [667]
indx = [949]
model_trainer.model.idm_sim.arbiter.attention_temp = 20
traces_n = 30
data_sample_h = np.repeat(xs_h[indx, :, 1:-1], traces_n, axis=0)
data_sample_f_scaled = np.repeat(xs_f_scaled[indx, :, 1:-1], traces_n, axis=0)
data_sample_f = np.repeat(xs_f[indx, :, 1:-1], traces_n, axis=0)
data_sample_merger_xas = np.repeat(merger_xas[indx, :, 1:], traces_n, axis=0)

enc_h = model_trainer.model.history_state_enc(data_sample_h)
enc_f_acts = model_trainer.model.future_action_enc(data_sample_merger_xas)
prior_param = model_trainer.model.belief_estimator([enc_h, enc_f_acts], dis_type='prior')
# prior_param[1] = prior_param[1] * [0.1, .1]
sampled_z = model_trainer.model.belief_estimator.sample_z(prior_param).numpy()
# plt.scatter(z[:,0], z[:,0])
# context = z
context = tf.concat([z, enc_h], axis=1)

decoder_output = model_trainer.model.decoder(sampled_z)
idm_param = model_trainer.model.idm_layer(enc_h)

# ones = np.ones([traces_n, 1], dtype='float32')
# idm_param = [ones*25, ones*1.5, ones*2, ones*1.4, ones*2]
# idm_param = [np.random.normal(0, 1, [traces_n, 1]) + ones*19.4, ones*2, ones*4, ones*0.8, ones*1]
# idm_param = [ones*19.4, ones*2, ones*4, ones*0.8, ones*1]

act_seq, att_scores = model_trainer.model.idm_sim.rollout([data_sample_f, idm_param, decoder_output])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
plt.figure()
for sample_trace_i in range(traces_n):
    plt.plot(act_seq[sample_trace_i, :, :].flatten(), color='grey')
plt.plot(ys_f[indx, :20, -1].flatten(), color='black', linestyle='--')
plt.plot(range(19, 40), ys_f[indx, 19:, -1].flatten(), color='red', linestyle='--')
plt.ylim(-2, 2)

plt.grid()

##########
plt.figure()
plt.plot(xs_f[indx, :20, -1].flatten(), color='black', linestyle='--')
plt.plot(range(19, 40), xs_f[indx, 19:, -1].flatten(), color='red', linestyle='--')

for sample_trace_i in range(traces_n):
    plt.plot(att_scores[sample_trace_i, :, :].flatten(), color='grey')
plt.ylim(-0.1, 1.1)
plt.title(indx)
plt.grid()
##########
#
# plt.figure()
# desired_vs = idm_param[0].numpy().flatten()
# desired_tgaps = idm_param[1].numpy().flatten()
# plt.scatter(desired_vs, desired_tgaps, color='grey', s=3)
#
# plt.scatter(19.4, 2, color='red')
# plt.xlim(15, 25)
# plt.ylim(1, 3)

# plt.scatter(25, 1.4, color='red')
# plt.xlim(20, 30)
# plt.ylim(0, 3)

# plt.scatter(30, 1, color='red')
# plt.xlim(25, 35)
# plt.ylim(0, 2)

plt.title(indx)
plt.grid()
############
state_indx = -3
plt.figure()
plt.plot(xs_f[indx, :20, state_indx].flatten(), color='black', linestyle='--')
plt.plot(range(19, 40), xs_f[indx, 19:, state_indx].flatten(), color='red', linestyle='--')
plt.grid()
############
#
# plt.figure()
# desired_vs = idm_param[0].numpy().flatten()
# desired_tgaps = idm_param[1].numpy().flatten()
# plt.scatter(desired_vs, desired_tgaps, color='grey', s=3)
# plt.scatter(25, 1.4, color='red')
# plt.xlim(20, 30)
# plt.ylim(0, 3)
# plt.title(indx)
# plt.grid()






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
p = beta.pdf(x,  15, 15)
plt.plot(x, p)

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
vel_noise =
np.random.normal(0, 1)
np.random.beta(2, 2)
# %%

samples = np.random.beta(10, 2, 50)
plt.scatter(samples*1.85, [0]*len(samples))
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
