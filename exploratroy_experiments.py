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
    # encoder_states = model_trainer.model.encoder(np.zeros([1, :, 1:]))
    encoder_states = model_trainer.model.encoder(xs_h[i:i+1, 0:1, 1:])
    z_mean, z_log_sigma = model_trainer.model.belief_estimator(encoder_states[0])
    tf.print(np.exp(z_log_sigma.numpy()))
encoder_states[0].shape

# %%
"""visualse latent vector.
"""
xs_h, xs_f, xs_f, ys_f = training_data
train_indx = int(len(xs_h)*0.8)
xs_h = xs_h[train_indx:, :, :]
xs_f = xs_f[train_indx:, :, :]
# xs_f = xs_f[train_indx:, :, :]
ys_f = ys_f[train_indx:, :, :]

indxs = np.random.choice(range(len(xs_h)), 500, replace=False)
episodes = xs_h[indxs, 0, 0]
# np.unique(indxs).shape
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

# %%
def latent_samples(model_trainer, indx):
    encoder_states = model_trainer.model.history_enc(xs_h[indx, :, 1:])
    prior_param = model_trainer.model.belief_estimator(encoder_states[0], dis_type='prior')
    samples = model_trainer.model.belief_estimator.sample_z(prior_param).numpy()

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

for indx in agg[0: 20]:
    indx = [indx]
    plt.figure()

    xs_h = np.float32(xs_h)
    xs_f = np.float32(xs_f)
    data_sample_h = np.repeat(xs_h[indx, :, 1:], 30, axis=0)
    data_sample_f = np.repeat(xs_f[indx, :, 1:], 30, axis=0)

    train_indx = int(len(xs_h)*0.8)
    encoder_states = model_trainer.model.history_enc(data_sample_h)
    prior_param = model_trainer.model.belief_estimator(encoder_states[0], dis_type='prior')
    z = model_trainer.model.belief_estimator.sample_z(prior_param).numpy()

    context = tf.concat([z, encoder_states[0]], axis=1)
    decoder_output = model_trainer.model.decoder(context)
    current_v = data_sample_h[:, -1, 1:2]
    idm_param = model_trainer.model.idm_layer([decoder_output, current_v])

    env_states = [data_sample_f, data_sample_f]
    data_sample_f.shape
    act_seq = model_trainer.model.idm_sim.rollout([env_states, idm_param, [_, _]]).numpy()
    act_seq.shape
    for sample_trace_i in range(5):
        plt.plot(act_seq[sample_trace_i, :, :].flatten(), color='grey')
    plt.plot(ys_f[indx, :, -1].flatten(), color='red')
    plt.ylim(act_seq.mean()-2, act_seq.mean()+2)
    ys_f[indx, :, -1]
    act_seq.mean()
    plt.title(indx)
    ##########
    plt.figure()

    desired_vs = idm_param[0].numpy().flatten()
    # plt.plot(desired_vs)
    # plt.grid()
    # plt.plot(desired_tgaps)
    # plt.grid()

    desired_tgaps = idm_param[1].numpy().flatten()
    plt.scatter(desired_vs, desired_tgaps, color='grey', s=3)
    plt.scatter(30, 1., color='red')
    plt.xlim(20, 40)
    plt.ylim(0, 3)
    plt.title(indx)
