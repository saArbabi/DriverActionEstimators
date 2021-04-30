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
training_samples_n = 6000
# training_data = dnn_prep(training_samples_n)
# training_data = seq_prep(30, training_samples_n=training_samples_n)
training_data, info, scaler = seqseq_prep(h_len=20, f_len=20, training_samples_n=training_samples_n)
training_data[1].shape

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
            from models.core.driver_model import  Encoder
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

        elif self.model_type == 'lstm_seq_idm' or self.model_type == 'vae_idm'\
                                         or self.model_type == 'driver_model':
            xs_h, xs_f, ys_f = training_data
            train_input = [xs_h[0:train_indx, :, 1:], xs_f[0:train_indx, :, 1:], \
                                            ys_f[0:train_indx, :, 1:]]
            val_input = [xs_h[train_indx:, :, 1:], xs_f[train_indx:, :, 1:], \
                                            ys_f[train_indx:, :, 1:]]


        for epoch in range(epochs):
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'vae_idm':
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
# model_trainer.train(training_data, epochs=5)
# plt.figure()
# plt.plot(model_trainer.valid_mseloss)
# plt.plot(model_trainer.train_mseloss)
# plt.legend(['val', 'train'])
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss (MSE)')
# plt.title('MSE')
#
# plt.figure()
# plt.plot(model_trainer.valid_klloss)
# plt.plot(model_trainer.train_klloss)
# plt.legend(['val', 'train'])
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss (KL)')
# plt.title('KL')
model_trainer.train(training_data, epochs=5)
loss_view_lim = 0

plt.plot(model_trainer.valid_loss[loss_view_lim:])
plt.plot(model_trainer.train_loss[loss_view_lim:])

plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
print(model_trainer.valid_loss[-1])
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
# model_name ='lstm_seq2s_idm'
model_name ='driver_model'
model_trainer.save_model(model_name =model_name)
# model_trainer.save_model(model_name = model_trainer.model_type)
# %%
exp_dir = './models/experiments/driver_model/model'
# exp_dir = './models/experiments/dnn/model'
#
# %%
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)

# %%
"""visualse latent vector.
"""
model_trainer.model.model_use = 'debug'

xs_h, xs_f, ys_f = training_data
train_indx = int(len(xs_h)*0.8)

xs_h = xs_h[train_indx:, :, :]
xs_f = xs_f[train_indx:, :, :]
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
    a_, mean, logvar = model_trainer.model([xs_h[indx, :, 1:], \
                                xs_f[indx, :, 1:]])
    samples = model_trainer.model.sample([mean, logvar]).numpy()
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
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf

class Encoder(AbstractModel):
    def __init__(self, config, model_use):
        super(Encoder, self).__init__(config)
        self.enc_units = 50
        self.model_use = model_use # can be training or inference
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1, activation=K.exp)
        self.neu_min_jamx = Dense(1)
        self.neu_max_act = Dense(1, activation=K.exp)
        self.neu_min_act = Dense(1, activation=K.exp)
        self.attention_lstm = LSTM(100, return_sequences=True, return_state=True)
        self.neu_attention = TimeDistributed(Dense(1, K.sigmoid))
        self.neu_attention_1 = TimeDistributed(Dense(100, activation=K.relu))
        self.neu_attention_2 = TimeDistributed(Dense(100, activation=K.relu))
        self.neu_attention_3 = TimeDistributed(Dense(40, activation=K.relu))
        self.neu_attention_4 = TimeDistributed(Dense(1, K.sigmoid))

    def attention(self, context):
        x = self.neu_attention_1(context)
        x = self.neu_attention_2(x)
        x = self.neu_attention_3(x)
        return self.neu_attention_4(x)

    def param_activation(self, batch_size, x, min_val, max_val):
        activation_function = tf.tanh(x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def compute_idm_param(self, x, batch_size):
        desired_v = self.param_activation(batch_size, self.neu_desired_v(x), 15., 35.)
        min_jamx = tf.abs(self.param_activation(batch_size, self.neu_min_jamx(x), -5., 5.))
        desired_tgap = self.neu_desired_tgap(x)
        max_act = self.neu_max_act(x)
        min_act = self.neu_min_act(x)
        return [desired_v, desired_tgap, min_jamx, max_act, min_act]

    def idm(self, vel, dv, dx, idm_param):
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param

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

        act = tf.multiply(max_act, subtract_2)
        return act

    # def apply_alphas(self, act_fl_seq, act_fm_seq, alphas):
    #     great_bool = tf.cast(tf.math.greater_equal(alphas, 0.5), dtype='float')
    #     less_bool = tf.cast(tf.math.less(alphas, 0.5), dtype='float')
    #     act_seq = tf.math.add(tf.multiply(great_bool, act_fl_seq), tf.multiply(less_bool, act_fm_seq))
    #     return act_seq

    def apply_alphas(self, act_fl_seq, act_fm_seq, alphas):
        # great_bool = tf.cast(tf.math.greater_equal(alphas, 0.5), dtype='float')
        # less_bool = tf.cast(tf.math.less(alphas, 0.5), dtype='float')
        act_seq = tf.math.add(tf.multiply(alphas, act_fl_seq), tf.multiply((1-alphas), act_fm_seq))
        return act_seq

    def idm_sim(self, state, h_t):
        # state: [v, dv, dx]
        batch_size = tf.shape(state)[0]
        idm_param = self.compute_idm_param(h_t, batch_size)

        # alpha = tf.fill([batch_size, 1], 0.6)
        # alphas = tf.reshape(alpha, [batch_size, 1, 1])
        # tf.print(tf.reduce_min(alpha))
        # tf.print(tf.slice(alpha, [0, 0], [1, 1]))
        # tf.print(h_t)
        context = tf.slice(state, [0, 0, 0], [batch_size, 1, 8])
        # context = tf.reshape(context, [batch_size, 8])
        # outputs, state_h_m, state_c_m = self.attention_lstm(context)
        h_t = tf.reshape(h_t, [batch_size, 1, self.enc_units]) # encoder hidden state
        params = tf.reshape(tf.concat(idm_param, axis=1), [batch_size, 1, 5]) # encoder hidden state

        # outputs, state_h_m, state_c_m = self.attention_lstm(tf.concat([h_t, context], axis=2))
        state_h_m, state_c_m = tf.zeros([batch_size, 100]), tf.zeros([batch_size, 100])
        # tf.print(tf.shape(state_h_m))
        # tf.print(tf.shape(state_c_m))
        if self.model_use == 'training' or self.model_use == 'debug':
            act_fl_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            act_fm_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            alphas = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

            for step in tf.range(20):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (act_fl_seq, tf.TensorShape([None,None,None])),
                                (act_fm_seq, tf.TensorShape([None,None,None])),
                                 (alphas, tf.TensorShape([None,None,None]))])

                vel = tf.slice(state, [0, step, 0], [batch_size, 1, 1])
                dv = tf.slice(state, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 3], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fl_act = self.idm(vel, dv, dx, idm_param)

                dv = tf.slice(state, [0, step, 5], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 6], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fm_act = self.idm(vel, dv, dx, idm_param)

                # fl_act_true = tf.slice(state, [0, step, 7], [batch_size, 1, 1])
                # fm_act_true = tf.slice(state, [0, step, 8], [batch_size, 1, 1])
                # fl_act_true = tf.reshape(fl_act_true, [batch_size, 1])
                # fm_act_true = tf.reshape(fm_act_true, [batch_size, 1])
                # alpha = self.attention(tf.fill([batch_size, 1], 0.7))
                # alpha = tf.fill([batch_size, 1], 0.3)
                context = tf.slice(state, [0, step, 0], [batch_size, 1, 8])
                # context = tf.reshape(context, [batch_size, 8])

                outputs, state_h_m, state_c_m = self.attention_lstm(tf.concat([h_t, context, params], axis=2), \
                                        initial_state=[state_h_m, state_c_m])
                # alpha = self.attention(outputs)
                alpha = self.attention(tf.concat([outputs, context, params], axis=2))

                # alpha = self.neu_attention(outputs)

                # alpha = self.attention(h_t)

                # alpha = 1
                # alpha = tf.reshape(alpha, [batch_size, 1])
                # act = (1-alpha)*fl_act + (alpha)*fm_act
                act_fl_seq = tf.concat([act_fl_seq, tf.reshape(fl_act, [batch_size, 1, 1])], axis=1)
                act_fm_seq = tf.concat([act_fm_seq, tf.reshape(fm_act, [batch_size, 1, 1])], axis=1)
                alphas = tf.concat([alphas, alpha], axis=1)
                #
                # tf.print(alpha)
                # tf.print(dy)

            act_seq = self.apply_alphas(act_fl_seq, act_fm_seq, alphas)
            return act_seq
            # return act_seq, idm_param

        elif self.model_use == 'inference':
            return idm_param

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs[0])
        if self.model_use == 'training' or self.model_use == 'debug':
            action = self.idm_sim(inputs[1], h_t)
            return action

        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1], h_t)
            return idm_param
