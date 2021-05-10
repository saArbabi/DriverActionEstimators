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
        train_indx = int(len(training_data[0])*0.7)
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

# -%%
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

train_loss = model_trainer.train_loss[loss_view_lim:]
valid_loss = model_trainer.valid_loss[loss_view_lim:]
plt.plot(valid_loss)
plt.plot(train_loss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
print(model_trainer.valid_loss[-1])
# %%
from scipy.stats import norm
x = np.linspace(-2, 2, 1000)
y = -0.1*(abs(np.tanh(5*(x-0.5))) - 1)
# y = (tf.tanh(x))**2
# y = np.exp(x)
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
from tensorflow.keras.layers import Dense, LSTM
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

    def attention_loss(self, alphas, mse_loss):
        # return tf.reduce_mean(tf.abs(tf.sigmoid(alphas)))
        return tf.reduce_mean(-mse_loss*0.1*(tf.abs(tf.tanh(5*(alphas-0.5))) - 1))

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, alphas = self(states)
            mse_loss = self.mse(targets, act_pred)
            loss = mse_loss
            # loss = mse_loss + self.attention_loss(alphas, mse_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, alphas = self(states)
        mse_loss = self.mse(targets, act_pred)
        loss = mse_loss
        # loss = mse_loss + self.attention_loss(alphas, mse_loss)

        self.test_loss.reset_states()
        self.test_loss(loss)

    def param_activation(self, batch_size, x, min_val, max_val):
        activation_function = tf.tanh(x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def architecture_def(self):
        self.histroy_enc = LSTM(self.enc_units, return_state=True)
        self.future_dec = LSTM(self.enc_units, return_sequences=True, return_state=True)

        self.des_v_layer = Dense(self.enc_units)
        self.des_v_neu = Dense(1)

        self.des_tgap_layer = Dense(self.enc_units)
        self.des_tgap_neu = Dense(1)

        self.min_jamx_layer = Dense(self.enc_units)
        self.min_jamx_neu = Dense(1)

        self.max_act_layer = Dense(self.enc_units)
        self.max_act_neu = Dense(1)

        self.min_act_layer = Dense(self.enc_units)
        self.min_act_neu = Dense(1)

        self.attention_layer = Dense(self.enc_units)
        self.attention_neu = Dense(1, K.sigmoid)

    def compute_idm_param(self, x, current_vel):
        desired_v = self.get_des_v(x, current_vel)
        desired_tgap = self.get_des_tgap(x)
        min_jamx = self.get_min_jamx(x)
        max_act = self.get_max_act(x)
        min_act = self.get_min_act(x)
        return [desired_v, desired_tgap, min_jamx, max_act, min_act]

    def get_des_v(self, x, current_v):
        input = self.des_v_layer(x)
        output = self.des_v_neu(input) + current_v
        return output

    def get_des_tgap(self, x):
        input = self.des_tgap_layer(x)
        output = tf.abs(self.des_tgap_neu(input)) + 1
        return output

    def get_min_jamx(self, x):
        input = self.min_jamx_layer(x)
        output = tf.abs(self.min_jamx_neu(input)+1)
        return output

    def get_max_act(self, x):
        input = self.max_act_layer(x)
        output = tf.abs(self.max_act_neu(input)) + 0.5
        return output

    def get_min_act(self, x):
        input = self.min_act_layer(x)
        output = tf.abs(self.min_act_neu(input)) + 0.5
        return output

    def idm_driver(self, vel, dv, dx, idm_param):
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param
        desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)
        return act

    def apply_alphas(self, act_fl_seq, act_fm_seq, alphas):
        # great_bool = tf.cast(tf.math.greater_equal(alphas, 0.5), dtype='float')
        # less_bool = tf.cast(tf.math.less(alphas, 0.5), dtype='float')
        act_seq = tf.math.add(tf.multiply(alphas, act_fl_seq), tf.multiply((1-alphas), act_fm_seq))
        return act_seq

    def get_attention(self, x):
        return self.attention_neu(self.attention_layer(x))

    def idm_sim(self, env_states, encoder_states):
        # env_states: [v, dv, dx]
        batch_size = 256
        # batch_size = tf.shape(env_states)[0]
        h_t, c_t = encoder_states
        #
        # desired_v = self.get_des_v(h_t, env_states[:, 0, 0:1])
        # desired_tgap = self.get_des_tgap(h_t)
        # min_jamx = self.get_min_jamx(h_t)
        # max_act = self.get_max_act(h_t)
        # min_act = self.get_min_act(h_t)

        # idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)
        # tf.print('desired_v: ', tf.reduce_mean(desired_v))
        # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        # tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
        # tf.print('max_act: ', tf.reduce_mean(max_act))
        # tf.print('min_act: ', tf.reduce_mean(min_act))

        if self.model_use == 'training' or self.model_use == 'debug':
            act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            # idm_param = tf.zeros([batch_size, 1], dtype=tf.float32)
            alphas = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            desired_v = tf.zeros([batch_size, 1], dtype=tf.float32)
            desired_tgap = tf.zeros([batch_size, 1], dtype=tf.float32)
            min_jamx = tf.zeros([batch_size, 1], dtype=tf.float32)
            max_act = tf.zeros([batch_size, 1], dtype=tf.float32)
            min_act = tf.zeros([batch_size, 1], dtype=tf.float32)

            for step in tf.range(20):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                # (act_fl_seq, tf.TensorShape([None,None,None])),
                                # (act_fm_seq, tf.TensorShape([None,None,None])),
                                (alphas, tf.TensorShape([None,None,None])),
                                (act_seq, tf.TensorShape([None,None,None]))])

                s = env_states[:, step:step+1, :]
                vel = tf.slice(env_states, [0, step, 0], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                outputs, h_t, c_t = self.future_dec(s, initial_state=[h_t, c_t])
                # tf.print(tf.shape(outputs))
                outputs = tf.reshape(outputs, [batch_size, self.enc_units])
                idm_param = self.compute_idm_param(outputs, vel)
                desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param


                dv = tf.slice(env_states, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(env_states, [0, step, 3], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fl_act = self.idm_driver(vel, dv, dx, idm_param)

                dv = tf.slice(env_states, [0, step, 5], [batch_size, 1, 1])
                dx = tf.slice(env_states, [0, step, 6], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fm_act = self.idm_driver(vel, dv, dx, idm_param)

                # alpha = tf.fill([batch_size, 1], 1.)
                # alpha = self.get_attention(outputs)
                alpha = self.get_attention(tf.concat([fl_act, fm_act, outputs], axis=1))

                act = alpha*fl_act + (1-alpha)*fm_act
                # tf.math.add(tf.multiply(alphas, fl_act), tf.multiply((1-alphas), fm_act))
                # act_fl_seq = tf.concat([act_fl_seq, tf.reshape(fl_act, [batch_size, 1, 1])], axis=1)
                # act_fm_seq = tf.concat([act_fm_seq, tf.reshape(fm_act, [batch_size, 1, 1])], axis=1)
                alphas = tf.concat([alphas, tf.reshape(alpha, [batch_size, 1, 1])], axis=1)
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)

            tf.print('######')
            tf.print('desired_v: ', tf.reduce_mean(desired_v))
            tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
            tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
            tf.print('max_act: ', tf.reduce_mean(max_act))
            tf.print('min_act: ', tf.reduce_mean(min_act))
            tf.print('alphas: ', tf.reduce_min(alphas))
            tf.print('alphas: ', tf.reduce_max(alphas))

            return act_seq, alphas
            # return act_seq, idm_param

        # elif self.model_use == 'inference':
        #     s = env_states[:, 0:1, :]
        #     vel = tf.slice(env_states, [0, 0, 0], [batch_size, 1, 1])
        #     dv = tf.slice(env_states, [0, 0, 2], [batch_size, 1, 1])
        #     dx = tf.slice(env_states, [0, 0, 3], [batch_size, 1, 1])
        #     vel = tf.reshape(vel, [batch_size, 1])
        #     dv = tf.reshape(dv, [batch_size, 1])
        #     dx = tf.reshape(dx, [batch_size, 1])
        #
        #     outputs, h_t, c_t = self.future_dec(s, initial_state=[h_t, c_t])
        #     # tf.print(tf.shape(outputs))
        #     outputs = tf.reshape(outputs, [batch_size, self.enc_units])
        #     desired_v = self.get_des_v(outputs, s[:, 0, 0:1])
        #     desired_tgap = self.get_des_tgap(outputs)
        #     min_jamx = self.get_min_jamx(outputs)
        #     max_act = self.get_max_act(outputs)
        #     min_act = self.get_min_act(outputs)
        #
        #     idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)
        #
        #     return idm_param

    def call(self, inputs):
        _, h_t, c_t = self.histroy_enc(inputs[0])
        if self.model_use == 'training' or self.model_use == 'debug':
            action = self.idm_sim(inputs[1], [h_t, c_t])
            return action

        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1], [h_t, c_t])
            return idm_param
