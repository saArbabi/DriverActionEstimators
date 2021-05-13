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
print(training_data[1].shape)
# scaler.mean_
# scaler.var_
# dir(scaler)

# training_data[3][0, -1, :]
# %%

for i in range(1, 10):
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

#- %%
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

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



class Encoder(AbstractModel):
    def __init__(self, config, model_use):
        super(Encoder, self).__init__(config)
        self.enc_units = 50
        self.model_use = model_use # can be training or inference
        self.architecture_def()

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, ys_f in train_ds:
            self.train_step([xs_h, scaled_xs_f, unscaled_xs_f], ys_f)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, ys_f in train_ds:
            self.test_step([xs_h, scaled_xs_f, unscaled_xs_f], ys_f)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred = self(states)
            loss = self.mse(targets, act_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred = self(states)
        loss = self.mse(targets, act_pred)
        self.test_loss.reset_states()
        self.test_loss(loss)

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
        # self.attention_neu = Dense(1, activation=K.sigmoid)
        self.attention_neu = Dense(1)

    def param_activation(self, batch_size, val, min_val, max_val):
        activation_function = tf.tanh(val)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

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
        output = tf.exp(self.des_tgap_neu(input)) + 1
        return output

    def get_min_jamx(self, x):
        input = self.min_jamx_layer(x)
        output = tf.exp(self.min_jamx_neu(input)) + 1
        return output

    def get_max_act(self, x):
        input = self.max_act_layer(x)
        output = tf.exp(self.max_act_neu(input)) + 0.5
        return output

    def get_min_act(self, x):
        input = self.min_act_layer(x)
        output = tf.exp(self.min_act_neu(input)) + 0.5
        return output

    def idm_driver(self, vel, dv, dx, idm_param):
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param
        desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)
        return self.action_clip(act)

    def get_att_score(self, x):
        x = self.attention_layer(x)
        x = self.attention_neu(x)

        return 1/(1+tf.exp(-5*x))

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3.5, clip_value_max=3.5)

    def idm_sim(self, env_states, encoder_states):
        batch_size = 256
        h_t, c_t = encoder_states
        scaled_s, unscaled_s = env_states
        #
        desired_v = self.get_des_v(h_t, unscaled_s[:, 0, 2:3])
        desired_tgap = self.get_des_tgap(h_t)
        min_jamx = self.get_min_jamx(h_t)
        max_act = self.get_max_act(h_t)
        min_act = self.get_min_act(h_t)

        idm_param = [desired_v, desired_tgap, min_jamx, max_act, min_act]

        if self.model_use == 'training' or self.model_use == 'debug':
            act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            fl_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            fm_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
            alphas = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

            for step in tf.range(20):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (alphas, tf.TensorShape([None,None,None])),
                                (fl_seq, tf.TensorShape([None,None,None])),
                                (fm_seq, tf.TensorShape([None,None,None])),
                                (act_seq, tf.TensorShape([None,None,None]))
                                 ])


                # alpha, fm_alpha = self.get_att_score(tf.concat([fl_act, fm_act, outputs], axis=1))
                vel = tf.slice(unscaled_s, [0, step, 2], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])

                dv = tf.slice(unscaled_s, [0, step, 4], [batch_size, 1, 1])
                dx = tf.slice(unscaled_s, [0, step, 5], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fl_act = self.idm_driver(vel, dv, dx, idm_param)

                dv = tf.slice(unscaled_s, [0, step, 7], [batch_size, 1, 1])
                dx = tf.slice(unscaled_s, [0, step, 8], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fm_act = self.idm_driver(vel, dv, dx, idm_param)

                outputs, h_t, c_t = self.future_dec(scaled_s[:, step:step+1, :], initial_state=[h_t, c_t])
                outputs = tf.reshape(outputs, [batch_size, self.enc_units])
                alpha = self.get_att_score(outputs)

                act = alpha*fl_act + (1-alpha)*fm_act
                # act = alpha*fl_act + (1-alpha)*fm_act
                # act = alpha*fl_act
                # act = fl_act
                alphas = tf.concat([alphas, tf.reshape(alpha, [batch_size, 1, 1])], axis=1)
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)
                fl_seq = tf.concat([fl_seq, tf.reshape(fl_act, [batch_size, 1, 1])], axis=1)
                fm_seq = tf.concat([fm_seq, tf.reshape(fm_act, [batch_size, 1, 1])], axis=1)

            tf.print('######')
            tf.print('desired_v: ', tf.reduce_mean(desired_v))
            tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
            tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
            tf.print('max_act: ', tf.reduce_mean(max_act))
            tf.print('min_act: ', tf.reduce_mean(min_act))
            tf.print('alpha_max: ', tf.reduce_max(alphas))
            tf.print('alpha_min: ', tf.reduce_min(alphas))
            tf.print('alpha_mean: ', tf.reduce_mean(alphas))

            # tf.print('fm_alpha: ', tf.reduce_max(fm_alphas))
            # tf.print('sum: ', tf.reduce_mean(alphas+fm_alphas))

            return act_seq
            # return act_seq, idm_param

        elif self.model_use == 'inference':
            outputs, h_t, c_t = self.future_dec(scaled_s[:, 0:1, :], initial_state=[h_t, c_t])
            outputs = tf.reshape(outputs, [1, self.enc_units])
            alpha = self.get_att_score(outputs)
            return idm_param, alpha

    def call(self, inputs):
        _, h_t, c_t = self.histroy_enc(inputs[0])
        if self.model_use == 'training' or self.model_use == 'debug':
            action = self.idm_sim(inputs[1:], [h_t, c_t])
            return action

        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1:], [h_t, c_t])
            return idm_param
