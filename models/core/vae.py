from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
import numpy as np
# %%
# tfd = tfp.distributions
# dist1 = tfd.Normal(loc=0., scale=3.)
# dist2 = tfd.Normal(loc=[0, 10.], scale=[3., 3.])
# tfd.kl_divergence(
#     dist1, mvn, allow_nan_stats=True, name=None)
#
# %%
class VAEIDM(AbstractModel):
    def __init__(self, config, model_use):
        super(VAEIDM, self).__init__(config)
        self.encoder = Encoder()
        self.decoder = Decoder(config)
        # mse_loss = 0
        # self.kl_loss = 0

    def callback_def(self):
        self.train_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mseloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_mseloss = tf.keras.metrics.Mean(name='train_loss')

    # @tf.function
    def sample(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0], self.encoder.latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_actions(self):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        desired_v = self.decoder(z)
        a_ = self.idm(desired_v)
        return  a_

    def kl_loss(self, z_mean, z_log_sigma):
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(kl_loss)

    def vae_loss(self, mse_loss, kl_loss):
        return  kl_loss +  mse_loss

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, mean, logvar = self(states)
            mse_loss = self.mse(targets, act_pred)
            kl_loss = self.kl_loss(mean, logvar)
            loss = self.vae_loss(mse_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_mseloss.reset_states()
        self.train_klloss.reset_states()
        self.train_mseloss(mse_loss)
        self.train_klloss(kl_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, mean, logvar = self(states)
        mse_loss = self.mse(targets, act_pred)
        kl_loss = self.kl_loss(mean, logvar)
        loss = self.vae_loss(mse_loss, kl_loss)
        self.test_klloss.reset_states()
        self.test_mseloss.reset_states()
        self.test_mseloss(mse_loss)
        self.test_klloss(kl_loss)

    def call(self, inputs):
        mean, logvar = self.encoder(inputs[0])
        z = self.sample([mean, logvar])
        idm_param = self.decoder(z)
        a_ = self.decoder.idm_sim(inputs[1], idm_param)
        return a_, mean, logvar

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.model_use = 'training'
        self.architecture_def()

    def architecture_def(self):
        self.layer_1 = Dense(10, activation=K.relu)
        self.layer_2 = Dense(60, activation=K.relu)
        self.layer_3 = Dense(60, activation=K.relu)
        self.layer_4 = Dense(10, activation=K.relu)
        self.lstm_attention = LSTM(10, return_sequences=True, return_state=True)
        self.neu_attention_1 = TimeDistributed(Dense(14))
        self.neu_attention_2 = TimeDistributed(Dense(1, K.softmax))

        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1, activation=K.exp)
        self.neu_min_jamx = Dense(1)
        self.neu_max_act = Dense(1, activation=K.exp)
        self.neu_min_act = Dense(1, activation=K.exp)

    def attention(self, dy):
        x = self.neu_attention_1(dy)
        return self.neu_attention_2(x)

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

    def idm_sim(self, state, idm_param):
        # state: [v, dv, dx]
        batch_size = tf.shape(state)[0]
        act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        dy = tf.slice(state, [0, 0, 7], [batch_size, 1, 1])
        # dy = tf.reshape(dy, [batch_size, 1])

        # outputs, lstm_h, lstm_c = self.lstm_attention(dy)
        alpha = self.attention(dy)

        for step in tf.range(20):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (act_seq, tf.TensorShape([None,None,None])),
                            (dy, tf.TensorShape([None,None,None])),
                             (alpha, tf.TensorShape([None,None,None]))])

            dy = tf.slice(state, [0, step, 7], [batch_size, 1, 1])
            vel = tf.slice(state, [0, step, 0], [batch_size, 1, 1])
            dv = tf.slice(state, [0, step, 2], [batch_size, 1, 1])
            dx = tf.slice(state, [0, step, 3], [batch_size, 1, 1])

            alpha = 0.5
            # alpha = tf.reshape(alpha, [batch_size, 1])
            # dy = tf.reshape(dy, [batch_size, 1])
            vel = tf.reshape(vel, [batch_size, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fl_act = self.idm(vel, dv, dx, idm_param)

            dv = tf.slice(state, [0, step, 5], [batch_size, 1, 1])
            dx = tf.slice(state, [0, step, 6], [batch_size, 1, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fm_act = self.idm(vel, dv, dx, idm_param)

            act = (1-alpha)*fl_act + (alpha)*fm_act
            act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)
            #
            # outputs, lstm_h, lstm_c = self.lstm_attention(dy, \
            #                         initial_state=[lstm_h, lstm_c])
            # alpha = self.attention(dy)
        # tf.print(tf.reduce_min(alpha))
        tf.print(tf.reduce_mean(dy))
        return act_seq

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        # outputs, lstm_h, lstm_c = self.lstm_attention(x)
        # alpha = self.neu_attention(outputs)

        batch_size = tf.shape(inputs)[0]
        idm_param = self.compute_idm_param(x, batch_size)

        return idm_param

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = 50
        self.latent_dim = 2
        # TODO: FULL cov distribution
        # self.prior = tfd.Normal(loc=0., scale=3.)
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.z_mean = Dense(self.latent_dim)
        self.z_log_sigma = Dense(self.latent_dim)
        #
        # tfpl.MultivariateNormalTriL(self.latent_dim,
        #     activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=1.0)),

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs)
        z_mean = self.z_mean(h_t)
        z_log_sigma = self.z_log_sigma(h_t)
        return [z_mean, z_log_sigma]

        # if self.model_use == 'training' or self.model_use == 'debug':
        #     action = self.idm_sim(inputs[1], h_t)
        #     return action
        #
        # elif self.model_use == 'inference':
        #     idm_param = self.idm_sim(inputs[1], h_t)
        #     return idm_param
