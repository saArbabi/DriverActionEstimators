from tensorflow.keras.layers import Dense, LSTM
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM
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

    def param_activation(self, batch_size, x, min_val, max_val):
        activation_function = tf.tanh(x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    # @tf.function
    def sample(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0], self.encoder.latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        des_v = self.decoder(z)
        des_v = self.param_activation(tf.shape(z)[0], des_v, 15., 35.)
        return des_v

    def get_actions(self):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        desired_v = self.decode(z)
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
        mean, logvar = self.encode(inputs[0])
        z = self.sample([mean, logvar])
        des_v = self.decode(z)
        a_ = self.decoder.idm_sim(inputs[1], des_v)
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
        self.layer_out = Dense(1)

    def idm_sim(self, state, desired_v):
        # state: [v, dv, dx]
        # if self.model_use == 'training':
        #     batch_size = 256
        # elif self.model_use == 'inference' or self.model_use == 'debug':
        batch_size = tf.shape(desired_v)[0]

        # desired_v = self.param_activation(batch_size, self.neu_desired_v(h_t), 15., 35.)
        desired_tgap = tf.fill([batch_size, 1], 1.5)
        min_jamx = tf.fill([batch_size, 1], 2.)
        max_act = tf.fill([batch_size, 1], 1.4)
        min_act = tf.fill([batch_size, 1], 2.)

        # idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)

        if self.model_use == 'training' or self.model_use == 'debug':
            act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

            for step in tf.range(20):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (act_seq, tf.TensorShape([None,None,None]))])

                vel = tf.slice(state, [0, step, 0], [batch_size, 1, 1])
                dv = tf.slice(state, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 3], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])

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
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)

            return act_seq
            # return act_seq, idm_param

        elif self.model_use == 'inference':
            return idm_param

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return self.layer_out(x)

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
