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

    def param_activation(self, batch_size, x, min_val, max_val):
        activation_function = tf.tanh(x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1, activation=K.exp)
        self.neu_min_jamx = Dense(1)
        self.neu_max_act = Dense(1, activation=K.exp)
        self.neu_min_act = Dense(1, activation=K.exp)

    def idm_sim(self, state, h_t):
        # state: [v, dv, dx]
        # print(state.shape)
        if self.model_use == 'training':
            batch_size = 256
        elif self.model_use == 'inference':
            batch_size = 1

        desired_v = self.param_activation(batch_size, self.neu_desired_v(h_t), 15., 35.)
        desired_tgap = self.neu_desired_tgap(h_t)
        min_jamx = tf.abs(self.param_activation(batch_size, self.neu_min_jamx(h_t), -5., 5.))
        max_act = self.neu_max_act(h_t)
        min_act = self.neu_min_act(h_t)

        if self.model_use == 'training':
            # state: [follower_v, lead_v, dv, dx]
            vel = tf.slice(state, [0, 0], [batch_size, 1])
            dv = tf.slice(state, [0, 2], [batch_size, 1])
            dx = tf.slice(state, [0, 3], [batch_size, 1])
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
            return act

        elif self.model_use == 'inference':
            idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)
            return idm_param

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs[0])
        if self.model_use == 'training':
            action = self.idm_sim(inputs[1], h_t)
            return action
        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1], h_t)
            return idm_param
