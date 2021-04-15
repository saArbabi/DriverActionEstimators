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

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1, activation=K.exp)
        self.neu_min_jamx = Dense(1)
        self.neu_max_act = Dense(1, activation=K.exp)
        self.neu_min_act = Dense(1, activation=K.exp)
        self.neu_attention = Dense(1, K.sigmoid)
        # self.neu_attention = TimeDistributed(Dense(1, K.softmax))

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

    def idm_sim(self, state, h_t):
        # state: [v, dv, dx]
        batch_size = tf.shape(state)[0]
        idm_param = self.compute_idm_param(h_t, batch_size)

        if self.model_use == 'training' or self.model_use == 'debug':
            act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

            for step in tf.range(20):
                # tf.autograph.experimental.set_loop_options(shape_invariants=[
                #                 (act_seq, tf.TensorShape([None,None,None])),
                #                 (dy, tf.TensorShape([None,None,None])),
                #                  (alpha, tf.TensorShape([None,None,None]))])
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (act_seq, tf.TensorShape([None,None,None]))])

                dy = tf.slice(state, [0, step, 7], [batch_size, 1, 1])
                vel = tf.slice(state, [0, step, 0], [batch_size, 1, 1])
                dv = tf.slice(state, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 3], [batch_size, 1, 1])

                # alpha = 0.5

                dy = tf.reshape(dy, [batch_size, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fl_act = self.idm(vel, dv, dx, idm_param)

                dv = tf.slice(state, [0, step, 5], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 6], [batch_size, 1, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])
                fm_act = self.idm(vel, dv, dx, idm_param)

                alpha = self.neu_attention(dy)
                # alpha = self.neu_attention(tf.concat([dy, h_t], axis=1))

                # alpha = 1
                # alpha = tf.reshape(alpha, [batch_size, 1])
                act = (1-alpha)*fl_act + (alpha)*fm_act
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)
                #
                # tf.print(alpha)
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
