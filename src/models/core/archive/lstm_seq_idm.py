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
        if self.model_use == 'training':
            batch_size = 256
        elif self.model_use == 'inference' or self.model_use == 'debug':
            batch_size = 1

        desired_v = self.param_activation(batch_size, self.neu_desired_v(h_t), 15., 35.)
        min_jamx = tf.abs(self.param_activation(batch_size, self.neu_min_jamx(h_t), -5., 5.))
        desired_tgap = self.neu_desired_tgap(h_t)
        max_act = self.neu_max_act(h_t)
        min_act = self.neu_min_act(h_t)

        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)
        # tf.print('desired_v: ', tf.reduce_mean(desired_v))
        # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        # tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
        # tf.print('max_act: ', tf.reduce_mean(max_act))
        # tf.print('min_act: ', tf.reduce_mean(min_act))

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

                desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                                (2*tf.sqrt(max_act*min_act))

                act = max_act*(1-(vel/desired_v)**4-\
                                                    (desired_gap/dx)**2)
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)

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
