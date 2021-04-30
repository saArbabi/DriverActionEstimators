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

    def get_des_v(self, h_t, current_v):
        input = self.des_v_layer(h_t)
        output = self.des_v_neu(input) + current_v
        return output

    def get_des_tgap(self, h_t):
        input = self.des_tgap_layer(h_t)
        output = tf.abs(self.des_tgap_neu(input)) + 1
        return output

    def get_min_jamx(self, h_t):
        input = self.min_jamx_layer(h_t)
        output = tf.abs(self.min_jamx_neu(input)+1)
        return output

    def get_max_act(self, h_t):
        input = self.max_act_layer(h_t)
        output = tf.abs(self.max_act_neu(input)) + 0.5
        return output

    def get_min_act(self, h_t):
        input = self.min_act_layer(h_t)
        output = tf.abs(self.min_act_neu(input)) + 0.5
        return output

    def idm_sim(self, env_states, encoder_states):
        # env_states: [v, dv, dx]
        batch_size = tf.shape(env_states)[0]
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
            desired_v = tf.zeros([batch_size, 1], dtype=tf.float32)
            desired_tgap = tf.zeros([batch_size, 1], dtype=tf.float32)
            min_jamx = tf.zeros([batch_size, 1], dtype=tf.float32)
            max_act = tf.zeros([batch_size, 1], dtype=tf.float32)
            min_act = tf.zeros([batch_size, 1], dtype=tf.float32)

            for step in tf.range(20):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (desired_v, tf.TensorShape([None,None])),
                                (desired_tgap, tf.TensorShape([None,None])),
                                (min_jamx, tf.TensorShape([None,None])),
                                (max_act, tf.TensorShape([None,None])),
                                (min_act, tf.TensorShape([None,None])),
                                (act_seq, tf.TensorShape([None,None,None]))])

                s = env_states[:, step:step+1, :]
                vel = tf.slice(env_states, [0, step, 0], [batch_size, 1, 1])
                dv = tf.slice(env_states, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(env_states, [0, step, 3], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])

                outputs, h_t, c_t = self.future_dec(s, initial_state=[h_t, c_t])
                # tf.print(tf.shape(outputs))
                outputs = tf.reshape(outputs, [batch_size, self.enc_units])
                desired_v = self.get_des_v(outputs, s[:, 0, 0:1])
                desired_tgap = self.get_des_tgap(outputs)
                min_jamx = self.get_min_jamx(outputs)
                max_act = self.get_max_act(outputs)
                min_act = self.get_min_act(outputs)

                # desired_v = self.get_des_v(h_t, vel)
                # desired_tgap = self.get_des_tgap(h_t)
                # min_jamx = self.get_min_jamx(h_t)
                # max_act = self.get_max_act(h_t)
                # min_act = self.get_min_act(h_t)

                desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                                (2*tf.sqrt(max_act*min_act))

                act = max_act*(1-(vel/desired_v)**4-\
                                                    (desired_gap/dx)**2)

                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)

            # tf.print('desired_v: ', tf.reduce_mean(desired_v))
            # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
            # tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
            # tf.print('max_act: ', tf.reduce_mean(max_act))
            # tf.print('min_act: ', tf.reduce_mean(min_act))

            return act_seq
            # return act_seq, idm_param

        elif self.model_use == 'inference':
            s = env_states[:, 0:1, :]
            vel = tf.slice(env_states, [0, 0, 0], [batch_size, 1, 1])
            dv = tf.slice(env_states, [0, 0, 2], [batch_size, 1, 1])
            dx = tf.slice(env_states, [0, 0, 3], [batch_size, 1, 1])
            vel = tf.reshape(vel, [batch_size, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])

            outputs, h_t, c_t = self.future_dec(s, initial_state=[h_t, c_t])
            # tf.print(tf.shape(outputs))
            outputs = tf.reshape(outputs, [batch_size, self.enc_units])
            desired_v = self.get_des_v(outputs, s[:, 0, 0:1])
            desired_tgap = self.get_des_tgap(outputs)
            min_jamx = self.get_min_jamx(outputs)
            max_act = self.get_max_act(outputs)
            min_act = self.get_min_act(outputs)

            idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)

            return idm_param

    def call(self, inputs):
        _, h_t, c_t = self.histroy_enc(inputs[0])
        if self.model_use == 'training' or self.model_use == 'debug':
            action = self.idm_sim(inputs[1], [h_t, c_t])
            return action

        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1], [h_t, c_t])
            return idm_param
