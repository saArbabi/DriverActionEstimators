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
