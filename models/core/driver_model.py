from tensorflow.keras.layers import Dense, LSTM
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
tf.random.set_seed(1234)

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
        return self.action_clip(act)

    def get_att_score(self, x):
        x = self.attention_layer(x)
        x = self.attention_neu(x)

        return 1/(1+tf.exp(-1*x))

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
