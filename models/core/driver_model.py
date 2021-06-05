from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(1234)

class NeurIDMModel(AbstractModel):
    def __init__(self, config, model_use):
        super(NeurIDMModel, self).__init__(config)
        self.future_state_enc = FutureEncoder()
        self.history_state_enc = HistoryEncoder()
        self.future_action_enc = FutureEncoder() # sdv's future action
        self.belief_estimator = BeliefModel()
        self.decoder = Decoder(config)
        self.idm_layer = IDMLayer()
        self.idm_sim = IDMForwardSim()
        self.model_use = model_use

    def callback_def(self):
        self.train_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mseloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_mseloss = tf.keras.metrics.Mean(name='train_loss')

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, merger_xas, ys_f in train_ds:
            self.train_step([xs_h, scaled_xs_f, unscaled_xs_f, merger_xas], ys_f)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, merger_xas, ys_f in train_ds:
            self.test_step([xs_h, scaled_xs_f, unscaled_xs_f, merger_xas], ys_f)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, prior_param, posterior_param = self(states)
            mse_loss = self.mse(targets, act_pred)
            kl_loss = self.kl_loss(prior_param, posterior_param)
            loss = self.vae_loss(mse_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_mseloss.reset_states()
        self.train_klloss.reset_states()
        self.train_mseloss(mse_loss)
        self.train_klloss(kl_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, prior_param, posterior_param = self(states)
        mse_loss = self.mse(targets, act_pred)
        kl_loss = self.kl_loss(prior_param, posterior_param)
        loss = self.vae_loss(mse_loss, kl_loss)
        self.test_klloss.reset_states()
        self.test_mseloss.reset_states()
        self.test_mseloss(mse_loss)
        self.test_klloss(kl_loss)

    def kl_loss(self, prior_param, posterior_param):
        prior = tfd.Normal(loc=prior_param[0], scale=tf.exp(prior_param[1]))
        posterior = tfd.Normal(loc=posterior_param[0], scale=tf.exp(posterior_param[1]))
        return tf.reduce_mean(tfp.distributions.kl_divergence(posterior, prior))

    def vae_loss(self, mse_loss, kl_loss):
        return  0.1*kl_loss +  mse_loss

    def call(self, inputs):
        # inputs: [xs_h, scaled_xs_f, unscaled_xs_f, merger_xas]
        h_enc_state = self.history_state_enc(inputs[0]) # history lstm state
        f_enc_action = self.future_action_enc(inputs[-1])

        if self.model_use == 'training':
            f_enc_state = self.future_state_enc(inputs[1])
            prior_param, posterior_param = self.belief_estimator(\
                                    [h_enc_state[0], f_enc_state[0], f_enc_action[0]], dis_type='both')
            z = self.belief_estimator.sample_z(posterior_param)
            context = tf.concat([z, h_enc_state[0]], axis=1)
            decoder_output = self.decoder(context)
            idm_param = self.idm_layer(decoder_output)
            act_seq, _ = self.idm_sim.rollout([inputs[2], z, idm_param, h_enc_state])
            return act_seq, prior_param, posterior_param

        elif self.model_use == 'inference':
            h_t, c_t = h_enc_state
            prior_param = self.belief_estimator(h_t, dis_type='prior')
            z = self.belief_estimator.sample_z(prior_param)
            decoder_output = self.decoder(z)
            idm_param = self.idm_layer([decoder_output, current_v])
            # att_score, _, _ = self.idm_sim.arbiter([inputs[1][:, 0:1, :], h_t, c_t])
            return idm_param
            # return idm_param, att_score

class BeliefModel(tf.keras.Model):
    def __init__(self):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.latent_dim = 2
        self.architecture_def()

    def architecture_def(self):
        self.pri_mean = Dense(self.latent_dim)
        self.pri_logsigma = Dense(self.latent_dim)
        self.pri_linear_layer = Dense(50)
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)
        self.pos_linear_layer = Dense(100)

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        return z_mean + K.exp(z_logsigma) * epsilon

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            ht_history, ht_future, ht_action = inputs
            # prior
            context = self.pri_linear_layer(tf.concat([ht_history, ht_action], axis=-1))
            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)
            # posterior
            context = self.pos_linear_layer(tf.concat([ht_history, ht_future, ht_action], axis=-1))
            pos_mean = self.pos_mean(context)
            pos_logsigma = self.pos_logsigma(context)
            return [pri_mean, pri_logsigma], [pos_mean, pos_logsigma]

        elif dis_type == 'prior':
            ht_history, ht_action = inputs
            context = self.pri_linear_layer(tf.concat([ht_history, ht_action], axis=-1))
            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)
            return [pri_mean, pri_logsigma]

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs)
        return [h_t, c_t]

class FutureEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureEncoder, self).__init__(name="FutureEncoder")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = Bidirectional(LSTM(self.enc_units))

    def call(self, inputs):
        h_t = self.lstm_layer(inputs)
        return [h_t, 0]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.architecture_def()

    def architecture_def(self):
        self.layer_1 = Dense(10, activation=K.relu)
        self.layer_2 = Dense(60, activation=K.relu)
        self.layer_3 = Dense(60, activation=K.relu)
        self.layer_4 = Dense(10, activation=K.relu)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_4(x)

class Arbiter(tf.keras.Model):
    def __init__(self):
        super(Arbiter, self).__init__(name="Arbiter")
        self.enc_units = 50
        self.attention_temp = 5 # the higher, the sharper the attention
        self.architecture_def()

    def architecture_def(self):
        self.future_dec = LSTM(self.enc_units, return_sequences=True, return_state=True)
        self.attention_layer = Dense(self.enc_units)
        self.attention_neu = Dense(1)

    def call(self, inputs):
        scaled_s, h_t, c_t = inputs
        batch_size = tf.shape(scaled_s)[0]
        outputs, h_t, c_t = self.future_dec(scaled_s, initial_state=[h_t, c_t])
        outputs = tf.reshape(outputs, [batch_size, self.enc_units])
        x = self.attention_layer(outputs)
        x = self.attention_neu(x)

        return 1/(1+tf.exp(-self.attention_temp*x)), h_t, c_t

class IDMForwardSim(tf.keras.Model):
    def __init__(self):
        super(IDMForwardSim, self).__init__(name="IDMForwardSim")
        self.arbiter = Arbiter()

    def idm_driver(self, vel, dv, dx, idm_param):
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param
        desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)
        return self.action_clip(act)

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3.5, clip_value_max=3.5)

    def rollout(self, inputs):
        unscaled_s, z, idm_param, encoder_states = inputs
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param

        h_t, c_t = encoder_states
        batch_size = tf.shape(unscaled_s)[0]

        act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        fl_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        fm_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        att_scores = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

        for step in tf.range(20):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (att_scores, tf.TensorShape([None,None,None])),
                            (fl_seq, tf.TensorShape([None,None,None])),
                            (fm_seq, tf.TensorShape([None,None,None])),
                            (act_seq, tf.TensorShape([None,None,None]))
                             ])


            vel = tf.slice(unscaled_s, [0, step, 0], [batch_size, 1, 1])
            vel = tf.reshape(vel, [batch_size, 1])

            dv = tf.slice(unscaled_s, [0, step, 2], [batch_size, 1, 1])
            dx = tf.slice(unscaled_s, [0, step, 3], [batch_size, 1, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fl_act = self.idm_driver(vel, dv, dx, idm_param)

            dv = tf.slice(unscaled_s, [0, step, 5], [batch_size, 1, 1])
            dx = tf.slice(unscaled_s, [0, step, 6], [batch_size, 1, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fm_act = self.idm_driver(vel, dv, dx, idm_param)

            att_score, h_t, c_t = self.arbiter([tf.reshape(z, [batch_size, 1, 2]), h_t, c_t])
            act = att_score*fl_act + (1-att_score)*fm_act

            att_scores = tf.concat([att_scores, tf.reshape(att_score, [batch_size, 1, 1])], axis=1)
            act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)
            fl_seq = tf.concat([fl_seq, tf.reshape(fl_act, [batch_size, 1, 1])], axis=1)
            fm_seq = tf.concat([fm_seq, tf.reshape(fm_act, [batch_size, 1, 1])], axis=1)

        tf.print('######')
        tf.print('desired_v_mean: ', tf.reduce_mean(desired_v))
        tf.print('desired_v_max: ', tf.reduce_max(desired_v))
        tf.print('desired_v_min: ', tf.reduce_min(desired_v))
        tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
        tf.print('max_act: ', tf.reduce_mean(max_act))
        tf.print('min_act: ', tf.reduce_mean(min_act))
        tf.print('att_score_max: ', tf.reduce_max(att_scores))
        tf.print('att_score_min: ', tf.reduce_min(att_scores))
        tf.print('att_score_mean: ', tf.reduce_mean(att_scores))

        return act_seq, att_scores

class IDMLayer(tf.keras.Model):
    def __init__(self):
        super(IDMLayer, self).__init__(name="IDMLayer")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
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

    def get_des_v(self, x):
        x = self.des_v_layer(x)
        output = self.des_v_neu(x) + 20
        return output

    def get_des_tgap(self, x):
        x = self.des_tgap_layer(x)
        output = tf.abs(self.des_tgap_neu(x)) + 1
        return output

    def get_min_jamx(self, x):
        x = self.min_jamx_layer(x)
        output = tf.abs(self.min_jamx_neu(x))
        return output

    def get_max_act(self, x):
        x = self.max_act_layer(x)
        output = tf.abs(self.max_act_neu(x)) + 0.5
        return output

    def get_min_act(self, x):
        x = self.min_act_layer(x)
        output = tf.abs(self.min_act_neu(x)) + 0.5
        return output

    def call(self, inputs):
        # _, h_t, c_t = self.histroy_enc(inputs[0])
        decoder_output = inputs
        desired_v = self.get_des_v(decoder_output)
        desired_tgap = self.get_des_tgap(decoder_output)
        min_jamx = self.get_min_jamx(decoder_output)
        max_act = self.get_max_act(decoder_output)
        min_act = self.get_min_act(decoder_output)
        idm_param = [desired_v, desired_tgap, min_jamx, max_act, min_act]
        return idm_param
