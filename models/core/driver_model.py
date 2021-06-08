from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed
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
        self.arbiter = Arbiter()
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.act_encoder = FutureEncoder() # sdv's future action
        self.belief_net = BeliefModel()
        self.idm_layer = IDMLayer()
        self.idm_sim = IDMForwardSim()
        self.model_use = model_use
        # self.vae_loss_weight = 0.01

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
        return  self.vae_loss_weight*kl_loss + (1-self.vae_loss_weight)*mse_loss

    def call(self, inputs):
        # inputs: [xs_h, scaled_xs_f, unscaled_xs_f, merger_xas]
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f_acts = self.act_encoder(inputs[-1])

        if self.model_use == 'training':
            enc_f = self.f_seq_encoder(inputs[1])
            prior_param, posterior_param = self.belief_net(\
                                    [enc_h, enc_f_acts, enc_f], dis_type='both')
            sampled_z = self.belief_net.sample_z(posterior_param)
            att_scores = self.arbiter([sampled_z, enc_h, enc_f_acts])
            # att_scores = self.arbiter(sampled_z)


            # idm_param = self.idm_layer(enc_h)
            batch_size = tf.shape(sampled_z)[0]
            # idm_params = tf.repeat(tf.constant([[25, 1.5, 2, 1.4, 2]]), batch_size, axis=0)

            idm_params = tf.repeat(tf.constant([[25, 1.5, 2, 1.4, 2]]), 40, axis=0)
            idm_params = tf.reshape(idm_params, [1, 40, 5])
            idm_params = tf.repeat(idm_params, batch_size, axis=0)

            act_seq = self.idm_sim.rollout([att_scores, idm_params, inputs[2]])

            tf.print('att_score_max: ', tf.reduce_max(att_scores))
            tf.print('att_score_min: ', tf.reduce_min(att_scores))
            tf.print('att_score_mean: ', tf.reduce_mean(att_scores))

            return act_seq, prior_param, posterior_param

        elif self.model_use == 'inference':
            h_t, c_t = h_enc_state
            prior_param = self.belief_net(h_t, dis_type='prior')
            z = self.belief_net.sample_z(prior_param)
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
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)

        self.pri_encoding_layer_1 = Dense(100)
        # self.pri_encoding_layer_2 = Dense(100, activation=K.relu)
        # self.pri_encoding_layer_3 = Dense(100, activation=K.relu)

        self.pos_encoding_layer_1 = Dense(100)
        # self.pos_encoding_layer_2 = Dense(100, activation=K.relu)
        # self.pos_encoding_layer_3 = Dense(100, activation=K.relu)

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        return z_mean + K.exp(z_logsigma) * epsilon

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_f_acts, enc_f = inputs
            # prior
            context = self.pri_encoding_layer_1(enc_h+enc_f_acts)
            # context = self.pri_encoding_layer_2(context)
            # context = self.pri_encoding_layer_3(context)

            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)

            # posterior
            context = self.pos_encoding_layer_1(enc_h+enc_f_acts+enc_f)
            # context = self.pos_encoding_layer_2(context)
            # context = self.pos_encoding_layer_3(context)

            pos_mean = self.pos_mean(context)
            pos_logsigma = self.pos_logsigma(context)
            return [pri_mean, pri_logsigma], [pos_mean, pos_logsigma]

        elif dis_type == 'prior':
            enc_h, enc_f_acts = inputs
            context = self.pri_encoding_layer_1(enc_h+enc_f_acts)
            # context = self.pri_encoding_layer_2(context)
            # context = self.pri_encoding_layer_3(context)

            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)
            return [pri_mean, pri_logsigma]

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units)

    def call(self, inputs):
        enc_h = self.lstm_layer(inputs)
        return enc_h

class FutureEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureEncoder, self).__init__(name="FutureEncoder")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = Bidirectional(LSTM(self.enc_units), merge_mode='mul')

    def call(self, inputs):
        enc_f_acts = self.lstm_layer(inputs)
        return enc_f_acts

class Arbiter(tf.keras.Model):
    def __init__(self):
        super(Arbiter, self).__init__(name="Arbiter")
        self.enc_units = 50
        self.attention_temp = 20 # the higher, the sharper the attention
        self.architecture_def()

    def architecture_def(self):
        # self.future_dec = LSTM(self.enc_units, return_sequences=True, return_state=True)
        # self.context_layer = TimeDistributed(Dense(100))
        self.attention_layer_1 = Dense(50, activation=K.relu)
        self.attention_layer_2 = Dense(50, activation=K.relu)
        self.attention_layer_3 = Dense(50, activation=K.relu)
        self.attention_layer_4 = Dense(50, activation=K.relu)
        self.attention_neu = Dense(40)

    def call(self, inputs):
        sampled_z, enc_h, enc_f_acts = inputs
        # att_context, h_t, c_t = inputs
        # batch_size = tf.shape(att_context)[0]
        # att_context = self.context_layer(att_context)
        # outputs, h_t, c_t = self.future_dec(att_context, initial_state=[h_t, c_t])
        # outputs = tf.reshape(outputs, [batch_size, self.enc_units])
        # x = self.attention_layer(outputs)

        x = self.attention_layer_1(sampled_z)
        x = self.attention_layer_2(x)
        x = self.attention_layer_3(x)
        x = self.attention_layer_4(x)
        # x = self.attention_layer_2(x+enc_h+enc_f_acts)
        x = self.attention_neu(x)
        # return x
        return 1/(1+tf.exp(-self.attention_temp*x))

class IDMForwardSim(tf.keras.Model):
    def __init__(self):
        super(IDMForwardSim, self).__init__(name="IDMForwardSim")

    def idm_driver(self, vel, dv, dx, idm_params):
        # desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param

        desired_v = idm_params[:,:,0:1]
        desired_tgap = idm_params[:,:,1:2]
        min_jamx = idm_params[:,:,2:3]
        max_act = idm_params[:,:,3:4]
        min_act = idm_params[:,:,4:5]

        desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)
        return self.action_clip(act)

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3.5, clip_value_max=3.5)

    def rollout(self, inputs):
        att_scores, idm_params, unscaled_s = inputs
        batch_size = tf.shape(unscaled_s)[0]

        # get idm actions
        vel = tf.slice(unscaled_s, [0, 0, 0], [batch_size, 40, 1])

        dv = tf.slice(unscaled_s, [0, 0, 2], [batch_size, 40, 1])
        dx = tf.slice(unscaled_s, [0, 0, 3], [batch_size, 40, 1])
        fl_act = self.idm_driver(vel, dv, dx, idm_params)

        dv = tf.slice(unscaled_s, [0, 0, 5], [batch_size, 40, 1])
        dx = tf.slice(unscaled_s, [0, 0, 6], [batch_size, 40, 1])
        fm_act = self.idm_driver(vel, dv, dx, idm_params)

        att_scores = tf.reshape(att_scores, [batch_size, 40, 1])
        act_seq = att_scores*fl_act + (1-att_scores)*fm_act

        return act_seq

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
