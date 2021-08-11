from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Masking
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
    def __init__(self, config=None):
        super(NeurIDMModel, self).__init__(config)
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.act_encoder = FutureEncoder() # sdv's future action
        self.belief_net = BeliefModel()
        self.idm_layer = IDMLayer()
        self.idm_sim = IDMForwardSim()
        self.vae_loss_weight = 0.1 # default

    def callback_def(self):
        self.train_mseloss = tf.keras.metrics.Mean()
        self.test_mseloss = tf.keras.metrics.Mean()
        self.train_att_klloss = tf.keras.metrics.Mean()
        self.test_att_klloss = tf.keras.metrics.Mean()
        self.train_idm_klloss = tf.keras.metrics.Mean()
        self.test_idm_klloss = tf.keras.metrics.Mean()

    def mse(self, act_true, act_pred):
        act_true = (act_true)/0.1
        # act_true += tf.random.normal(shape=(256, 40, 1), mean=0, stddev=0.6)
        act_pred = (act_pred)/0.1
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

    def train_loop(self, data_objs):
        # tf.print('######## TRAIN #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, future_sca, future_idm_s, future_m_veh_a, future_ego_a in train_ds:
            self.train_step([history_sca, future_sca, future_idm_s, future_m_veh_a], future_ego_a)

    def test_loop(self, data_objs, epoch):
        # tf.print('######## TEST #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, future_sca, future_idm_s, future_m_veh_a, future_ego_a in train_ds:
            self.test_step([history_sca, future_sca, future_idm_s, future_m_veh_a], future_ego_a)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, pri_params, pos_params = self(states)
            mse_loss = self.mse(targets, act_pred)
            kl_att, kl_idm = self.kl_loss(pri_params, pos_params)
            kl_loss = kl_att + kl_idm
            loss = self.vae_loss(mse_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_mseloss.reset_states()
        self.train_att_klloss.reset_states()
        self.train_idm_klloss.reset_states()
        self.train_mseloss(mse_loss)
        self.train_att_klloss(kl_att)
        self.train_idm_klloss(kl_idm)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, pri_params, pos_params = self(states)
        mse_loss = self.mse(targets, act_pred)
        kl_att, kl_idm = self.kl_loss(pri_params, pos_params)
        kl_loss = kl_att + kl_idm
        loss = self.vae_loss(mse_loss, kl_loss)
        self.test_mseloss.reset_states()
        self.test_att_klloss.reset_states()
        self.test_idm_klloss.reset_states()
        self.test_mseloss(mse_loss)
        self.test_att_klloss(kl_att)
        self.test_idm_klloss(kl_idm)

    def kl_loss(self, pri_params, pos_params):
        pri_att_mean, pri_idm_mean, pri_att_logsigma, pri_idm_logsigma = pri_params
        pos_att_mean, pos_idm_mean, pos_att_logsigma, pos_idm_logsigma = pos_params

        prior_att = tfd.Normal(loc=pri_att_mean, scale=tf.exp(pri_att_logsigma))
        posterior_att = tfd.Normal(loc=pos_att_mean, scale=tf.exp(pos_att_logsigma))
        kl_att = tf.reduce_mean(tfp.distributions.kl_divergence(posterior_att, prior_att))

        prior_idm = tfd.Normal(loc=pri_idm_mean, scale=tf.exp(pri_idm_logsigma))
        posterior_idm = tfd.Normal(loc=pos_idm_mean, scale=tf.exp(pos_idm_logsigma))
        kl_idm = tf.reduce_mean(tfp.distributions.kl_divergence(posterior_idm, prior_idm))

        return kl_att, kl_idm

    def vae_loss(self, mse_loss, kl_loss):
        return  self.vae_loss_weight*kl_loss + mse_loss

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f = self.f_seq_encoder(inputs[1])

        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')
        sampled_att_z, sampled_idm_z = self.belief_net.sample_z(pos_params)
        att_inputs = [sampled_att_z, enc_h]

        idm_params = self.idm_layer([sampled_idm_z, enc_h])
        # idm_params = tf.repeat(idm_params, 40, axis=1)

        act_seq, _ = self.idm_sim.rollout([att_inputs, idm_params, inputs[2], inputs[-1]])
        # tf.print('###############:')
        # tf.print('att_score_max: ', tf.reduce_max(att_scores))
        # tf.print('att_score_min: ', tf.reduce_min(att_scores))
        # tf.print('att_score_mean: ', tf.reduce_mean(att_scores))

        return act_seq, pri_params, pos_params

class BeliefModel(tf.keras.Model):
    def __init__(self):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.latent_dim = 2
        self.architecture_def()

    def architecture_def(self):
        self.pri_att_mean = Dense(self.latent_dim)
        self.pri_att_logsigma = Dense(self.latent_dim)
        self.pos_att_mean = Dense(self.latent_dim)
        self.pos_att_logsigma = Dense(self.latent_dim)

        self.pri_idm_mean = Dense(self.latent_dim)
        self.pri_idm_logsigma = Dense(self.latent_dim)
        self.pos_idm_mean = Dense(self.latent_dim)
        self.pos_idm_logsigma = Dense(self.latent_dim)

        self.pri_linear_att = Dense(100)
        self.pri_linear_idm = Dense(100)
        self.pos_linear_att = Dense(100)
        self.pos_linear_idm = Dense(100)

    def sample_z(self, dis_params):
        z_att_mean, z_idm_mean, z_att_logsigma, z_idm_logsigma = dis_params
        _epsilon = K.random_normal(shape=(tf.shape(z_att_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        sampled_att_z = z_att_mean + K.exp(z_att_logsigma) * _epsilon
        _epsilon = K.random_normal(shape=(tf.shape(z_att_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        sampled_idm_z = z_idm_mean + K.exp(z_idm_logsigma) * _epsilon

        return sampled_att_z, sampled_idm_z

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_f = inputs
            # prior
            context_att = self.pri_linear_att(enc_h)
            context_idm = self.pri_linear_idm(enc_h)
            pri_att_mean = self.pri_att_mean(context_att)
            pri_att_logsigma = self.pri_att_logsigma(context_att)
            pri_idm_mean = self.pri_idm_mean(context_idm)
            pri_idm_logsigma = self.pri_idm_logsigma(context_idm)

            # posterior
            pos_context = tf.concat([enc_h, enc_f], axis=-1)
            context_att = self.pos_linear_att(pos_context)
            context_idm = self.pos_linear_idm(pos_context)
            pos_att_mean = self.pos_att_mean(context_att)
            pos_att_logsigma = self.pos_att_logsigma(context_att)
            pos_idm_mean = self.pos_idm_mean(context_idm)
            pos_idm_logsigma = self.pos_idm_logsigma(context_idm)

            pri_params = [pri_att_mean, pri_idm_mean, pri_att_logsigma, pri_idm_logsigma]
            pos_params = [pos_att_mean, pos_idm_mean, pos_att_logsigma, pos_idm_logsigma]
            return pri_params, pos_params

        elif dis_type == 'prior':
            context_att = self.pri_linear_att(inputs)
            context_idm = self.pri_linear_idm(inputs)
            pri_att_mean = self.pri_att_mean(context_att)
            pri_att_logsigma = self.pri_att_logsigma(context_att)
            pri_idm_mean = self.pri_idm_mean(context_idm)
            pri_idm_logsigma = self.pri_idm_logsigma(context_idm)

            pri_params = [pri_att_mean, pri_idm_mean, pri_att_logsigma, pri_idm_logsigma]
            return pri_params

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 100
        self.architecture_def()

    def architecture_def(self):
        self.linear_layer = TimeDistributed(Dense(100))
        self.lstm_layer = LSTM(self.enc_units)
        # self.masking = Masking()
    def call(self, inputs):
        # enc_h = self.lstm_layer(self.masking(inputs))
        enc_h = self.lstm_layer(self.linear_layer(inputs))
        return enc_h

class FutureEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureEncoder, self).__init__(name="FutureEncoder")
        self.enc_units = 100
        self.architecture_def()

    def architecture_def(self):
        self.linear_layer = TimeDistributed(Dense(100))
        self.lstm_layer = Bidirectional(LSTM(self.enc_units), merge_mode='concat')

    def call(self, inputs):
        enc_acts = self.lstm_layer(self.linear_layer(inputs))
        return enc_acts

class IDMForwardSim(tf.keras.Model):
    def __init__(self):
        super(IDMForwardSim, self).__init__(name="IDMForwardSim")
        self.architecture_def()
        self.attention_temp = 5 # the higher, the sharper the attention

    def architecture_def(self):
        self.linear_layer = Dense(100)
        self.lstm_layer = LSTM(100, return_sequences=True, return_state=True)
        self.attention_neu = TimeDistributed(Dense(1))

    def idm_driver(self, vel, dv, dx, idm_params):
        dx = tf.clip_by_value(dx, clip_value_min=0.5, clip_value_max=1000.)
        desired_v = idm_params[:,:,0:1]
        desired_tgap = idm_params[:,:,1:2]
        min_jamx = idm_params[:,:,2:3]
        max_act = idm_params[:,:,3:4]
        min_act = idm_params[:,:,4:5]
        # tf.print('desired_v: ', tf.reduce_mean(desired_v))
        # tf.print('desired_v_max: ', tf.reduce_max(desired_v))
        # tf.print('desired_v_min: ', tf.reduce_min(desired_v))
        # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        # tf.print('min_jamx: ', tf.reduce_min(min_jamx))
        # tf.print('max_act: ', tf.reduce_mean(max_act))
        # tf.print('min_act: ', tf.reduce_mean(min_act))
        desired_gap = min_jamx + K.relu(desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act)))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)

        return self.action_clip(act) # action in simulator are also clipped

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3., clip_value_max=3.)

    def add_noise(self, idm_action, idm_veh_exists, batch_size):
        """
        To deal with nonexisting cars.
        """
        idm_action = idm_veh_exists*(idm_action) + \
                (1-idm_veh_exists)*tf.random.normal((batch_size, 1, 1), 0, 1)
        return idm_action

    def att_context(self, inputs, batch_size):
        att_projection, enc_h = inputs
        att_projection = tf.reshape(att_projection, [batch_size, 1, 100])
        enc_h = tf.reshape(enc_h, [batch_size, 1, 100])
        return tf.concat([att_projection, enc_h], axis=-1)

    def rollout(self, inputs):
        att_inputs, idm_params, idm_s, sdv_acts = inputs
        sampled_att_z, enc_h = att_inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        att_projection = self.linear_layer(sampled_att_z)
        att_context = tf.reshape(att_projection, [batch_size, 1, 100])
        # att_context = self.att_context([att_projection, enc_h], batch_size)
        state_h, state_c = att_projection, att_projection

        for step in range(40):
            f_veh_v = idm_s[:, step:step+1, 1:2]
            m_veh_v = idm_s[:, step:step+1, 2:3]
            f_veh_glob_x = idm_s[:, step:step+1, 4:5]
            m_veh_glob_x = idm_s[:, step:step+1, 5:6]
            # these to deal with missing cars
            f_veh_exists = idm_s[:, step:step+1, -2:-1]
            m_veh_exists = idm_s[:, step:step+1, -1:]
            if step == 0:
                ego_v = idm_s[:, step:step+1, 0:1]
                ego_glob_x = idm_s[:, step:step+1, 3:4]
            else:
                ego_v += _act*0.1
                ego_glob_x += ego_v*0.1 + 0.5*_act*0.1**2

            ef_delta_x = (f_veh_glob_x - ego_glob_x)
            em_delta_x = (m_veh_glob_x - ego_glob_x)
            ef_dv = (ego_v - f_veh_v)
            em_dv = (ego_v - m_veh_v)
            # tf.print('############ ef_act ############')
            ef_act = self.idm_driver(ego_v, ef_dv, ef_delta_x, idm_params)
            ef_act = self.add_noise(ef_act, f_veh_exists, batch_size)

            # tf.print('############ em_act ############')
            # tf.Assert(tf.greater(tf.reduce_min(em_delta_x), 0.),[em_delta_x])
            # tf.Assert(tf.greater(tf.reduce_min(ef_delta_x), 0.),[ef_delta_x])
            em_act = self.idm_driver(ego_v, em_dv, em_delta_x, idm_params)
            em_act = self.add_noise(em_act, m_veh_exists, batch_size)

            sdv_act = sdv_acts[:, step:step+1, :]
            lstm_output, state_h, state_c = self.lstm_layer(tf.concat([att_context, sdv_act], axis=-1), \
                                                            initial_state=[state_h, state_c])
            att_score = 1/(1+tf.exp(-self.attention_temp*self.attention_neu(lstm_output)))
            # att_score = idm_s[:, step:step+1, -3:-2]
            _act = (1-att_score)*ef_act + att_score*em_act
            if step == 0:
                act_seq = _act
                att_seq = att_score
            else:
                act_seq = tf.concat([act_seq, _act], axis=1)
                att_seq = tf.concat([att_seq, att_score], axis=1)

        return act_seq, att_seq

class IDMLayer(tf.keras.Model):
    def __init__(self):
        super(IDMLayer, self).__init__(name="IDMLayer")
        self.architecture_def()

    def architecture_def(self):
        self.linear_layer = Dense(100)
        self.des_v_neu = Dense(1)
        self.des_tgap_neu = Dense(1)
        self.min_jamx_neu = Dense(1)
        self.max_act_neu = Dense(1)
        self.min_act_neu = Dense(1)

    def param_activation(self, x, min_val, max_val, batch_size):
        activation_function = tf.tanh(0.5*x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def get_des_v(self, x, batch_size):
        output = self.des_v_neu(x)
        # return 15 + 15*(1/(1+tf.exp(-1*output)))
        return 15 + 20*(1/(1+tf.exp(-1*output)))

    def get_des_tgap(self, x, batch_size):
        output = self.des_tgap_neu(x)
        # return 1 + 1*(1/(1+tf.exp(-1*output)))
        return 0.5 + 2*(1/(1+tf.exp(-1*output)))

    def get_min_jamx(self, x, batch_size):
        output = self.min_jamx_neu(x)
        # return 4*(1/(1+tf.exp(-1*output)))
        return 5*(1/(1+tf.exp(-1*output)))

    def get_max_act(self, x, batch_size):
        output = self.max_act_neu(x)
        # return 0.8 + 1.2*(1/(1+tf.exp(-1*output)))
        return 0.5 + 2*(1/(1+tf.exp(-1*output)))

    def get_min_act(self, x, batch_size):
        output = self.min_act_neu(x)
        # return 1 + 2*(1/(1+tf.exp(-1*output)))
        return 0.5 + 3*(1/(1+tf.exp(-1*output)))

    def call(self, inputs):
        sampled_idm_z, enc_h = inputs
        batch_size = tf.shape(sampled_idm_z)[0]
        x = self.linear_layer(sampled_idm_z)
        # x = tf.concat([self.linear_layer(sampled_idm_z), enc_h], axis=-1)

        desired_v = self.get_des_v(x, batch_size)
        desired_tgap = self.get_des_tgap(x, batch_size)
        min_jamx = self.get_min_jamx(x, batch_size)
        max_act = self.get_max_act(x, batch_size)
        min_act = self.get_min_act(x, batch_size)
        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=-1)
        return idm_param

class IDMForwardSimLaneKeep(IDMForwardSim):
    def __init__(self):
        super().__init__()

    def rollout(self, inputs):
        att_inputs, idm_params, idm_s, sdv_acts = inputs
        sampled_att_z, enc_h = att_inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        for step in range(40):
            f_veh_v = idm_s[:, step:step+1, 1:2]
            f_veh_glob_x = idm_s[:, step:step+1, 3:4]

            if step == 0:
                ego_v = idm_s[:, step:step+1, 0:1]
                ego_glob_x = idm_s[:, step:step+1, 2:3]
            else:
                ego_v += ef_act*0.1
                ego_glob_x += ego_v*0.1 + 0.5*ef_act*0.1**2

            ef_delta_x = (f_veh_glob_x - ego_glob_x)
            ef_dv = (ego_v - f_veh_v)
            tf.Assert(tf.greater(tf.reduce_min(ef_delta_x), 0.),[ef_delta_x])
            ef_act = self.idm_driver(ego_v, ef_dv, ef_delta_x, idm_params)
            if step == 0:
                act_seq = ef_act
            else:
                act_seq = tf.concat([act_seq, ef_act], axis=1)

        return act_seq, act_seq

class NeurIDMModelLaneKeep(NeurIDMModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.idm_sim = IDMForwardSimLaneKeep()
