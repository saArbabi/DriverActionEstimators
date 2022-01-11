
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, LeakyReLU
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class NeurIDMModel(AbstractModel):
    def __init__(self, config):
        super(NeurIDMModel, self).__init__(config)
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.belief_net = BeliefModel(config)
        self.forward_sim = IDMForwardSim(config)
        self.idm_layer = IDMLayer()

        self.loss_function = tf.keras.losses.Huber()
        self.vae_loss_weight = config['model_config']['vae_loss_weight']
        # self.loss_function = tf.keras.losses.MeanAbsoluteError()
        # self.loss_function = tf.keras.losses.MeanSquaredError()

    def callback_def(self):
        self.train_mseloss = tf.keras.metrics.Mean()
        self.test_mseloss = tf.keras.metrics.Mean()
        self.train_klloss = tf.keras.metrics.Mean()
        self.test_klloss = tf.keras.metrics.Mean()

    def mse(self, act_true, act_pred):
        act_true = (act_true)/0.1
        act_pred = (act_pred)/0.1
        loss = self.loss_function(act_true, act_pred)
        tf.debugging.check_numerics(loss, message='Checking loss')
        return loss

    def kl_loss(self, pri_params, pos_params):
        pri_mean, pri_logsigma = pri_params
        pos_mean, pos_logsigma = pos_params

        prior = tfd.Normal(loc=pri_mean, scale=tf.exp(pri_logsigma))
        posterior = tfd.Normal(loc=pos_mean, scale=tf.exp(pos_logsigma))
        return tf.reduce_mean(tfp.distributions.kl_divergence(posterior, prior))

    def train_loop(self, data_objs):
        # tf.print('######## TRAIN #######:')
        train_ds = self.batch_data(data_objs)

        for history_sca, future_sca, future_idm_s, future_m_veh_c, future_ego_a in train_ds:
            self.train_step([history_sca, future_sca, future_idm_s, future_m_veh_c], future_ego_a)

    def test_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for history_sca, future_sca, future_idm_s, future_m_veh_c, future_ego_a in train_ds:
            self.test_step([history_sca, future_sca, future_idm_s, future_m_veh_c], future_ego_a)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, pri_params, pos_params = self(states)
            mse_loss = self.mse(targets, act_pred)
            kl_loss = self.kl_loss(pri_params, pos_params)
            loss = self.vae_loss(mse_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_mseloss.reset_states()
        self.train_klloss.reset_states()
        self.train_mseloss(mse_loss)
        self.train_klloss(kl_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, pri_params, pos_params = self(states)
        mse_loss = self.mse(targets, act_pred)
        kl_loss = self.kl_loss(pri_params, pos_params)
        loss = self.vae_loss(mse_loss, kl_loss)
        self.test_mseloss.reset_states()
        self.test_klloss.reset_states()
        self.test_mseloss(mse_loss)
        self.test_klloss(kl_loss)

    def vae_loss(self, mse_loss, kl_loss):
        return  self.vae_loss_weight*kl_loss + mse_loss

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history
        enc_f = self.f_seq_encoder(inputs[1])
        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')

        z_idm, z_att = self.belief_net.sample_z(pos_params)
        proj_idm = self.belief_net.z_proj_idm(z_idm)
        proj_att = self.belief_net.z_proj_att(z_att)
        idm_params = self.idm_layer(proj_idm)
        act_seq, _ = self.forward_sim.rollout([\
                                idm_params, proj_att,
                                inputs[2], inputs[-1]])
        # tf.print('###############:')
        # tf.print('att_scoreax: ', tf.reduce_max(att_scores))
        # tf.print('att_scorein: ', tf.reduce_min(att_scores))
        # tf.print('att_scoreean: ', tf.reduce_mean(att_scores))
        return act_seq, pri_params, pos_params

class BeliefModel(tf.keras.Model):
    def __init__(self, config):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.proj_dim = 64
        self.latent_dim = config['model_config']['latent_dim']
        self.architecture_def()

    def architecture_def(self):
        self.pri_mean = Dense(self.latent_dim)
        self.pri_logsigma = Dense(self.latent_dim)
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)
        self.pri_projection = Dense(self.proj_dim, activation=LeakyReLU())
        self.pos_projection = Dense(self.proj_dim, activation=LeakyReLU())
        ####
        self.proj_idm_1 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_idm_2 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_idm_3 = Dense(self.proj_dim)

        self.proj_att_1 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_att_2 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_att_3 = Dense(self.proj_dim)

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        z_sigma =  K.exp(z_logsigma)
        sampled_z = z_mean + z_sigma*_epsilon
        # tf.print('z_min: ', tf.reduce_min(z_sigma))
        # tf.print('z_max: ', tf.reduce_max(z_sigma))
        return sampled_z[:, :3], sampled_z[:, 3:]
        # return sampled_z[:, :3], sampled_z

    def z_proj_idm(self, x):
        x = self.proj_idm_1(x)
        x = self.proj_idm_2(x)
        x = self.proj_idm_3(x)
        return x

    def z_proj_att(self, x):
        x = self.proj_att_1(x)
        x = self.proj_att_2(x)
        x = self.proj_att_3(x)
        return x

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_f = inputs
            # prior
            pri_context = self.pri_projection(enc_h)
            pri_mean = self.pri_mean(pri_context)
            pri_logsigma = self.pri_logsigma(pri_context)

            # posterior
            pos_context = self.pos_projection(tf.concat([enc_h, enc_f], axis=-1))
            pos_mean = self.pos_mean(pos_context)
            pos_logsigma = self.pos_logsigma(pos_context)

            pri_params = [pri_mean, pri_logsigma]
            pos_params = [pos_mean, pos_logsigma]
            return pri_params, pos_params

        elif dis_type == 'prior':
            pri_context = self.pri_projection(inputs)
            pri_mean = self.pri_mean(pri_context)
            pri_logsigma = self.pri_logsigma(pri_context)
            pri_params = [pri_mean, pri_logsigma]
            return pri_params

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer_1 = LSTM(self.enc_units, return_sequences=True)
        self.lstm_layer_2 = LSTM(self.enc_units)

    def call(self, inputs):
        whole_seq_output = self.lstm_layer_1(inputs)
        enc_h = self.lstm_layer_2(whole_seq_output)
        return enc_h

class FutureEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureEncoder, self).__init__(name="FutureEncoder")
        self.enc_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = Bidirectional(LSTM(self.enc_units), merge_mode='concat')

    def call(self, inputs):
        enc_acts = self.lstm_layer(inputs)
        return enc_acts

class IDMForwardSim(tf.keras.Model):
    def __init__(self, config):
        super(IDMForwardSim, self).__init__(name="IDMForwardSim")
        self.attention_temp = config['model_config']['attention_temp']
        self.proj_dim = 64
        self.dec_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.res_act_neu = TimeDistributed(Dense(1))
        self.att_neu = TimeDistributed(Dense(1))

    def idm_driver(self, vel, dv, dx, idm_params):
        dx = tf.clip_by_value(dx, clip_value_min=1, clip_value_max=1000.)

        desired_v = idm_params[:,:,0:1]
        desired_tgap = idm_params[:,:,1:2]
        min_jamx = idm_params[:,:,2:3]
        max_act = idm_params[:,:,3:4]
        min_act = idm_params[:,:,4:5]

        _gap_denum  = 2*tf.sqrt(max_act*min_act)
        _gap = desired_tgap*vel+(vel*dv)/_gap_denum
        desired_gap = min_jamx + K.relu(_gap)
        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)


        # return self.action_clip(act)
        return self.action_clip(act)

    def action_clip(self, action):
        "This is needed to avoid infinities"
        return tf.clip_by_value(action, clip_value_min=-5.5, clip_value_max=5.5)

    def scale_env_s(self, env_state):
        env_state = (env_state-self.env_scaler.mean_)/self.env_scaler.var_**0.5
        return env_state

    def get_att(self, lstm_output):
        att_x = self.att_neu(lstm_output)
        return 1/(1+tf.exp(-self.attention_temp*att_x))

    def handle_merger(self, att_score, dx, m_veh_exists):
        return att_score*tf.cast(tf.greater(dx*m_veh_exists, 0.), tf.float32)

    def rollout(self, inputs):
        idm_params, proj_belief, idm_s, merger_cs = inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        proj_belief  = tf.reshape(proj_belief, [batch_size, 1, self.proj_dim])
        # enc_h  = tf.reshape(enc_h, [batch_size, 1, 128])
        state_h = state_c = tf.zeros([batch_size, self.dec_units])

        for step in range(50):
            f_veh_v = idm_s[:, step:step+1, 1:2]
            m_veh_v = idm_s[:, step:step+1, 2:3]
            f_veh_glob_x = idm_s[:, step:step+1, 4:5]
            m_veh_glob_x = idm_s[:, step:step+1, 5:6]

            em_dv_true = idm_s[:, step:step+1, 8:9]
            em_delta_x_true = idm_s[:, step:step+1, 9:10]

            # these to deal with missing cars
            m_veh_exists = idm_s[:, step:step+1, -1:]
            if step == 0:
                ego_v = idm_s[:, step:step+1, 0:1]
                ego_glob_x = idm_s[:, step:step+1, 3:4]
            else:
                ego_v += _act*0.1
                ego_glob_x += ego_v*0.1 + 0.5*_act*0.1**2

            ef_delta_x = (f_veh_glob_x - ego_glob_x)
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_x']
            ef_dv = (ego_v - f_veh_v)
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_v']
            # tf.print('############ ef_act ############')

            env_state = tf.concat([ego_v, f_veh_v, \
                                    ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_env_s(env_state)
            merger_c = merger_cs[:, step:step+1, :]

            lstm_output, state_h, state_c = self.lstm_layer(tf.concat([\
                                proj_belief, env_state, merger_c], axis=-1), \
                                    initial_state=[state_h, state_c])

            att_score = self.get_att(lstm_output)
            # att_score = self.handle_merger(att_score, em_delta_x, m_veh_exists)

            ef_act = self.idm_driver(ego_v, ef_dv, ef_delta_x, idm_params)
            em_act = self.idm_driver(ego_v, em_dv, em_delta_x, idm_params)

            # att_score = idm_s[:, step:step+1, -3:-2]
            _act = (1-att_score)*ef_act + att_score*em_act
            if step == 0:
                act_seq = _act
                att_seq = att_score
            else:
                act_seq = tf.concat([act_seq, _act], axis=1)
                att_seq = tf.concat([att_seq, att_score], axis=1)
        # tf.print('att_score: ', tf.reduce_mean(att_seq))
        return act_seq, att_seq

class IDMLayer(tf.keras.Model):
    def __init__(self):
        super(IDMLayer, self).__init__(name="IDMLayer")
        self.architecture_def()

    def architecture_def(self):
        self.des_v_neu = Dense(1)
        self.des_tgap_neu = Dense(1)
        self.min_jamx_neu = Dense(1)
        self.max_act_neu = Dense(1)
        self.min_act_neu = Dense(1)

    def get_des_v(self, x):
        output = self.des_v_neu(x)
        minval = 10
        maxval = 30
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def get_des_tgap(self, x):
        output = self.des_tgap_neu(x)
        minval = 0
        maxval = 3
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def get_min_jamx(self, x):
        output = self.min_jamx_neu(x)
        minval = 0
        maxval = 6
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def get_max_act(self, x):
        output = self.max_act_neu(x)
        minval = 1
        maxval = 6
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def get_min_act(self, x):
        output = self.min_act_neu(x)
        minval = 1
        maxval = 6
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def call(self, x):
        desired_v = self.get_des_v(x)
        desired_tgap = self.get_des_tgap(x)
        min_jamx = self.get_min_jamx(x)
        max_act = self.get_max_act(x)
        min_act = self.get_min_act(x)
        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=-1)
        return idm_param
