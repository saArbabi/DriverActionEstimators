
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, LeakyReLU
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class LatentMLP(AbstractModel):
    def __init__(self, config):
        super(LatentMLP, self).__init__(config)
        self.h_seq_encoder = HistoryEncoder()
        self.belief_net = BeliefModel(config)
        self.forward_sim = ForwardSim()
        self.vae_loss_weight = config['model_config']['vae_loss_weight']

    def callback_def(self):
        self.train_llloss = tf.keras.metrics.Mean() # log likelihood loss
        self.test_llloss = tf.keras.metrics.Mean()
        self.train_klloss = tf.keras.metrics.Mean()
        self.test_klloss = tf.keras.metrics.Mean()

    def log_loss(self, act_true, pred_dis):
        likelihood = pred_dis.log_prob(act_true)
        return -tf.reduce_mean(likelihood)

    def kl_loss(self, pos_params):
        pos_mean, pos_logsigma = pos_params

        prior = tfd.Normal(loc=tf.zeros(self.belief_net.latent_dim), scale=1)
        posterior = tfd.Normal(loc=pos_mean, scale=tf.exp(pos_logsigma))
        return tf.reduce_mean(tfd.kl_divergence(posterior, prior))

    def train_loop(self, data_objs):
        # tf.print('######## TRAIN #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, _, future_idm_s, future_m_veh_c, future_ego_a in train_ds:
            self.train_step([history_sca, future_idm_s, future_m_veh_c], future_ego_a)

    def test_loop(self, data_objs, epoch):
        # tf.print('######## TEST #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, _, future_idm_s, future_m_veh_c, future_ego_a in train_ds:
            self.test_step([history_sca, future_idm_s, future_m_veh_c], future_ego_a)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_dis, pos_params = self(states)
            ll_loss = self.log_loss(targets, act_dis)
            kl_loss = self.kl_loss(pos_params)
            loss = self.vae_loss(ll_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_llloss.reset_states()
        self.train_klloss.reset_states()
        self.train_llloss(ll_loss)
        self.train_klloss(kl_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_dis, pos_params = self(states)
        ll_loss = self.log_loss(targets, act_dis)
        kl_loss = self.kl_loss(pos_params)
        loss = self.vae_loss(ll_loss, kl_loss)
        self.test_llloss.reset_states()
        self.test_klloss.reset_states()
        self.test_llloss(ll_loss)
        self.test_klloss(kl_loss)

    def vae_loss(self, ll_loss, kl_loss):
        return  self.vae_loss_weight*kl_loss + ll_loss

    def call(self, inputs):
        # inputs: [history_sca, future_idm_s, future_m_veh_c]
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        pos_params = self.belief_net(enc_h)
        sampled_z = self.belief_net.sample_z(pos_params)
        batch_size = tf.shape(inputs[0])[0]
        sampled_z = tf.reshape(sampled_z, [batch_size, 1, self.belief_net.latent_dim])
        mean_seq, var_seq = self.forward_sim.rollout([sampled_z, inputs[1], inputs[-1]])
        act_dis = tfd.Normal(mean_seq, var_seq, name='Normal')
        return act_dis, pos_params

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
        self.proj_layer_1 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_layer_2 = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_layer_3 = Dense(self.proj_dim)

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        z_sigma =  K.exp(z_logsigma)
        sampled_z = z_mean + z_sigma*_epsilon
        # tf.print('z_min: ', tf.reduce_min(z_sigma))
        return sampled_z

    def belief_proj(self, x):
        x = self.proj_layer_1(x)
        x = self.proj_layer_2(x)
        x = self.proj_layer_3(x)
        return x

    def call(self, inputs):
        # posterior
        pos_mean = self.pos_mean(inputs)
        pos_logsigma = self.pos_logsigma(inputs)
        pos_params = [pos_mean, pos_logsigma]
        return pos_params

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units)
    def call(self, inputs):
        enc_h = self.lstm_layer(inputs)
        return enc_h

class ForwardSim(tf.keras.Model):
    def __init__(self):
        super(ForwardSim, self).__init__(name="ForwardSim")
        self.proj_dim = 64
        self.architecture_def()

    def architecture_def(self):
        self.dense_1 = TimeDistributed(Dense(128, activation=K.relu))
        self.dense_2 = TimeDistributed(Dense(128, activation=K.relu))
        self.neu_mean = Dense(1)
        self.neu_var = Dense(1, activation=K.exp)

    def get_dis(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        neu_mean = self.neu_mean(x)
        neu_var = self.neu_var(x)
        return neu_mean, neu_var

    def scale_env_s(self, env_state):
        env_state = (env_state-self.env_scaler.mean_)/self.env_scaler.var_**0.5
        return env_state

    def rollout(self, inputs):
        sampled_z, idm_s, merger_cs = inputs

        for step in range(30):
            f_veh_v = idm_s[:, step:step+1, 1:2]
            m_veh_v = idm_s[:, step:step+1, 2:3]
            f_veh_glob_x = idm_s[:, step:step+1, 4:5]
            m_veh_glob_x = idm_s[:, step:step+1, 5:6]

            ef_dv_true = idm_s[:, step:step+1, 6:7]
            ef_delta_x_true = idm_s[:, step:step+1, 7:8]
            em_dv_true = idm_s[:, step:step+1, 8:9]
            em_delta_x_true = idm_s[:, step:step+1, 9:10]

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
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_x']
            ef_dv = (ego_v - f_veh_v)
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_v']

            env_state = tf.concat([ego_v, f_veh_v, \
                                    ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_env_s(env_state)
            merger_c = tf.concat([merger_cs[:, step:step+1, :], m_veh_exists], axis=-1)

            _mean, _var = self.get_dis(tf.concat([\
                                    sampled_z, env_state, merger_c], axis=-1))
            _act = tfd.Normal(_mean, _var, name='Normal').sample()
            if step == 0:
                mean_seq = _mean
                var_seq = _var
            else:
                mean_seq = tf.concat([mean_seq, _mean], axis=1)
                var_seq = tf.concat([var_seq, _var], axis=1)

        return mean_seq, var_seq
