
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Masking
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class NeurLatentModel(AbstractModel):
    def __init__(self, config=None):
        super(NeurLatentModel, self).__init__(config)
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.act_encoder = FutureEncoder() # sdv's future action
        self.belief_net = BeliefModel()
        self.forward_sim = ForwardSim()
        self.vae_loss_weight = 0.01 # default
        # self.loss_function = tf.keras.losses.Huber()
        self.loss_function = tf.keras.losses.Huber()

    def callback_def(self):
        self.train_mseloss = tf.keras.metrics.Mean()
        self.test_mseloss = tf.keras.metrics.Mean()
        self.train_klloss = tf.keras.metrics.Mean()
        self.test_klloss = tf.keras.metrics.Mean()

    def mse(self, act_true, act_pred):
        act_true = (act_true)/0.1
        act_pred = (act_pred)/0.1
        return self.loss_function(act_true, act_pred)

    def kl_loss(self, pri_params, pos_params):
        pri_mean, pri_logsigma = pri_params
        pos_mean, pos_logsigma = pos_params

        prior = tfd.Normal(loc=pri_mean, scale=tf.exp(pri_logsigma))
        posterior = tfd.Normal(loc=pos_mean, scale=tf.exp(pos_logsigma))
        return tf.reduce_mean(tfp.distributions.kl_divergence(posterior, prior))

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
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f = self.f_seq_encoder(inputs[1])

        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')
        sampled_z = self.belief_net.sample_z(pos_params)
        act_seq = self.forward_sim.rollout([sampled_z, inputs[2], inputs[-1]])

        return act_seq, pri_params, pos_params

class BeliefModel(tf.keras.Model):
    def __init__(self):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.latent_dim = 10
        self.architecture_def()

    def architecture_def(self):
        self.pri_mean = Dense(self.latent_dim)
        self.pri_logsigma = Dense(self.latent_dim)
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)
        self.pri_projection = Dense(100, activation='relu')
        self.pos_projection = Dense(100, activation='relu')

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        z_sigma =  K.exp(z_logsigma)
        sampled_z = z_mean + z_sigma*_epsilon
        # sampled_z = z_mean
        # tf.print('z_mean: ', tf.reduce_mean(K.exp(z_logsigma)))
        # tf.print('z_min: ', tf.reduce_min(K.exp(z_logsigma)))
        # tf.print('z1_max: ', tf.reduce_max(z_sigma[:, 0]))
        # tf.print('z2_max: ', tf.reduce_max(z_sigma[:, 1]))
        # tf.print('z3_max: ', tf.reduce_max(z_sigma[:, 2]))

        return sampled_z
        # return sampled_z, sampled_z for single latent

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

class ForwardSim(tf.keras.Model):
    def __init__(self):
        super(ForwardSim, self).__init__(name="ForwardSim")
        self.architecture_def()

    def architecture_def(self):
        self.proj_layer_1 = Dense(100, activation='relu')
        self.proj_layer_2 = Dense(100, activation='relu')
        self.lstm_layer = LSTM(100, return_sequences=True, return_state=True)
        self.action_neu = TimeDistributed(Dense(1)) # a form

    def scale_features(self, env_state):
        env_state = (env_state-self.scaler.mean_)/self.scaler.var_**0.5
        return env_state

    def projection(self, x):
        x = self.proj_layer_1(x)
        x = self.proj_layer_2(x)
        return x

    def rollout(self, inputs):
        sampled_z, idm_s, sdv_acts = inputs
        batch_size = tf.shape(idm_s)[0]
        latent_projection = self.projection(sampled_z)
        proj_latent  = tf.reshape(latent_projection, [batch_size, 1, 100])
        state_h, state_c = latent_projection, latent_projection

        for step in range(40):
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

            ef_delta_x = (f_veh_glob_x - ego_glob_x)*f_veh_exists+\
                                                (1-f_veh_exists)*ef_delta_x_true
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                                                (1-m_veh_exists)*em_delta_x_true
            ef_dv = (ego_v - f_veh_v)*f_veh_exists+\
                                                (1-f_veh_exists)*ef_dv_true
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                                                (1-m_veh_exists)*em_dv_true
            sdv_act = sdv_acts[:, step:step+1, :]
            env_state = tf.concat([ego_v, f_veh_v, m_veh_v, \
                            ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_features(env_state)

            lstm_output, state_h, state_c = self.lstm_layer(tf.concat([\
                                    proj_latent, sdv_act, env_state], axis=-1), \
                                    initial_state=[state_h, state_c])
            _act = self.action_neu(lstm_output)
            if step == 0:
                act_seq = _act
            else:
                act_seq = tf.concat([act_seq, _act], axis=1)

        return act_seq

class ForwardSimOneStep(ForwardSim):
    def __init__(self):
        super().__init__()

    def rollout(self, inputs):
        sampled_z, idm_s, sdv_acts = inputs
        batch_size = tf.shape(idm_s)[0]
        latent_projection = self.linear_layer(sampled_z)
        proj_latent  = tf.reshape(latent_projection, [batch_size, 1, 100])
        state_h, state_c = latent_projection, latent_projection

        for step in range(1):
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

            ef_delta_x = (f_veh_glob_x - ego_glob_x)*f_veh_exists+\
                                                (1-f_veh_exists)*ef_delta_x_true
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                                                (1-m_veh_exists)*em_delta_x_true
            ef_dv = (ego_v - f_veh_v)*f_veh_exists+\
                                                (1-f_veh_exists)*ef_dv_true
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                                                (1-m_veh_exists)*em_dv_true
            sdv_act = sdv_acts[:, step:step+1, :]
            env_state = tf.concat([ego_v, f_veh_v, m_veh_v, \
                            ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_features(env_state)

            lstm_output, state_h, state_c = self.lstm_layer(tf.concat([\
                                    proj_latent, sdv_act, env_state], axis=-1), \
                                    initial_state=[state_h, state_c])
            _act = self.action_neu(lstm_output)
            if step == 0:
                act_seq = _act
            else:
                act_seq = tf.concat([act_seq, _act], axis=1)

        return act_seq

class NeurLatentModelOneStep(NeurLatentModel):
    def __init__(self, config=None):
        super(NeurLatentModelOneStep, self).__init__(config)
        self.forward_sim = ForwardSimOneStep()

    def mse(self, act_true, act_pred):
        act_true = (act_true[:, 0:1, :])/0.1
        act_pred = (act_pred[:, 0:1, :])/0.1
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f = self.f_seq_encoder(inputs[1])

        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')
        sampled_z = self.belief_net.sample_z(pos_params)
        act_seq = self.forward_sim.rollout([sampled_z, inputs[2], inputs[-1]])

        return act_seq, pri_params, pos_params


        def rollout(self, inputs):
            sampled_z, idm_s, sdv_acts = inputs
            batch_size = tf.shape(idm_s)[0]
            latent_projection = self.linear_layer(sampled_z)
            proj_latent  = tf.reshape(latent_projection, [batch_size, 1, 100])
            state_h, state_c = latent_projection, latent_projection

            # for step in range(60):
            for step in range(1):
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

                ef_delta_x = (f_veh_glob_x - ego_glob_x)*f_veh_exists+\
                                                    (1-f_veh_exists)*ef_delta_x_true
                em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                                                    (1-m_veh_exists)*em_delta_x_true
                ef_dv = (ego_v - f_veh_v)*f_veh_exists+\
                                                    (1-f_veh_exists)*ef_dv_true
                em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                                                    (1-m_veh_exists)*em_dv_true
                sdv_act = sdv_acts[:, step:step+1, :]
                env_state = tf.concat([ego_v, f_veh_v, m_veh_v, \
                                ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
                env_state = self.scale_features(env_state)

                lstm_output, state_h, state_c = self.lstm_layer(tf.concat([\
                                        proj_latent, sdv_act, env_state], axis=-1), \
                                        initial_state=[state_h, state_c])
                _act = self.action_neu(lstm_output)
                if step == 0:
                    act_seq = _act
                else:
                    act_seq = tf.concat([act_seq, _act], axis=1)

            return act_seq
