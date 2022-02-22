
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, LeakyReLU
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.neural_idm import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class NeurLatentModel(NeurIDMModel):
    def __init__(self, config):
        super(NeurLatentModel, self).__init__(config)

    def get_kl_loss(self, pri_params, pos_params):
        pri_mean, pri_logsigma = pri_params
        pos_mean, pos_logsigma = pos_params

        prior = tfd.Normal(loc=pri_mean, scale=tf.exp(pri_logsigma))
        posterior = tfd.Normal(loc=pos_mean, scale=tf.exp(pos_logsigma))
        return tf.reduce_mean(tfp.distributions.kl_divergence(posterior, prior))


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets, step):
        with tf.GradientTape() as tape:
            _pred, pri_params, pos_params = self(states)
            displacement_loss = self.get_displacement_loss(targets, _pred[0])
            action_loss = self.get_action_loss(targets, _pred[1])
            kl_loss = self.get_kl_loss(pri_params, pos_params)
            loss = self.get_tot_loss(kl_loss,
                                     displacement_loss,
                                     action_loss)


        gradients = tape.gradient(loss, self.trainable_variables)
        # tf.print('xxxxx gradients: ', gradients)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        with self.train_writer.as_default():
            tf.summary.scalar('displacement_loss', displacement_loss, step=step)
            tf.summary.scalar('action_loss', action_loss, step=step)
            tf.summary.scalar('kl_loss', kl_loss, step=step)
            tf.summary.scalar('tot_loss', loss, step=step)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets, step):
        _pred, pri_params, pos_params = self(states)
        displacement_loss = self.get_displacement_loss(targets, _pred[0])
        action_loss = self.get_action_loss(targets, _pred[1])
        kl_loss = self.get_kl_loss(pri_params, pos_params)
        loss = self.get_tot_loss(kl_loss,
                                 displacement_loss,
                                 action_loss)

        with self.test_writer.as_default():
            tf.summary.scalar('displacement_loss', displacement_loss, step=step)
            tf.summary.scalar('action_loss', action_loss, step=step)
            tf.summary.scalar('kl_loss', kl_loss, step=step)
            tf.summary.scalar('tot_loss', loss, step=step)

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f = self.f_seq_encoder(inputs[1])

        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')
        sampled_z = self.belief_net.sample_z(pos_params)
        proj_latent = self.belief_net.belief_proj(sampled_z)
        displacement_seq, _ = self.forward_sim.rollout([proj_latent, enc_h, inputs[2], inputs[-1]])
        return displacement_seq, pri_params, pos_params

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

class ForwardSim(tf.keras.Model):
    def __init__(self):
        super(ForwardSim, self).__init__(name="ForwardSim")
        self.proj_dim = 64
        self.dec_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.dense_1 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.dense_2 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.dense_3 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.action_neu = TimeDistributed(Dense(1))
        self.lstm_layer = LSTM(self.dec_units, return_sequences=True, return_state=True)

    def scale_env_s(self, env_state):
        env_state = (env_state-self.env_scaler.mean_)/self.env_scaler.var_**0.5
        return env_state

    def get_action(self, inputs, lstm_states):
        lstm_output, state_h, state_c = self.lstm_layer(inputs, initial_state=lstm_states)
        x = self.dense_1(lstm_output)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.action_neu(x), [state_h, state_c]

    def rollout(self, inputs):
        proj_latent, enc_h, idm_s, merger_cs = inputs
        batch_size = tf.shape(idm_s)[0]
        proj_latent = tf.reshape(proj_latent, [batch_size, 1, self.proj_dim])
        enc_h = tf.reshape(enc_h, [batch_size, 1, self.dec_units])
        init_lstm_state = tf.zeros([batch_size, self.dec_units])
        lstm_states = [init_lstm_state, init_lstm_state]

        for step in range(self.rollout_len):
            f_veh_v = idm_s[:, step:step+1, 1:2]
            m_veh_v = idm_s[:, step:step+1, 2:3]
            f_veh_glob_x = idm_s[:, step:step+1, 4:5]
            m_veh_glob_x = idm_s[:, step:step+1, 5:6]

            ef_dv_true = idm_s[:, step:step+1, 6:7]
            ef_delta_x_true = idm_s[:, step:step+1, 7:8]
            em_dv_true = idm_s[:, step:step+1, 8:9]
            em_delta_x_true = idm_s[:, step:step+1, 9:10]

            # these to deal with missing cars
            m_veh_exists = idm_s[:, step:step+1, -1:]
            if step == 0:
                ego_v = idm_s[:, step:step+1, 0:1]
                ego_glob_x = idm_s[:, step:step+1, 3:4]
                displacement = tf.zeros([batch_size, 1, 1])
            else:
                ego_v += _act*0.1
                ego_glob_x += ego_v*0.1 + 0.5*_act*0.1**2
                displacement += ego_v*0.1 + 0.5*_act*0.1**2


            ef_delta_x = (f_veh_glob_x - ego_glob_x)
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_x']
            ef_dv = (ego_v - f_veh_v)
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_v']

            env_state = tf.concat([ego_v, f_veh_v, \
                                    ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_env_s(env_state)
            merger_c = merger_cs[:, step:step+1, :]
            inputs = tf.concat([proj_latent, enc_h, env_state, merger_c], axis=-1)

            _act, lstm_states = self.get_action(inputs, lstm_states)
            if step == 0:
                act_seq = _act
                displacement_seq = displacement

            else:
                act_seq = tf.concat([act_seq, _act], axis=1)
                displacement_seq = tf.concat([displacement_seq, displacement], axis=1)

        return displacement_seq, act_seq
