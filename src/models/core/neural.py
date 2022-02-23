
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, LeakyReLU
from keras import backend as K
from importlib import reload
from models.core import neural_idm
reload(neural_idm)
from models.core.neural_idm import  NeurIDMModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class NeurLatentModel(NeurIDMModel):
    def __init__(self, config, exp_id=None):
        super(NeurLatentModel, self).__init__(config, exp_id)
        self.belief_net = BeliefModel(config)
        self.forward_sim = ForwardSim()
        if exp_id:
            self.exp_dir = './src/models/experiments/' + 'neural_' + exp_id

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
        tf.debugging.check_numerics(loss, message='Checking loss')

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # with self.train_writer.as_default():
        #     tf.summary.scalar('displacement_loss', displacement_loss, step=step)
        #     tf.summary.scalar('action_loss', action_loss, step=step)
        #     tf.summary.scalar('kl_loss', kl_loss, step=step)
        #     tf.summary.scalar('tot_loss', loss, step=step)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets, step):
        _pred, pri_params, pos_params = self(states)
        displacement_loss = self.get_displacement_loss(targets, _pred[0])
        action_loss = self.get_action_loss(targets, _pred[1])
        kl_loss = self.get_kl_loss(pri_params, pos_params)
        loss = self.get_tot_loss(kl_loss,
                                 displacement_loss,
                                 action_loss)

        # with self.test_writer.as_default():
        #     tf.summary.scalar('displacement_loss', displacement_loss, step=step)
        #     tf.summary.scalar('action_loss', action_loss, step=step)
        #     tf.summary.scalar('kl_loss', kl_loss, step=step)
        #     tf.summary.scalar('tot_loss', loss, step=step)

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history lstm state
        enc_f = self.f_seq_encoder(inputs[1])

        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')
        sampled_z = self.belief_net.sample_z(pos_params)
        proj_latent = self.belief_net.z_proj(sampled_z)
        displacement_seq, action_seq = self.forward_sim.rollout([\
                                    proj_latent, enc_h, inputs[2], inputs[-1]])
        return [displacement_seq, action_seq], pri_params, pos_params

class BeliefModel(tf.keras.Model):
    def __init__(self, config):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.proj_dim = 128
        self.latent_dim = config['model_config']['latent_dim']
        self.architecture_def()

    def architecture_def(self):
        self.pri_mean = Dense(self.latent_dim)
        self.pri_logsigma = Dense(self.latent_dim)
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)

        self.proj_pri = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_pos = Dense(self.proj_dim, activation=LeakyReLU())

        self.proj_layer = Dense(self.proj_dim, activation=LeakyReLU())

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        z_sigma =  K.exp(z_logsigma)
        sampled_z = z_mean + z_sigma*_epsilon
        # tf.print('z_min: ', tf.reduce_min(z_sigma))
        return sampled_z

    def pri_proj(self, enc_h):
        return self.proj_pri(enc_h)

    def pos_proj(self, enc_h, enc_f):
        return self.proj_pos(tf.concat([enc_h, enc_f], axis=-1))

    def z_proj(self, x):
        return self.proj_layer(x)

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_f = inputs
            # prior
            pri_context = self.pri_proj(enc_h)
            pri_mean = self.pri_mean(pri_context)
            pri_logsigma = self.pri_logsigma(pri_context)

            # posterior
            pos_context = self.pos_proj(enc_h, enc_f)
            pos_mean = self.pos_mean(pos_context)
            pos_logsigma = self.pos_logsigma(pos_context)

            pri_params = [pri_mean, pri_logsigma]
            pos_params = [pos_mean, pos_logsigma]
            return pri_params, pos_params

        elif dis_type == 'prior':
            pri_context = self.pri_proj(inputs)
            pri_mean = self.pri_mean(pri_context)
            pri_logsigma = self.pri_logsigma(pri_context)
            pri_params = [pri_mean, pri_logsigma]
            return pri_params


class ForwardSim(tf.keras.Model):
    def __init__(self):
        super(ForwardSim, self).__init__(name="ForwardSim")
        self.proj_dim = 128
        self.dec_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.att_layer_1 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.att_layer_2 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.action_neu = TimeDistributed(Dense(1))

    def scale_env_s(self, env_state):
        env_state = (env_state-self.env_scaler.mean_)/self.env_scaler.var_**0.5
        return env_state

    def getego_veh_aion(self, inputs):
        x = self.att_layer_1(inputs)
        x = self.att_layer_2(x)
        return self.action_neu(x)

    def rollout(self, inputs):
        proj_latent, enc_h, idm_s, merger_cs = inputs
        batch_size = tf.shape(idm_s)[0]
        proj_latent = tf.reshape(proj_latent, [batch_size, 1, self.proj_dim])
        enc_h = tf.reshape(enc_h, [batch_size, 1, self.dec_units])

        displacement = tf.zeros([batch_size, 1, 1])
        displacement_seq = displacement
        ego_v = idm_s[:,  0:1, 0:1]
        ego_veh_a = idm_s[:, 0:1, 11:12]
        ego_glob_x = idm_s[:,  0:1, 3:4]
        lstm_output = self.lstm_layer(ego_glob_x)
        for step in range(1, self.rollout_len+1):
            f_veh_v = idm_s[:, step-1:step, 1:2]
            m_veh_v = idm_s[:, step-1:step, 2:3]
            f_veh_glob_x = idm_s[:, step-1:step, 4:5]
            m_veh_glob_x = idm_s[:, step-1:step, 5:6]
            f_veh_a = idm_s[:, step-1:step, 12:13]

            em_dv_true = idm_s[:, step-1:step, 8:9]
            em_delta_x_true = idm_s[:, step-1:step, 9:10]

            # these to deal with missing cars
            m_veh_exists = idm_s[:, step-1:step, -1:]

            ef_delta_x = (f_veh_glob_x - ego_glob_x)
            em_delta_x = (m_veh_glob_x - ego_glob_x)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_x']
            ef_dv = (ego_v - f_veh_v)
            em_dv = (ego_v - m_veh_v)*m_veh_exists+\
                            (1-m_veh_exists)*self.dummy_value_set['em_delta_v']

            env_state = tf.concat([ego_v, f_veh_v, ego_veh_a, f_veh_a,\
                                    ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_env_s(env_state)
            merger_c = merger_cs[:, step-1:step, :]
            inputs = tf.concat([proj_latent, enc_h, env_state, merger_c], axis=-1)

            ego_veh_a = self.getego_veh_aion(inputs)

            displacement += ego_v*0.1 + 0.5*ego_veh_a*0.1**2
            ego_glob_x += ego_v*0.1 + 0.5*ego_veh_a*0.1**2
            ego_v += ego_veh_a*0.1
            if step-1 == 0:
                act_seq = ego_veh_a

            else:
                act_seq = tf.concat([act_seq, ego_veh_a], axis=1)
            displacement_seq = tf.concat([displacement_seq, displacement], axis=1)

        return displacement_seq, act_seq
