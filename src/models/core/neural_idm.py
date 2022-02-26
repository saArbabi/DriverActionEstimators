
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
    def __init__(self, config, exp_id=None):
        super(NeurIDMModel, self).__init__(config)
        self.f_seq_encoder = FutureSeqEncoder()
        self.h_seq_encoder = HistorySeqEncoder()
        self.belief_net = BeliefModel(config)
        self.forward_sim = IDMForwardSim(config)
        self.idm_layer = IDMLayer()

        self.vae_loss_weight = config['model_config']['vae_loss_weight']
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.MeanSquaredError()
        if exp_id:
            self.exp_dir = './src/models/experiments/' + 'neural_idm_' + exp_id

    def make_event_files(self):
        self.train_writer = tf.summary.create_file_writer(self.exp_dir+'/logs/train')
        self.test_writer = tf.summary.create_file_writer(self.exp_dir+'/logs/test')

    def get_displacement_loss(self, _true, _pred):
        _true = (_true[:, :, -1:] - self.disx_means)/self.disx_std
        _pred = (_pred[:, 1:, :] - self.disx_means)/self.disx_std
        return self.loss_function(_true, _pred)

    def get_action_loss(self, _true, _pred):
        _true = (_true[:, :, 0:1] - self.action_means)/self.action_std
        _pred = (_pred[:, :, :] - self.action_means)/self.action_std
        return self.loss_function(_true, _pred)

    def get_tot_loss(self, kl_loss, displacement_loss, action_loss):
        return self.vae_loss_weight*kl_loss + displacement_loss + action_loss

    def get_kl_loss(self, pri_params, pos_params):
        pri_mean, pri_logsigma = pri_params
        pos_mean, pos_logsigma = pos_params

        prior = tfd.Normal(loc=pri_mean, scale=tf.exp(pri_logsigma))
        posterior = tfd.Normal(loc=pos_mean, scale=tf.exp(pos_logsigma))
        return tf.reduce_mean(tfp.distributions.kl_divergence(posterior, prior))

    def train_test_loop(self, train_test_data):
        # tf.print('######## TRAIN #######:')
        train_ds = self.batch_data(train_test_data[0], self.batch_size)
        test_ds = self.batch_data(train_test_data[1], int(self.batch_size/2))
        for step, batch_data in enumerate(zip(train_ds, test_ds)):
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            # train step
            self.train_step(batch_data[0][:-1], batch_data[0][-1], step)
            # test step
            # self.test_step(batch_data[1][:-1], batch_data[1][-1], step)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets, step):
        with tf.GradientTape() as tape:
            _pred, pri_params, pos_params = self(states)
            displacement_loss = self.get_displacement_loss(targets, _pred[0])
            action_loss = self.get_action_loss(targets, _pred[1])
            kl_loss = self.get_kl_loss(pri_params, pos_params)
            # tf.print('kl_loss_idm: ', kl_loss_idm)
            # tf.print('kl_loss_att: ', kl_loss_att)


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
        #
        # with self.test_writer.as_default():
        #     tf.summary.scalar('displacement_loss', displacement_loss, step=step)
        #     tf.summary.scalar('action_loss', action_loss, step=step)
        #     tf.summary.scalar('kl_loss', kl_loss, step=step)
        #     tf.summary.scalar('tot_loss', loss, step=step)

    def call(self, inputs):
        enc_h = self.h_seq_encoder(inputs[0]) # history
        enc_f = self.f_seq_encoder(inputs[1])
        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_f], dis_type='both')

        z_idm, z_att = self.belief_net.sample_z(pos_params)
        proj_idm = self.belief_net.z_proj_idm(z_idm)
        proj_att = self.belief_net.z_proj_att(z_att)
        idm_params = self.idm_layer(proj_idm)
        displacement_seq, action_seq, _ = self.forward_sim.rollout([\
                                idm_params, proj_att, enc_h,
                                inputs[2], inputs[-1]])
        # tf.print('###############:')
        # tf.print('att_scoreax: ', tf.reduce_max(att_scores))
        # tf.print('att_scorein: ', tf.reduce_min(att_scores))
        # tf.print('att_scoreean: ', tf.reduce_mean(att_scores))
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
        ####
        self.proj_idm = Dense(self.proj_dim, activation=LeakyReLU())
        self.proj_att = Dense(self.proj_dim, activation=LeakyReLU())

    def sample_z(self, distribution_params):
        z_mean, z_logsigma = distribution_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        z_sigma = K.exp(z_logsigma)
        sampled_z = z_mean + z_sigma*_epsilon
        # tf.print('z_min: ', tf.reduce_min(z_sigma))
        # tf.print('z_max: ', tf.reduce_max(z_sigma))
        # return sampled_z[:, :3], sampled_z[:, 3:]
        return sampled_z, sampled_z

    def pri_proj(self, enc_h):
        return self.proj_pri(enc_h)

    def pos_proj(self, enc_h, enc_f):
        return self.proj_pos(tf.concat([enc_h, enc_f], axis=-1))

    def z_proj_idm(self, x):
        return self.proj_idm(x)

    def z_proj_att(self, x):
        return self.proj_att(x)

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

class HistorySeqEncoder(tf.keras.Model):
    def __init__(self):
        super(HistorySeqEncoder, self).__init__(name="HistorySeqEncoder")
        self.enc_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units)

    def call(self, inputs):
        return self.lstm_layer(inputs)

class FutureSeqEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureSeqEncoder, self).__init__(name="FutureSeqEncoder")
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
        self.proj_dim = 128
        self.dec_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.f_att_neu = TimeDistributed(Dense(1))
        self.m_att_neu = TimeDistributed(Dense(1))
        self.lstm_layer = LSTM(self.dec_units)
        self.att_layer_1 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))
        self.att_layer_2 = TimeDistributed(Dense(self.dec_units, activation=LeakyReLU()))

    def idm_driver(self, idm_state, idm_params):
        vel, dv, dx = idm_state
        dx = tf.clip_by_value(dx, 0.1, 500)
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_params
        _gap_denum  = 2 * tf.sqrt(max_act * min_act)
        _gap = desired_tgap * vel + (vel * dv)/_gap_denum
        desired_gap = min_jamx + K.relu(_gap)
        act = max_act*(1 - (vel/desired_v)**4 - (desired_gap/dx)**2)
        return self.clip_value(act, 6)

    def clip_value(self, tensor, clip_lim):
        "This is needed to avoid infinities"
        return tf.clip_by_value(tensor, -clip_lim, clip_lim)

    def scale_env_s(self, env_state):
        env_state = (env_state-self.env_scaler.mean_)/self.env_scaler.var_**0.5
        return env_state

    def get_att(self, inputs):
        x = self.att_layer_1(inputs)
        x = self.att_layer_2(x)
        f_att_x = self.clip_value(self.f_att_neu(x), 20)
        m_att_x = self.clip_value(self.m_att_neu(x), 20)
        # f_att_x = self.f_att_neu(x)
        # m_att_x = self.m_att_neu(x)
        # if step == 10:
        #     tf.print('xxxxx f_att_x  min: ', tf.reduce_min(f_att_x))
        #     tf.print('xxxxx f_att_x  max: ', tf.reduce_max(f_att_x))
        #     tf.print('xxxxx m_att_x  min: ', tf.reduce_min(m_att_x))
        #     tf.print('xxxxx m_att_x  max: ', tf.reduce_max(m_att_x))
        f_att_score = tf.exp(f_att_x * self.attention_temp)
        m_att_score = tf.exp(m_att_x * self.attention_temp)
        att_sum = f_att_score + m_att_score
        f_att_score = f_att_score/att_sum
        m_att_score = m_att_score/att_sum
        return f_att_score, m_att_score

    def reshape_idm_params(self, idm_params, batch_size):
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        return tf.split(idm_params, 5, axis=-1)

    def rollout(self, inputs):
        idm_params, proj_latent, enc_h, idm_s, merger_cs = inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = self.reshape_idm_params(idm_params, batch_size)
        proj_latent = tf.reshape(proj_latent, [batch_size, 1, self.proj_dim])
        enc_h = tf.reshape(enc_h, [batch_size, 1, self.dec_units])

        displacement = tf.zeros([batch_size, 1, 1])
        displacement_seq = displacement
        ego_v = idm_s[:,  0:1, 0:1]
        ego_veh_a = idm_s[:, 0:1, 11:12]
        ego_glob_x = idm_s[:,  0:1, 3:4]
        # lstm_output = self.lstm_layer(ego_glob_x)

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
            # tf.print('############ ef_act ############')

            env_state = tf.concat([ego_veh_a, f_veh_a, ego_v, f_veh_v, \
                                    ef_dv, ef_delta_x, em_dv, em_delta_x], axis=-1)
            env_state = self.scale_env_s(env_state)
            merger_c = merger_cs[:, step-1:step, :]

            inputs = tf.concat([proj_latent, enc_h, env_state, merger_c], axis=-1)
            f_att_score, m_att_score = self.get_att(inputs)
            # m_att_score = idm_s[:, step-1:step, -3:-2]
            # f_att_score = 1 - m_att_score
            idm_state = [ego_v, ef_dv, ef_delta_x]
            ef_act = self.idm_driver(idm_state, idm_params)
            idm_state = [ego_v, em_dv, em_delta_x]
            em_act = self.idm_driver(idm_state, idm_params)
            # _act = f_att_score*ef_act + m_att_score*em_act
            ego_veh_a = f_att_score*ef_act + m_att_score*em_act
            displacement += ego_v*0.1 + 0.5*ego_veh_a*0.1**2
            ego_glob_x += ego_v*0.1 + 0.5*ego_veh_a*0.1**2
            ego_v += ego_veh_a*0.1

            if step-1 == 0:
                # first step
                act_seq = ego_veh_a
                f_att_seq = f_att_score
                m_att_seq = m_att_score

            else:
                act_seq = tf.concat([act_seq, ego_veh_a], axis=1)
                f_att_seq = tf.concat([f_att_seq, f_att_score], axis=1)
                m_att_seq = tf.concat([m_att_seq, m_att_score], axis=1)

            displacement_seq = tf.concat([displacement_seq, displacement], axis=1)

        return displacement_seq, act_seq, [f_att_seq, m_att_seq]

class IDMLayer(tf.keras.Model):
    def __init__(self):
        super(IDMLayer, self).__init__(name="IDMLayer")
        self.architecture_def()

    def architecture_def(self):
        self.idm_param_neu = Dense(5)

    def logistic_function(self, x, minval, maxval):
        dif_val = maxval - minval
        return minval + dif_val/(1+tf.exp(-(4/dif_val)*x))

    def get_des_v(self, x):
        minval = 15
        maxval = 25
        return self.logistic_function(x, minval, maxval)

    def get_des_tgap(self, x):
        minval = 0.5
        maxval = 2
        return self.logistic_function(x, minval, maxval)

    def get_min_jamx(self, x):
        minval = 1
        maxval = 5
        return self.logistic_function(x, minval, maxval)

    def get_max_act(self, x):
        minval = 2
        maxval = 4
        return self.logistic_function(x, minval, maxval)

    def get_min_act(self, x):
        minval = 2
        maxval = 4
        return self.logistic_function(x, minval, maxval)

    def call(self, x):
        x = self.idm_param_neu(x)
        x = tf.clip_by_value(x, -100, 100)
        desired_v = self.get_des_v(x[:, 0:1])
        desired_tgap = self.get_des_tgap(x[:, 1:2])
        min_jamx = self.get_min_jamx(x[:, 2:3])
        max_act = self.get_max_act(x[:, 3:4])
        min_act = self.get_min_act(x[:, 4:5])

        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=-1)
        # tf.print('xxxxx min: ', tf.reduce_min(x))
        # # tf.print('xxxxx: max', tf.reduce_max(x))
        # tf.print('idm_param  min: ', tf.reduce_min(x, axis=0))
        # tf.print('idm_param  max: ', tf.reduce_max(x, axis=0))
        # tf.print('idm_param_med: ', tf.reduce_mean(x, axis=0))
        tf.debugging.check_numerics(idm_param, message='Checking idm_param')
        return idm_param
