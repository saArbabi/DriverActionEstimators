
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Masking
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class NeurIDMModel(AbstractModel):
    def __init__(self, config=None):
        super(NeurIDMModel, self).__init__(config)
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.act_encoder = FutureEncoder() # sdv's future action
        self.belief_net = BeliefModel()
        self.idm_layer = IDMLayer()
        self.forward_sim = IDMForwardSim()
        self.vae_loss_weight = 0.1 # default
        # self.loss_function = tf.keras.losses.Huber()

    def callback_def(self):
        self.train_mseloss = tf.keras.metrics.Mean()
        self.test_mseloss = tf.keras.metrics.Mean()
        self.train_klloss = tf.keras.metrics.Mean()
        self.test_klloss = tf.keras.metrics.Mean()

    def mse(self, act_true, act_pred):
        act_true = (act_true)/0.1
        act_pred = (act_pred)/0.1
        # return self.loss_function(act_true, act_pred)
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

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

        idm_params = self.idm_layer(sampled_z)
        # idm_params = tf.repeat(idm_params, 40, axis=1)
        act_seq, _ = self.forward_sim.rollout([sampled_z, idm_params, inputs[2], inputs[-1]])
        # tf.print('###############:')
        # tf.print('att_score_max: ', tf.reduce_max(att_scores))
        # tf.print('att_score_min: ', tf.reduce_min(att_scores))
        # tf.print('att_score_mean: ', tf.reduce_mean(att_scores))

        return act_seq, pri_params, pos_params

class BeliefModel(tf.keras.Model):
    def __init__(self):
        super(BeliefModel, self).__init__(name="BeliefModel")
        self.latent_dim = 3
        self.architecture_def()

    def architecture_def(self):
        self.pri_mean = Dense(self.latent_dim)
        self.pri_logsigma = Dense(self.latent_dim)
        self.pos_mean = Dense(self.latent_dim)
        self.pos_logsigma = Dense(self.latent_dim)

        self.pri_linear = Dense(100)
        self.pos_linear = Dense(100)

    def sample_z(self, dis_params):
        z_mean, z_logsigma = dis_params
        _epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        sampled_z = z_mean + K.exp(z_logsigma) * _epsilon
        return sampled_z
        # return sampled_z, sampled_z for single latent

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_f = inputs
            # prior
            context = self.pri_linear(enc_h)
            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)

            # posterior
            pos_context = tf.concat([enc_h, enc_f], axis=-1)
            context = self.pos_linear(pos_context)
            pos_mean = self.pos_mean(context)
            pos_logsigma = self.pos_logsigma(context)

            pri_params = [pri_mean, pri_logsigma]
            pos_params = [pos_mean, pos_logsigma]
            return pri_params, pos_params

        elif dis_type == 'prior':
            context = self.pri_linear(inputs)
            pri_mean = self.pri_mean(context)
            pri_logsigma = self.pri_logsigma(context)
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
        # two = K.relu(desired_tgap*vel+(vel*dv)/ \
        #                                 (2*tf.sqrt(max_act*min_act)))
        desired_gap = min_jamx + K.relu(desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act)))
        # tf.print('min: ', tf.reduce_min(two))
        # tf.print('mean: ', tf.reduce_mean(two))

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
                (1-idm_veh_exists)*tf.random.normal((batch_size, 1, 1), 0, 0.5)
        return idm_action

    def transition_dynamic(self, current_ego_state, current_ego_action):
        """
        Two things to figure out:
        should you start from step 0?
        Why balancing data is not helping
        Use LSTMs. But ensure any ego state is ego generated - a true rollout if you like
        """
        return next_ego_state

    def scale_features(self, env_state):
        env_state = (env_state-self.scaler.mean_)/self.scaler.var_**0.5
        return env_state

    def rollout(self, inputs):
        sampled_z, idm_params, idm_s, sdv_acts = inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        latent_projection = self.linear_layer(sampled_z)
        proj_latent  = tf.reshape(latent_projection, [batch_size, 1, 100])
        state_h, state_c = latent_projection, latent_projection

        for step in range(60):
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
            # tf.print('############ ef_act ############')
            ef_act = self.idm_driver(ego_v, ef_dv, ef_delta_x, idm_params)
            ef_act = self.add_noise(ef_act, f_veh_exists, batch_size)

            # tf.print('############ em_act ############')
            # tf.Assert(tf.greater(tf.reduce_min(em_delta_x), 0.),[em_delta_x])
            # tf.Assert(tf.greater(tf.reduce_min(ef_delta_x), 0.),[ef_delta_x])
            em_act = self.idm_driver(ego_v, em_dv, em_delta_x, idm_params)
            em_act = self.add_noise(em_act, m_veh_exists, batch_size)

            sdv_act = sdv_acts[:, step:step+1, :]
            lstm_output, state_h, state_c = self.lstm_layer(tf.concat([\
                                    proj_latent, sdv_act], axis=-1), \
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
        self.des_v_neu = Dense(1)
        self.des_v_linear = Dense(100)
        self.des_tgap_neu = Dense(1)
        self.des_tgap_linear = Dense(100)
        self.min_jamx_neu = Dense(1)
        self.min_jamx_linear = Dense(100)
        self.max_act_neu = Dense(1)
        self.max_act_linear = Dense(100)
        self.min_act_neu = Dense(1)
        self.min_act_linear = Dense(100)

    def param_activation(self, x, min_val, max_val, batch_size):
        activation_function = tf.tanh(0.5*x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def get_des_v(self, x):
        output = self.des_v_neu(self.des_v_linear(x))
        minval = 15
        maxval = 35
        return minval + (maxval-minval)/(1+tf.exp(-1.*output))

    def get_des_tgap(self, x):
        output = self.des_tgap_neu(self.des_tgap_linear(x))
        return tf.exp(output)

    def get_min_jamx(self, x):
        output = self.min_jamx_neu(self.min_jamx_linear(x))
        return tf.exp(output)

    def get_max_act(self, x):
        output = self.max_act_neu(self.max_act_linear(x))
        return tf.exp(output)

    def get_min_act(self, x):
        output = self.min_act_neu(self.min_act_linear(x))
        return tf.exp(output)

    def call(self, x):
        desired_v = self.get_des_v(x)
        desired_tgap = self.get_des_tgap(x)
        min_jamx = self.get_min_jamx(x)
        max_act = self.get_max_act(x)
        min_act = self.get_min_act(x)
        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=-1)
        return idm_param
#
# class IDMForwardSimLaneKeep(IDMForwardSim):
#     def __init__(self):
#         super().__init__()
#
#     def rollout(self, inputs):
#         att_inputs, idm_params, idm_s, sdv_acts = inputs
#         sampled_att_z, enc_h = att_inputs
#         batch_size = tf.shape(idm_s)[0]
#         idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
#         for step in range(40):
#             f_veh_v = idm_s[:, step:step+1, 1:2]
#             f_veh_glob_x = idm_s[:, step:step+1, 3:4]
#
#             if step == 0:
#                 ego_v = idm_s[:, step:step+1, 0:1]
#                 ego_glob_x = idm_s[:, step:step+1, 2:3]
#             else:
#                 ego_v += ef_act*0.1
#                 ego_glob_x += ego_v*0.1 + 0.5*ef_act*0.1**2
#
#             ef_delta_x = (f_veh_glob_x - ego_glob_x)
#             ef_dv = (ego_v - f_veh_v)
#             tf.Assert(tf.greater(tf.reduce_min(ef_delta_x), 0.),[ef_delta_x])
#             ef_act = self.idm_driver(ego_v, ef_dv, ef_delta_x, idm_params)
#             if step == 0:
#                 act_seq = ef_act
#             else:
#                 act_seq = tf.concat([act_seq, ef_act], axis=1)
#
#         return act_seq, act_seq

# class NeurIDMModelLaneKeep(NeurIDMModel):
#     def __init__(self, config=None):
#         super().__init__(config)
#         self.forward_sim = IDMForwardSimLaneKeep()
