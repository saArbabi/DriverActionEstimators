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
        self.arbiter = Arbiter()
        self.f_seq_encoder = FutureEncoder()
        self.h_seq_encoder = HistoryEncoder()
        self.act_encoder = FutureEncoder() # sdv's future action
        self.belief_net = BeliefModel()
        self.idm_layer = IDMLayer()
        self.idm_sim = IDMForwardSim()
        # self.vae_loss_weight = 0.01

    def callback_def(self):
        self.train_mseloss = tf.keras.metrics.Mean()
        self.test_mseloss = tf.keras.metrics.Mean()
        self.train_att_klloss = tf.keras.metrics.Mean()
        self.test_att_klloss = tf.keras.metrics.Mean()
        self.train_idm_klloss = tf.keras.metrics.Mean()
        self.test_idm_klloss = tf.keras.metrics.Mean()

    def mse(self, act_true, act_pred):
        # act_true += tf.random.normal(shape=(256, 40, 1), mean=0, stddev=0.6)
        act_true = (act_true+0.2)/0.1
        act_pred = (act_pred+0.2)/0.1
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

    def train_loop(self, data_objs):
        # tf.print('######## TRAIN #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, future_sca, future_idm_s, future_merger_a, future_ego_a in train_ds:
            self.train_step([history_sca, future_sca, future_idm_s, future_merger_a], future_ego_a)

    def test_loop(self, data_objs, epoch):
        # tf.print('######## TEST #######:')
        train_ds = self.batch_data(data_objs)
        for history_sca, future_sca, future_idm_s, future_merger_a, future_ego_a in train_ds:
            self.test_step([history_sca, future_sca, future_idm_s, future_merger_a], future_ego_a)

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
        enc_acts = self.act_encoder(inputs[-1])
        batch_size = tf.shape(inputs[0])[0]

        enc_f = self.f_seq_encoder(inputs[1])
        pri_params, pos_params = self.belief_net(\
                                [enc_h, enc_acts, enc_f], dis_type='both')
        sampled_att_z, sampled_idm_z = self.belief_net.sample_z(pos_params)
        att_scores = self.arbiter([sampled_att_z, enc_h, enc_acts])

        idm_params = self.idm_layer([sampled_idm_z, enc_h])
        # idm_params = tf.repeat(idm_params, 40, axis=1)

        act_seq = self.idm_sim.rollout([att_scores, idm_params, inputs[2]])
        #
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
        att_epsilon = K.random_normal(shape=(tf.shape(z_att_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)

        idm_epsilon = K.random_normal(shape=(tf.shape(z_att_mean)[0],
                                 self.latent_dim), mean=0., stddev=1)
        sampled_att_z = z_att_mean + K.exp(z_att_logsigma) * att_epsilon
        sampled_idm_z = z_idm_mean + K.exp(z_idm_logsigma) * idm_epsilon

        return sampled_att_z, sampled_idm_z

    def call(self, inputs, dis_type):
        if dis_type == 'both':
            enc_h, enc_acts, enc_f = inputs
            # prior
            context_att = self.pri_linear_att(enc_h+enc_acts)
            context_idm = self.pri_linear_idm(enc_h)
            pri_att_mean = self.pri_att_mean(context_att)
            pri_att_logsigma = self.pri_att_logsigma(context_att)
            pri_idm_mean = self.pri_idm_mean(context_idm)
            pri_idm_logsigma = self.pri_idm_logsigma(context_idm)

            # posterior
            context_att = self.pos_linear_att(enc_h+enc_acts+enc_f)
            context_idm = self.pos_linear_idm(enc_h+enc_f)
            pos_att_mean = self.pos_att_mean(context_att)
            pos_att_logsigma = self.pos_att_logsigma(context_att)
            pos_idm_mean = self.pos_idm_mean(context_idm)
            pos_idm_logsigma = self.pos_idm_logsigma(context_idm)

            pri_params = [pri_att_mean, pri_idm_mean, pri_att_logsigma, pri_idm_logsigma]
            pos_params = [pos_att_mean, pos_idm_mean, pos_att_logsigma, pos_idm_logsigma]
            return pri_params, pos_params

        elif dis_type == 'prior':
            enc_h, enc_acts = inputs

            context_att = self.pri_linear_att(enc_h+enc_acts)
            context_idm = self.pri_linear_idm(enc_h)
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
        self.lstm_layer = LSTM(self.enc_units)
        # self.masking = Masking()
    def call(self, inputs):
        # enc_h = self.lstm_layer(self.masking(inputs))
        enc_h = self.lstm_layer(inputs)
        return enc_h

class FutureEncoder(tf.keras.Model):
    def __init__(self):
        super(FutureEncoder, self).__init__(name="FutureEncoder")
        self.enc_units = 100
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = Bidirectional(LSTM(self.enc_units), merge_mode='sum')

    def call(self, inputs):
        enc_acts = self.lstm_layer(inputs)
        return enc_acts

class Arbiter(tf.keras.Model):
    def __init__(self):
        super(Arbiter, self).__init__(name="Arbiter")
        self.attention_temp = 5 # the higher, the sharper the attention
        self.architecture_def()

    def architecture_def(self):
        self.linear_layer = Dense(100)
        self.attention_neu = Dense(40)

    def call(self, inputs):
        sampled_att_z, enc_h, enc_acts = inputs
        x = self.linear_layer(sampled_att_z) + enc_h + enc_acts
        # x = self.linear_layer(sampled_att_z) + enc_h
        # x = self.linear_layer(sampled_att_z)
        x = self.attention_neu(x)
        x = tf.clip_by_value(x, clip_value_min=-3., clip_value_max=3.)
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
        # tf.print('desired_v: ', tf.reduce_mean(desired_v))
        # tf.print('desired_v_max: ', tf.reduce_max(desired_v))
        # tf.print('desired_v_min: ', tf.reduce_min(desired_v))
        # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        # tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
        # tf.print('max_act: ', tf.reduce_mean(max_act))
        # tf.print('min_act: ', tf.reduce_mean(min_act))
        desired_gap = min_jamx + K.relu(desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act)))



        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)

        return self.action_clip(act)
        # return act

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3., clip_value_max=3.)

    @tf.function(experimental_relax_shapes=True)
    def rollout(self, inputs):
        att_scores, idm_params, idm_s = inputs
        batch_size = tf.shape(idm_s)[0]
        idm_params = tf.reshape(idm_params, [batch_size, 1, 5])
        att_scores = tf.reshape(att_scores, [batch_size, 40, 1])

        for step in range(40):
            leader_v = idm_s[:, step:step+1, 1:2]
            merger_v = idm_s[:, step:step+1, 2:3]
            leader_glob_x = idm_s[:, step:step+1, 4:5]
            merger_glob_x = idm_s[:, step:step+1, 5:6]
            # these to deal with missing cars
            leader_exists = idm_s[:, step:step+1, -2:-1]
            merger_exists = idm_s[:, step:step+1, -1:]
            if step == 0:
                ego_v = idm_s[:, step:step+1, 0:1]
                ego_glob_x = idm_s[:, step:step+1, 3:4]
            else:
                ego_v += _act*0.1
                ego_glob_x += ego_v*0.1 + 0.5*_act*0.1**2

            fl_delta_x = leader_glob_x - ego_glob_x
            fm_delta_x = merger_glob_x - ego_glob_x
            fl_dv = ego_v - leader_v
            fm_dv = ego_v - merger_v

            dv = fl_dv*leader_exists
            dx = fl_delta_x*leader_exists + 1000*(1-leader_exists)
            # tf.print('############ fl_act ############')
            fl_act = self.idm_driver(ego_v, dv, dx, idm_params)

            dv = fm_dv*merger_exists
            dx = fm_delta_x*merger_exists + 1000*(1-merger_exists)
            # tf.print('############ fm_act ############')
            fm_act = self.idm_driver(ego_v, dv, dx, idm_params)

            att_score = att_scores[:, step:step+1, :]
            _act = (1-att_score)*fl_act + att_score*fm_act
            if step == 0:
                act_seq = _act
            else:
                act_seq = tf.concat([act_seq, _act], axis=1)
            # act_seq = act_seq.write(step, _act)

        return act_seq

class IDMLayer(tf.keras.Model):
    def __init__(self):
        super(IDMLayer, self).__init__(name="IDMLayer")
        self.enc_units = 100
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
        # return  10 + 30*(1/(1+tf.exp(-1*output)))
        return 15 + 15*(1/(1+tf.exp(-1*output)))

    def get_des_tgap(self, x, batch_size):
        output = self.des_tgap_neu(x)
        return 1 + 1*(1/(1+tf.exp(-1*output)))

    def get_min_jamx(self, x, batch_size):
        output = self.min_jamx_neu(x)
        return 4*(1/(1+tf.exp(-1*output)))

    def get_max_act(self, x, batch_size):
        output = self.max_act_neu(x)
        return 0.8 + 1.2*(1/(1+tf.exp(-1*output)))

    def get_min_act(self, x, batch_size):
        output = self.min_act_neu(x)
        return 1 + 2*(1/(1+tf.exp(-1*output)))

    def call(self, inputs):
        sampled_idm_z, enc_h = inputs
        batch_size = tf.shape(sampled_idm_z)[0]

        # x = enc_h
        x = self.linear_layer(sampled_idm_z) + enc_h
        # x = self.linear_layer(sampled_idm_z)
        # desired_v = tf.fill([batch_size, 1], 19.4)
        # desired_tgap = tf.fill([batch_size, 1], 2.0)
        # min_jamx = tf.fill([batch_size, 1], 4.0)
        # max_act = tf.fill([batch_size, 1], 0.8)
        # min_act = tf.fill([batch_size, 1], 1.0)

        # desired_v = tf.fill([batch_size, 1], 24.7)
        # desired_tgap = tf.fill([batch_size, 1], 1.5)
        # min_jamx = tf.fill([batch_size, 1], 2.0)
        # max_act = tf.fill([batch_size, 1], 1.4)
        # min_act = tf.fill([batch_size, 1], 2.0)
        # #
        desired_v = self.get_des_v(x, batch_size)
        desired_tgap = self.get_des_tgap(x, batch_size)
        min_jamx = self.get_min_jamx(x, batch_size)
        max_act = self.get_max_act(x, batch_size)
        min_act = self.get_min_act(x, batch_size)
        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=-1)
        return idm_param
