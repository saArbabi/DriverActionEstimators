from tensorflow.keras.layers import Dense, LSTM
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
tf.random.set_seed(1234)

class NeurIDMModel(AbstractModel):
    def __init__(self, config, model_use):
        super(NeurIDMModel, self).__init__(config)
        self.encoder = Encoder()
        self.decoder = Decoder(config)
        self.idm_sim = IDMForwardSim()
        self.idm_layer = IDMLayer()
        self.model_use = model_use
        # mse_loss = 0
        # self.kl_loss = 0

    def callback_def(self):
        self.train_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mseloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_klloss = tf.keras.metrics.Mean(name='train_loss')
        self.test_mseloss = tf.keras.metrics.Mean(name='train_loss')

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, ys_f in train_ds:
            self.train_step([xs_h, scaled_xs_f, unscaled_xs_f], ys_f)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for xs_h, scaled_xs_f, unscaled_xs_f, ys_f in train_ds:
            self.test_step([xs_h, scaled_xs_f, unscaled_xs_f], ys_f)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, mean, logvar = self(states)
            mse_loss = self.mse(targets, act_pred)
            kl_loss = self.kl_loss(mean, logvar)
            loss = self.vae_loss(mse_loss, kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_mseloss.reset_states()
        self.train_klloss.reset_states()
        self.train_mseloss(mse_loss)
        self.train_klloss(kl_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, mean, logvar = self(states)
        mse_loss = self.mse(targets, act_pred)
        kl_loss = self.kl_loss(mean, logvar)
        loss = self.vae_loss(mse_loss, kl_loss)
        self.test_klloss.reset_states()
        self.test_mseloss.reset_states()
        self.test_mseloss(mse_loss)
        self.test_klloss(kl_loss)
    # @tf.function
    def sample(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0], self.encoder.latent_dim),
                                  mean=0., stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_actions(self):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        desired_v = self.decoder(z)
        a_ = self.idm(desired_v)
        return  a_

    def kl_loss(self, z_mean, z_log_sigma):
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(kl_loss)

    def vae_loss(self, mse_loss, kl_loss):
        return  kl_loss +  mse_loss

    def call(self, inputs):
        # inputs: [xs_h, scaled_xs_f, unscaled_xs_f]
        current_v = inputs[2][:, 0, 2:3]
        mean, logvar, encoder_states = self.encoder(inputs[0])
        z = self.sample([mean, logvar])
        decoder_output = self.decoder(z)
        idm_param = self.idm_layer([decoder_output, current_v])

        if self.model_use == 'training':
            act_seq = self.idm_sim.rollout([inputs[1:], idm_param, encoder_states])
            return act_seq, mean, logvar

        elif self.model_use == 'inference':
            h_t, c_t = encoder_states
            att_score, _, _ = self.idm_sim.arbiter([inputs[1][:, 0:1, :], h_t, c_t])
            return idm_param, att_score

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__(name="Encoder")
        self.enc_units = 50
        self.latent_dim = 2
        # TODO: FULL cov distribution
        # self.prior = tfd.Normal(loc=0., scale=3.)
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.z_mean = Dense(self.latent_dim)
        self.z_log_sigma = Dense(self.latent_dim)
        #
        # tfpl.MultivariateNormalTriL(self.latent_dim,
        #     activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=1.0)),

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs)
        z_mean = self.z_mean(h_t)
        z_log_sigma = self.z_log_sigma(h_t)
        return z_mean, z_log_sigma, [h_t, c_t]

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__(name="Decoder")
        self.architecture_def()

    def architecture_def(self):
        self.layer_1 = Dense(10, activation=K.relu)
        self.layer_2 = Dense(60, activation=K.relu)
        self.layer_3 = Dense(60, activation=K.relu)
        self.layer_4 = Dense(10, activation=K.relu)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_4(x)

class Arbiter(tf.keras.Model):
    def __init__(self):
        super(Arbiter, self).__init__(name="Arbiter")
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.future_dec = LSTM(self.enc_units, return_sequences=True, return_state=True)
        self.attention_layer = Dense(self.enc_units)
        self.attention_neu = Dense(1)

    def call(self, inputs):
        scaled_s, h_t, c_t = inputs
        batch_size = tf.shape(scaled_s)[0]
        outputs, h_t, c_t = self.future_dec(scaled_s, initial_state=[h_t, c_t])
        outputs = tf.reshape(outputs, [batch_size, self.enc_units])
        x = self.attention_layer(outputs)
        x = self.attention_neu(x)

        return 1/(1+tf.exp(-5*x)), h_t, c_t

class IDMForwardSim(tf.keras.Model):
    def __init__(self):
        super(IDMForwardSim, self).__init__(name="IDMForwardSim")
        self.arbiter = Arbiter()

    def idm_driver(self, vel, dv, dx, idm_param):
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param
        desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                        (2*tf.sqrt(max_act*min_act))

        act = max_act*(1-(vel/desired_v)**4-\
                                            (desired_gap/dx)**2)
        return self.action_clip(act)

    def action_clip(self, action):
        "this helps with avoiding vanishing gradients"
        return tf.clip_by_value(action, clip_value_min=-3.5, clip_value_max=3.5)

    def rollout(self, inputs):
        env_states, idm_param, encoder_states = inputs
        desired_v, desired_tgap, min_jamx, max_act, min_act = idm_param

        h_t, c_t = encoder_states
        scaled_s, unscaled_s = env_states
        batch_size = tf.shape(scaled_s)[0]

        act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        fl_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        fm_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)
        att_scores = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

        for step in tf.range(20):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                            (att_scores, tf.TensorShape([None,None,None])),
                            (fl_seq, tf.TensorShape([None,None,None])),
                            (fm_seq, tf.TensorShape([None,None,None])),
                            (act_seq, tf.TensorShape([None,None,None]))
                             ])


            # att_score, fm_att_score = self.get_att_score(tf.concat([fl_act, fm_act, outputs], axis=1))
            vel = tf.slice(unscaled_s, [0, step, 0], [batch_size, 1, 1])
            vel = tf.reshape(vel, [batch_size, 1])

            dv = tf.slice(unscaled_s, [0, step, 2], [batch_size, 1, 1])
            dx = tf.slice(unscaled_s, [0, step, 3], [batch_size, 1, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fl_act = self.idm_driver(vel, dv, dx, idm_param)

            dv = tf.slice(unscaled_s, [0, step, 5], [batch_size, 1, 1])
            dx = tf.slice(unscaled_s, [0, step, 6], [batch_size, 1, 1])
            dv = tf.reshape(dv, [batch_size, 1])
            dx = tf.reshape(dx, [batch_size, 1])
            fm_act = self.idm_driver(vel, dv, dx, idm_param)


            att_score, h_t, c_t = self.arbiter([scaled_s[:, step:step+1, :], h_t, c_t, ])

            act = att_score*fl_act + (1-att_score)*fm_act
            # act = att_score*fl_act + (1-att_score)*fm_act
            # act = att_score*fl_act
            # act = fl_act
            att_scores = tf.concat([att_scores, tf.reshape(att_score, [batch_size, 1, 1])], axis=1)
            act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)
            fl_seq = tf.concat([fl_seq, tf.reshape(fl_act, [batch_size, 1, 1])], axis=1)
            fm_seq = tf.concat([fm_seq, tf.reshape(fm_act, [batch_size, 1, 1])], axis=1)

        # tf.print('######')
        # tf.print('desired_v: ', tf.reduce_mean(desired_v))
        # tf.print('desired_tgap: ', tf.reduce_mean(desired_tgap))
        # tf.print('min_jamx: ', tf.reduce_mean(min_jamx))
        # tf.print('max_act: ', tf.reduce_mean(max_act))
        # tf.print('min_act: ', tf.reduce_mean(min_act))
        # tf.print('att_score_max: ', tf.reduce_max(att_scores))
        # tf.print('att_score_min: ', tf.reduce_min(att_scores))
        # tf.print('att_score_mean: ', tf.reduce_mean(att_scores))

        return act_seq

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

    def get_des_v(self, x, current_v):
        # input = self.des_v_layer(x)
        output = self.des_v_neu(x) + current_v
        return output

    def get_des_tgap(self, x):
        # input = self.des_tgap_layer(x)
        output = tf.exp(self.des_tgap_neu(x)) + 1
        return output

    def get_min_jamx(self, x):
        # input = self.min_jamx_layer(x)
        output = tf.exp(self.min_jamx_neu(x)) + 1
        return output

    def get_max_act(self, x):
        # input = self.max_act_layer(x)
        output = tf.exp(self.max_act_neu(x)) + 0.5
        return output

    def get_min_act(self, x):
        # input = self.min_act_layer(x)
        output = tf.exp(self.min_act_neu(x)) + 0.5
        return output

    def call(self, inputs):
        # _, h_t, c_t = self.histroy_enc(inputs[0])
        decoder_output, current_v = inputs
        desired_v = self.get_des_v(decoder_output, current_v)
        desired_tgap = self.get_des_tgap(decoder_output)
        min_jamx = self.get_min_jamx(decoder_output)
        max_act = self.get_max_act(decoder_output)
        min_act = self.get_min_act(decoder_output)
        idm_param = [desired_v, desired_tgap, min_jamx, max_act, min_act]
        return idm_param
