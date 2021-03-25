from numpy.random import seed # keep this at top
seed(2020)
import numpy as np
import tensorflow as tf
tf.random.set_seed(2020)
from keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, LSTM, TimeDistributed
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from importlib import reload

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.config = config['model_config']
        self.exp_dir = './models/experiments/'+config['exp_id']
        self.optimizer = tf.optimizers.Adam(self.config['learning_rate'])
        self.batch_size = self.config['batch_size']
        self.callback_def()

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def train_loop(self, data_objs):
        """Covers one epoch
        """
        train_ds = self.batch_data(data_objs)
        for s_scaled, s, t in train_ds:
            self.train_step([s_scaled, s], t)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for s_scaled, s, t in train_ds:
            self.test_step([s_scaled, s], t)
        # self.save_epoch_metrics(s, t, epoch)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred, _ = self(states)
            loss = self.mse(targets, act_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred, _ = self(states)
        loss = self.mse(targets, act_pred)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        data_slices = tuple([tf.cast(set, dtype='float32') for set in sets])
        dataset = tf.data.Dataset.from_tensor_slices(\
            data_slices).shuffle(len(data_slices[0])).batch(self.batch_size, drop_remainder=True)
        return dataset

    def mse(self, act_true, act_pred):
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

class Encoder(AbstractModel):
    def __init__(self, config, model_use):
        super(Encoder, self).__init__(config)
        self.enc_units = 50
        self.model_use = model_use # can be training or inference
        self.architecture_def()

    def param_activation(self, batch_size, x, min_val, max_val):
        activation_function = tf.tanh(x)
        scale = tf.fill([batch_size, 1], (max_val-min_val)/2.)
        min_val = tf.fill([batch_size, 1], min_val)
        return tf.add_n([tf.multiply(activation_function, scale), min_val, scale])

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        # # idm params
        # self.neu_desired_v = Dense(1)
        # self.neu_desired_tgap = Dense(1)
        # self.neu_min_jamx = Dense(1)
        # self.neu_act = Dense(1)
        # self.neu_max_act = Dense(1)
        # self.neu_min_act = Dense(1)

        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1)
        self.neu_min_jamx = Dense(1, activation=K.relu)
        # self.neu_min_jamx = Dense(1, activation=K.exp)
        # self.neu_min_jamx = Dense(1)
        self.neu_max_act = Dense(1)
        self.neu_min_act = Dense(1)


    def idm_sim(self, state, h_t):
        # state: [v, dv, dx]
        # print(state.shape)
        if self.model_use == 'training':
            batch_size = 256
        elif self.model_use == 'inference' or self.model_use == 'debug':
            batch_size = 1

        desired_v = self.param_activation(batch_size, self.neu_desired_v(h_t), 15., 35.)
        desired_tgap = self.param_activation(batch_size, self.neu_desired_tgap(h_t), 0.5, 3.)
        min_jamx = self.neu_min_jamx(h_t)
        max_act = self.param_activation(batch_size, self.neu_max_act(h_t), 0.5, 3.)
        min_act = self.param_activation(batch_size, self.neu_min_act(h_t), 0.5, 4.)
        # # tf.print(min_jamx)

        # desired_tgap =  tf.fill([batch_size, 1], 1.5)
        # # min_jamx = self.neu_min_jamx(h_t)
        # min_jamx =  tf.fill([batch_size, 1], 2.)
        # max_act =  tf.fill([batch_size, 1], 1.4)
        # min_act =  tf.fill([batch_size, 1], 2.)
        # tf.print(min_jamx)
        idm_param = tf.concat([desired_v, desired_tgap, min_jamx, max_act, min_act], axis=1)

        if self.model_use == 'training' or self.model_use == 'debug':
            act_seq = tf.zeros([batch_size, 0, 1], dtype=tf.float32)

            for step in tf.range(100):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                                (act_seq, tf.TensorShape([None,None,None]))])

                vel = tf.slice(state, [0, step, 0], [batch_size, 1, 1])
                dv = tf.slice(state, [0, step, 2], [batch_size, 1, 1])
                dx = tf.slice(state, [0, step, 3], [batch_size, 1, 1])
                vel = tf.reshape(vel, [batch_size, 1])
                dv = tf.reshape(dv, [batch_size, 1])
                dx = tf.reshape(dx, [batch_size, 1])

                mult_1 = tf.multiply(max_act, min_act)
                mult_2 = tf.multiply(2., tf.sqrt(mult_1))
                mult_3 = tf.multiply(vel, dv)
                div_1 = tf.divide(mult_3, mult_2)
                mult_4 = tf.multiply(desired_tgap, vel)

                desired_gap = tf.add_n([min_jamx, mult_4, div_1])
                pow_1 = tf.pow(tf.divide(desired_gap, dx), 2.)
                pow_2 = tf.pow(tf.divide(vel, desired_v), 4.)
                subtract_1 = tf.add(pow_2, pow_1)
                subtract_2 = tf.subtract(1., subtract_1)

                act = tf.multiply(max_act, subtract_2)
                act_seq = tf.concat([act_seq, tf.reshape(act, [batch_size, 1, 1])], axis=1)

            return act_seq, idm_param

        elif self.model_use == 'inference':
            return idm_param

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs[0])
        if self.model_use == 'training' or self.model_use == 'debug':
            action, idm_param = self.idm_sim(inputs[1], h_t)
            return action, idm_param

        elif self.model_use == 'inference':
            idm_param = self.idm_sim(inputs[1], h_t)
            return idm_param
