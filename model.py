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
        self.pred_step_n = config['data_config']['pred_step_n']
        self.batch_count = None
        self.epochs_n = self.config['epochs_n']
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
        for s, t in train_ds:
            self.train_step(s, t)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.test_step(s, t)
        # self.save_epoch_metrics(s, t, epoch)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            pred_acc = self(states)
            loss = tf.keras.losses.MSE(targets, pred_acc)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        pred_acc = self(states)
        loss = tf.keras.losses.MSE(targets, pred_acc)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        data_slices = tuple([tf.cast(set, dtype='float32') for set in sets])
        dataset = tf.data.Dataset.from_tensor_slices(\
                                data_slices).batch(self.batch_size, drop_remainder=True)
        return dataset


class Encoder(AbstractModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.enc_units = 50

        self.architecture_def()
        self.default_config()

    def default_config(self):
        # TODO nonstationary params
        self.idm_param = {
                        # 'desired_v':self.v, # m/s
                        'desired_tgap':2.8, # s
                        'min_jamx':0., # m
                         'max_acc':3., # m/s^2
                        'max_decc':3., # m/s^2
                        }

        # self.desired_v = self.idm_param['desired_v']
        self.min_jamx = tf.fill([self.batch_size, 1], self.idm_param['min_jamx'])
        self.max_acc = tf.fill([self.batch_size, 1], self.idm_param['max_acc'])
        self.max_decc = tf.fill([self.batch_size, 1], self.idm_param['max_decc'])

    def param_activation(self, x, min_val, max_val):
        activation_function = tf.add(tf.tanh(x), 1)
        scale = (max_val-min_val)/2.
        return tf.add(tf.multiply(activation_function, scale), min_val)

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.linear_layer_1 = Dense(50, activation=K.relu)
        self.linear_layer_2 = Dense(50, activation=K.relu)
        self.linear_layer_3 = Dense(50, activation=K.relu)

        # idm params
        self.neu_desired_v = Dense(1)
        self.neu_desired_tgap = Dense(1)
        # self.desired_tgap = Dense(1, activation=self.param_activation(, ))
        # self.min_jamx = Dense(1, activation=self.param_activation(, ))
        # self.min_jamx = Dense(1, activation=self.param_activation(, ))
        self.neu_acc = Dense(1)
        # self.neu_acc = Dense(1, activation=K.tanh)
        # self.neu_acc = Dense(1, activation=self.param_activation(-3, 3))

    def idm_sim(self, state, h_t):
        # state: [v, dv, dx]
        # print(state.shape)
        batch_size = 50 # dynamiclaly assigned
        # tf.print(state)

        vel = tf.slice(state, [0, 29, 0], [batch_size, 1, 1])
        dv = tf.slice(state, [0, 29, 1], [batch_size, 1, 1])
        dx = tf.slice(state, [0, 29, 2], [batch_size, 1, 1])
        vel = tf.reshape(vel, [batch_size, 1])
        dv = tf.reshape(dv, [batch_size, 1])
        dx = tf.reshape(dx, [batch_size, 1])
        # print(vel.shape)

        # desired_v = self.neu_desired_v(h_t)
        desired_v = self.param_activation(self.neu_desired_v(h_t), 5, 50)
        desired_tgap = self.param_activation(self.neu_desired_tgap(h_t), 1, 3)
        # print(h_t.shape)
        # desired_v = tf.fill([self.batch_size, 1], 15.)
        self.desired_tgap = tf.fill([self.batch_size, 1], self.idm_param['desired_tgap'])

        mult_1 = tf.multiply(self.max_acc, self.max_decc)
        mult_2 = tf.multiply(2., tf.sqrt(mult_1))
        mult_3 = tf.multiply(vel, dv)
        div_1 = tf.divide(mult_3, mult_2)
        mult_4 = tf.multiply(desired_tgap, vel)
        desired_gap = tf.add_n([self.min_jamx, mult_4, div_1])
        ###
        pow_1 = tf.pow(tf.divide(desired_gap, dx), 2.)
        pow_2 = tf.pow(tf.divide(vel, desired_v), 4.)
        subtract_1 = tf.add(pow_2, pow_1)
        subtract_2 = tf.subtract(1., subtract_1)
        acc = tf.multiply(self.max_acc, subtract_2)
        # acc = self.param_activation(acc, -3, 3)

        # acc = tf.reshape(self.neu_acc(acc), [batch_size, 1])
                # tf.print(h_t)
        # tf.print(desired_v)
        # tf.print(acc)
        # tf.print(state)
        tf.print(desired_v)
        # tf.print(acc)

        return acc

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs)
        h_t = self.linear_layer_1(h_t)
        h_t = self.linear_layer_2(h_t)
        h_t = self.linear_layer_3(h_t)
        return self.idm_sim(inputs, h_t)
        # # return self.neu_acc(h_t)
        # acc = self.param_activation(self.neu_acc(h_t), -3, 3)
        # # tf.print(acc)
        #
        # return acc
