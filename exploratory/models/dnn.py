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
        self.architecture_def()

    def architecture_def(self):
        self.layer_1 = Dense(10, activation=K.relu)
        self.layer_2 = Dense(30, activation=K.relu)
        self.layer_3 = Dense(30, activation=K.relu)
        self.layer_out = Dense(1)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_out(x)
