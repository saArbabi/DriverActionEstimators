import numpy as np
np.random.seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from datetime import datetime
from models.core.tf_models.utils import loss_other, loss_other, covDet_mean
from tensorflow.keras.layers import Masking

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
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.exp_dir+'/logs/'
        self.writer_1 = tf.summary.create_file_writer(log_dir+'epoch_loss')
        self.writer_2 = tf.summary.create_file_writer(log_dir+'epoch_metrics')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def save_epoch_metrics(self, states, targets, conditions, epoch):
        with self.writer_1.as_default():
            tf.summary.scalar('_train', self.train_loss.result(), step=epoch)
            tf.summary.scalar('_val', self.test_loss.result(), step=epoch)
        self.writer_1.flush()

        with self.writer_2.as_default():
            gmm_mlon, gmm_mlat, gmm_y, gmm_f, gmm_fadj = self([states, conditions], training=True)
            # covDet = covDet_mean(gmm_mlon)
            # tf.summary.scalar('covDet_mean', covDet, step=epoch)
            tf.summary.scalar('loss_mlong', loss_other(targets[0], gmm_mlon), step=epoch)
            tf.summary.scalar('loss_mlat', loss_other(targets[1], gmm_mlat), step=epoch)
            tf.summary.scalar('loss_y', loss_other(targets[2], gmm_y), step=epoch)
            tf.summary.scalar('loss_f', loss_other(targets[3], gmm_f), step=epoch)
            tf.summary.scalar('loss_fadj', loss_other(targets[4], gmm_fadj), step=epoch)
        self.writer_2.flush()

    def train_loop(self, data_objs):
        """Covers one epoch
        """
        for seq_len in range(1, self.pred_step_n+1):
            train_seq_data = [data_objs[0][seq_len],
                                data_objs[1][seq_len][0],
                                data_objs[1][seq_len][1],
                                data_objs[1][seq_len][2],
                                data_objs[1][seq_len][3],
                                data_objs[1][seq_len][4],
                                data_objs[2][seq_len][0],
                                data_objs[2][seq_len][1],
                                data_objs[2][seq_len][2],
                                data_objs[2][seq_len][3],
                                data_objs[2][seq_len][4]]

            train_ds = self.batch_data(train_seq_data)
            for s, t0, t1, t2, t3, t4, c0, c1, c2, c3, c4 in train_ds:
                self.train_step(s, [t0, t1, t2, t3, t4], [c0, c1, c2, c3, c4])

    def test_loop(self, data_objs, epoch):
        for seq_len in range(1, self.pred_step_n+1):
            test_seq_data = [data_objs[0][seq_len],
                                data_objs[1][seq_len][0],
                                data_objs[1][seq_len][1],
                                data_objs[1][seq_len][2],
                                data_objs[1][seq_len][3],
                                data_objs[1][seq_len][4],
                                data_objs[2][seq_len][0],
                                data_objs[2][seq_len][1],
                                data_objs[2][seq_len][2],
                                data_objs[2][seq_len][3],
                                data_objs[2][seq_len][4]]

            test_ds = self.batch_data(test_seq_data)
            for s, t0, t1, t2, t3, t4, c0, c1, c2, c3, c4 in test_ds:
                self.test_step(s, [t0, t1, t2, t3, t4], [c0, c1, c2, c3, c4])
        self.save_epoch_metrics(s, [t0, t1, t2, t3, t4], [c0, c1, c2, c3, c4], epoch)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets, conditions):
        with tf.GradientTape() as tape:
            gmm_mlon, gmm_mlat, gmm_y, gmm_f, gmm_fadj = self([states, conditions])

            loss = loss_other(targets[0], gmm_mlon) + \
                    loss_other(targets[1], gmm_mlat) + \
                    loss_other(targets[2], gmm_y) + \
                    loss_other(targets[3], gmm_f) + \
                    loss_other(targets[4], gmm_fadj)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets, conditions):
        gmm_mlon, gmm_mlat, gmm_y, gmm_f, gmm_fadj = self([states, conditions])
        loss = loss_other(targets[0], gmm_mlon) + \
                loss_other(targets[1], gmm_mlat) + \
                loss_other(targets[2], gmm_y) + \
                loss_other(targets[3], gmm_f) + \
                loss_other(targets[4], gmm_fadj)

        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets):
        data_slices = tuple([tf.cast(set, dtype='float32') for set in sets])
        dataset = tf.data.Dataset.from_tensor_slices(data_slices).batch(self.batch_size)
        return dataset
