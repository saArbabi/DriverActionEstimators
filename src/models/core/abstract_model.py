import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# print(physical_devices)

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.callback_def()
        if config:
            self.config = config['model_config']
            self.optimizer = tf.optimizers.Adam(self.config['learning_rate'])
            self.batch_size = self.config['batch_size']

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def mse(self, act_true, act_pred):
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for xs_h, xs_f, ys_f in train_ds:
            self.train_step([xs_h, xs_f], ys_f)

    def test_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for xs_h, xs_f, ys_f in train_ds:
            self.test_step([xs_h, xs_f], ys_f)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            act_pred = self(states)
            loss = self.mse(targets, act_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.reset_states()
        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        act_pred = self(states)
        loss = self.mse(targets, act_pred)
        self.test_loss.reset_states()
        self.test_loss(loss)

    def batch_data(self, sets, batch_size):
        data_slices = tuple([tf.cast(set, dtype='float32') for set in sets])
        dataset = tf.data.Dataset.from_tensor_slices(\
            data_slices).shuffle(\
                len(data_slices[0]), reshuffle_each_iteration=True).batch(batch_size)
        return dataset

    def mse(self, act_true, act_pred):
        return tf.reduce_mean((tf.square(tf.subtract(act_pred, act_true))))
