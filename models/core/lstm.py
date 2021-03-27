from tensorflow.keras.layers import Dense, LSTM
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel

class Encoder(AbstractModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.enc_units = 50
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer = LSTM(self.enc_units, return_state=True)
        self.neu_acc = Dense(1)

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.train_step(s, t)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.test_step(s, t)

    def call(self, inputs):
        _, h_t, c_t = self.lstm_layer(inputs)
        return self.neu_acc(h_t)
