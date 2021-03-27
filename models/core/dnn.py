from tensorflow.keras.layers import Dense
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel

class Encoder(AbstractModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.architecture_def()

    def architecture_def(self):
        self.layer_1 = Dense(10, activation=K.relu)
        self.layer_2 = Dense(60, activation=K.relu)
        self.layer_3 = Dense(60, activation=K.relu)
        self.layer_4 = Dense(60, activation=K.relu)
        self.layer_out = Dense(1)

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.train_step(s, t)

    def test_loop(self, data_objs, epoch):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.test_step(s, t)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return self.layer_out(x)
