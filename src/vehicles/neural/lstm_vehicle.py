from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
import numpy as np
import tensorflow as tf

class LSTMVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def load_model(self, config, exp_path):
        from models.core.lstm import LSTMEncoder
        self.model = LSTMEncoder(config)
        self.model.load_weights(exp_path).expect_partial()

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        obs_history = self.scale_state(self.obs_history.copy(), 'full')

        pred_dis = self.model(obs_history)
        act_long = pred_dis.sample().numpy()
        return act_long[0][0]
