from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
import numpy as np
import tensorflow as tf

class MLPVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def load_model(self, config, exp_path):
        from models.core.mlp import MLP
        self.model = MLP(config)
        self.model.load_weights(exp_path).expect_partial()

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        _context = tf.concat([env_state, merger_c, m_veh_exists], axis=-1)

        pred_dis = self.model(_context[:, 0, :])
        act_long = pred_dis.sample().numpy()
        return act_long[0][0]
