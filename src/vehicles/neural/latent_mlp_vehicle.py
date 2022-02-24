from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class LatentMLPVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def load_model(self, config, exp_path):
        from models.core.latent_mlp import LatentMLP
        self.model = LatentMLP(config)
        self.model.load_weights(exp_path).expect_partial()

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        if self.time_lapse_since_last_param_update == 0:
            obs_history = self.scale_state(self.obs_history.copy(), 'full')
            enc_h = self.model.h_seq_encoder(obs_history)
            pos_params = self.model.belief_net(enc_h)
            sampled_z = self.model.belief_net.sample_z(pos_params)
            sampled_z = tf.reshape(\
                            sampled_z, [1, 1, self.model.belief_net.latent_dim])
            self._latent = sampled_z
        self.time_lapse_since_last_param_update += 1

        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        _context = tf.concat([self._latent, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)

        _mean, _var = self.model.forward_sim.get_dis(_context)
        act_long = tfd.Normal(_mean, _var, name='Normal').sample().numpy()
        return act_long[0][0][0]
