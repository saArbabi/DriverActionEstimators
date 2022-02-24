from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
import numpy as np
import tensorflow as tf

class NeuralVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def load_model(self, config, exp_path):
        from models.core.neural  import NeurLatentModel
        self.model = NeurLatentModel(config)
        self.model.load_weights(exp_path).expect_partial()

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        if self.time_lapse_since_last_param_update == 0:
            obs_history = self.scale_state(self.obs_history.copy(), 'full')
            enc_h = self.model.h_seq_encoder(obs_history)
            latent_dis_param = self.model.belief_net(enc_h , dis_type='prior')
            sampled_z = self.model.belief_net.sample_z(latent_dis_param)
            proj_latent = self.model.belief_net.z_proj(sampled_z)
            self.belief_update(proj_latent)
            self.enc_h = tf.reshape(enc_h, [self.samples_n, 1, 128])
        self.time_lapse_since_last_param_update += 1

        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        _context = tf.concat([self.proj_latent, self.enc_h, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)

        act_long = self.model.forward_sim.get_action(_context).numpy()
        return self.action_clip(act_long[0][0][0])
