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
            latent_dis_param = self.model.belief_net(enc_h, dis_type='prior')
            sampled_z = self.model.belief_net.sample_z(latent_dis_param)
            proj_belief = self.model.belief_net.belief_proj(sampled_z)
            self.belief_update(proj_belief)
        self.time_lapse_since_last_param_update += 1

        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        _context = tf.concat([self.proj_latent, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)

        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                                    _context, initial_state=[self.state_h, self.state_c])
        act_long = self.model.forward_sim.action_neu(lstm_output).numpy()
        return act_long[0][0][0]
