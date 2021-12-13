from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
import numpy as np
import pickle
from importlib import reload
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import json

class LatentMLPVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def load_model(self, config, exp_path):
        from models.core.latent_mlp import LatentMLP
        self.model = LatentMLP(config)
        self.model.load_weights(exp_path).expect_partial()

    def initialize_agent(self, model_name, epoch_count, data_id):
        exp_dir = './src/models/experiments/'+model_name
        exp_path = exp_dir+'/model_epo'+epoch_count
        self.samples_n = 1
        self.history_len = 50 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, self.history_len, self.state_dim])
        dataset_name = 'sim_data_'+data_id
        data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'

        with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
            self.env_scaler = pickle.load(handle)

        with open(data_files_dir+'m_scaler.pickle', 'rb') as handle:
            self.m_scaler = pickle.load(handle)

        with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        self.indxs = {}
        feature_names = ['e_veh_speed', 'f_veh_speed','m_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                        'em_delta_y', 'delta_x_to_merge', 'm_veh_exists']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1
        # self.model.forward_sim.attention_temp = 20
        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)
        self.load_model(config, exp_path)
        print(json.dumps(config, ensure_ascii=False, indent=4))

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        if self.time_lapse_since_last_param_update == 0:
            prior = tfd.Normal(loc=tf.zeros([1, self.model.belief_net.latent_dim]), scale=1)
            sampled_z = prior.sample()
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
