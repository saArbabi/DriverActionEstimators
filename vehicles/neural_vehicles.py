from vehicles.idmmobil_vehicle import IDMMOBILVehicle
import numpy as np
import pickle
from importlib import reload
import tensorflow as tf

class NeuralIDMVehicle(IDMMOBILVehicle):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse_since_last_param_update = 0
        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        exp_dir = './models/experiments/'+'h_z_f_idm_act088_epo_25'+'/model'

        with open('./models/experiments/env_scaler_024.pickle', 'rb') as handle:
            self.env_scaler = pickle.load(handle)

        with open('./models/experiments/m_scaler_024.pickle', 'rb') as handle:
            self.m_scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set_024.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModel
        self.model = NeurIDMModel()
        self.model.load_weights(exp_dir).expect_partial()
        # self.model.forward_sim.attention_temp = 20

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def neur_observe(self, e_veh, f_veh, m_veh):
        if self.collision_detected:
            return
        if not m_veh:
            m_veh_exists = 0
            m_veh_speed = self.dummy_value_set['m_veh_speed']
            m_veh_action = self.dummy_value_set['m_veh_action']
            em_delta_x = self.dummy_value_set['em_delta_x']
            em_delta_v = self.dummy_value_set['em_delta_v']
            em_delta_y = self.dummy_value_set['em_delta_y']
            delta_x_to_merge = self.dummy_value_set['delta_x_to_merge']

        else:
            m_veh_exists = 1
            m_veh_speed = m_veh.speed
            m_veh_action = m_veh.act_long
            em_delta_x = m_veh.glob_x-e_veh.glob_x
            em_delta_y = abs(m_veh.glob_y-e_veh.glob_y)
            em_delta_v = e_veh.speed-m_veh_speed
            delta_x_to_merge = 200-m_veh.glob_x

        if not f_veh:
            f_veh_exists = 0
            f_veh_speed = self.dummy_value_set['f_veh_speed']
            el_delta_x = self.dummy_value_set['el_delta_x']
            el_delta_v = self.dummy_value_set['el_delta_v']
        else:
            f_veh_exists = 1
            f_veh_speed = f_veh.speed
            el_delta_x = f_veh.glob_x-e_veh.glob_x
            el_delta_v = e_veh.speed-f_veh_speed

        obs_t0 = [e_veh.speed, f_veh_speed, m_veh_speed]

        obs_t0.extend([el_delta_v,
                             el_delta_x])
        obs_t0.extend([em_delta_v,
                             em_delta_x,
                             em_delta_y,
                             delta_x_to_merge])

        obs_t0.append(m_veh_exists)
        self.m_veh_exists = m_veh_exists
        merger_c = [m_veh_speed, em_delta_y, delta_x_to_merge, m_veh_exists]
        neighbours = [f_veh, m_veh]
        if min([el_delta_x, em_delta_x]) <= 0:
            self.collision_detected = True
            return

        return [obs_t0, merger_c, neighbours]

    def driver_params_update(self, inputs):
        idm_params = self.model.idm_layer(inputs).numpy()[0]
        self.driver_params['desired_v'] = idm_params[0]
        self.driver_params['desired_tgap'] = idm_params[1]
        self.driver_params['min_jamx'] = idm_params[2]
        self.driver_params['max_act'] = idm_params[3]
        self.driver_params['min_act'] = idm_params[4]

    def latent_projection_update(self, sampled_z):
        latent_projection = self.model.forward_sim.projection(sampled_z)
        self.latent_projection = tf.reshape(latent_projection, [self.samples_n, 1, 50])
        # latent_projection = self.latent_projection([latent_projection, enc_h], batch_size)
        # self.state_h, self.state_c = latent_projection, latent_projection
        self.state_h = self.state_c = tf.zeros([self.samples_n, 100])

    def prep_obs_seq(self, obs_history):
        obs_history = np.float32(obs_history)
        obs_history.shape = (self.samples_n*20, self.state_dim)
        obs_history[:, :-3] = self.scaler.transform(obs_history[:, :-3])
        obs_history.shape = (self.samples_n, 20, self.state_dim)
        return obs_history

    def get_neur_att(self, sdv_act):
        att_inputs = tf.concat([self.latent_projection, sdv_act], axis=-1)
        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                                    att_inputs, initial_state=[self.state_h, self.state_c])
        attention_temp = 5
        att_score = 1/(1+tf.exp(-attention_temp*self.model.forward_sim.attention_neu(lstm_output))).numpy()
        return att_score

    def act(self, obs):
        obs_t0, merger_c, neighbours = obs
        if self.time_lapse_since_last_param_update % 20 == 0:
            obs_history = self.prep_obs_seq(self.obs_history.copy())
            enc_h = self.model.h_seq_encoder(obs_history)
            prior_param = self.model.belief_net(enc_h, dis_type='prior')
            sampled_z = self.model.belief_net.sample_z(prior_param)

            self.driver_params_update([sampled_z, obs_history[:, -1, :]])
            self.latent_projection_update(sampled_z)
            self.time_lapse_since_last_param_update = 0

        f_veh_exists = merger_c[-2]
        m_veh_exists = merger_c[-1]
        sdv_act = np.array([[merger_c]])

        att_score = self.get_neur_att(sdv_act)[0][0][0]
        att_score = (f_veh_exists*att_score + 1*(1-f_veh_exists))*m_veh_exists
        self.att = att_score

        f_veh, m_veh = neighbours
        ef_act = self.idm_action(self, f_veh)
        em_act = self.idm_action(self, m_veh)

        act_long = (1-att_score)*ef_act + att_score*em_act
        self.time_lapse_since_last_param_update += 1
        return act_long

class LSTMVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 30 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        model_name = 'lstm_model'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core.lstm import Encoder
        self.model = Encoder()
        self.model.load_weights(exp_dir).expect_partial()

    def act(self, obs):
        obs_history = self.prep_obs_seq(self.obs_history.copy())
        pred_dis = self.model(obs_history)
        act_long = pred_dis.sample().numpy()[0][0]
        self.att = -1
        return act_long

class NeurLatentVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        with open('./models/experiments/scaler_009.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set_009.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core.h_z_f_act import NeurLatentModel
        self.model = NeurLatentModel()
        exp_dir = './models/experiments/'+'h_z_f_act009_epo_15'+'/model'
        self.model.load_weights(exp_dir).expect_partial()

    def latent_projection_update(self, sampled_z):
        latent_projection = self.model.forward_sim.projection(sampled_z)
        self.latent_projection = tf.reshape(latent_projection, [self.samples_n, 1, 50])
        self.state_h = self.state_c = tf.zeros([self.samples_n, 100])

    def neur_observe(self, e_veh, f_veh, m_veh):
        if self.collision_detected:
            return
        if not m_veh:
            m_veh_exists = 0
            m_veh_speed = self.dummy_value_set['m_veh_speed']
            m_veh_action = self.dummy_value_set['m_veh_action']
            em_delta_x = self.dummy_value_set['em_delta_x']
            em_delta_v = self.dummy_value_set['em_delta_v']
            em_delta_y = self.dummy_value_set['em_delta_y']
        else:
            m_veh_exists = 1
            m_veh_speed = m_veh.speed
            m_veh_action = m_veh.act_long
            em_delta_x = m_veh.glob_x-e_veh.glob_x
            em_delta_y = abs(m_veh.glob_y-e_veh.glob_y)
            em_delta_v = e_veh.speed-m_veh_speed

        if not f_veh:
            f_veh_exists = 0
            f_veh_speed = self.dummy_value_set['f_veh_speed']
            el_delta_x = self.dummy_value_set['el_delta_x']
            el_delta_v = self.dummy_value_set['el_delta_v']
        else:
            f_veh_exists = 1
            f_veh_speed = f_veh.speed
            el_delta_x = f_veh.glob_x-e_veh.glob_x
            el_delta_v = e_veh.speed-f_veh_speed

        obs_t0 = [e_veh.speed, f_veh_speed, m_veh_speed]

        obs_t0.extend([el_delta_v,
                             el_delta_x])
        obs_t0.extend([em_delta_v,
                             em_delta_x,
                             em_delta_y])

        obs_t0.extend([f_veh_exists, m_veh_exists])
        self.m_veh_exists = m_veh_exists
        merger_c = [em_delta_y, m_veh_action, f_veh_exists, m_veh_exists]

        env_state = [e_veh.speed, f_veh_speed, m_veh_speed,
                     el_delta_v, el_delta_x, em_delta_v, em_delta_x]

        neighbours = [f_veh, m_veh]
        if min([el_delta_x, em_delta_x]) <= 0:
            self.collision_detected = True
            return

        return [obs_t0, merger_c, env_state]

    def scale_features(self, env_state):
        env_state = np.float32(env_state)
        env_state.shape = (self.samples_n*1, self.state_dim-3)
        env_state[:, :] = self.scaler.transform(env_state[:, :])
        env_state.shape = (self.samples_n, 1, self.state_dim-3)
        return env_state

    def act(self, obs):
        obs_t0, merger_c, env_state = obs
        if self.time_lapse_since_last_param_update % 20 == 0:
            obs_history = self.prep_obs_seq(self.obs_history.copy())
            enc_h = self.model.h_seq_encoder(obs_history)
            prior_param = self.model.belief_net(enc_h, dis_type='prior')
            sampled_z = self.model.belief_net.sample_z(prior_param)
            self.latent_projection_update(sampled_z)
            self.time_lapse_since_last_param_update = 0

        sdv_act = np.array([[merger_c]])
        env_state = self.scale_features(env_state)

        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                             tf.concat([self.latent_projection, sdv_act, env_state], axis=-1)\
                             , initial_state=[self.state_h, self.state_c])
        act_long = self.model.forward_sim.action_neu(lstm_output).numpy()

        self.time_lapse_since_last_param_update += 1
        return act_long[0][0][0]

class NeurLatentOneStepVehicle(NeurLatentVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 30 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core.h_z_f_act import NeurLatentModel
        self.model = NeurLatentModel()
        exp_dir = './models/experiments/'+'h_lat_act_epo_10'+'/model'
        self.model.load_weights(exp_dir).expect_partial()

class MLPVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        instance_len = 20 # steps
        self.state_dim = 10
        self.obs_instance = np.zeros([self.samples_n, self.state_dim])
        # self.action_instance = [[0., 0.]]*20

        model_name = 'mlp_model'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core.mlp import MLP
        self.model = MLP()
        self.model.load_weights(exp_dir).expect_partial()

    def prep_obs_seq(self, obs_instance):
        obs_instance = np.float32(obs_instance)
        obs_instance[:, :-3] = self.scaler.transform(obs_instance[:, :-3])
        return obs_instance

    def update_obs_history(self, o_t):
        self.obs_instance[:, :] = o_t

    def act(self, obs):
        obs_instance = self.prep_obs_seq(self.obs_instance.copy())
        pred_dis = self.model(obs_instance)
        act_long = pred_dis.sample().numpy()[0][0]
        self.att = -1
        return act_long
