from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
import numpy as np
import pickle
from importlib import reload
import tensorflow as tf

class NeuralIDMVehicle(IDMMOBILVehicleMerge):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse_since_last_param_update = 0
        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        exp_dir = './models/experiments/'+'h_z_f_idm_act_test_epo_10'+'/model'

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

        self.indxs = {}
        feature_names = ['e_veh_speed', 'f_veh_speed','m_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                        'em_delta_y', 'delta_x_to_merge', 'm_veh_exists']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1
        # self.model.forward_sim.attention_temp = 20

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t[0, 0, :]

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
        neighbours = [f_veh, m_veh]
        if min([el_delta_x, em_delta_x]) <= 0:
            self.collision_detected = True
            return

        return [np.array([[obs_t0]]), [[[float(m_veh_exists)]]], neighbours]

    def driver_params_update(self, idm_params):
        idm_params = idm_params.numpy()[0]
        self.driver_params['desired_v'] = idm_params[0]
        self.driver_params['desired_tgap'] = idm_params[1]
        self.driver_params['min_jamx'] = idm_params[2]
        self.driver_params['max_act'] = idm_params[3]
        self.driver_params['min_act'] = idm_params[4]

    def belief_update(self, proj_belief):
        self.proj_latent = tf.reshape(proj_belief, [self.samples_n, 1, 50])
        self.state_h = self.state_c = tf.zeros([self.samples_n, 100])

    # def prep_obs_seq(self, obs_history):
    #     obs_history = np.float32(obs_history)
    #     obs_history.shape = (self.samples_n*20, self.state_dim)
    #     obs_history[:, :-3] = self.scaler.transform(obs_history[:, :-3])
    #     obs_history.shape = (self.samples_n, 20, self.state_dim)
    #     return obs_history
    def names_to_index(self, col_names):
        if type(col_names) == list:
            return [self.indxs[item] for item in col_names]
        else:
            return self.indxs[col_names]

    def scale_state(self, state, state_type):
        if state_type == 'full':
            col_names = ['e_veh_speed', 'f_veh_speed',
                            'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']

            scalar_indexs = self.names_to_index(col_names)
            state[:, :, scalar_indexs] = \
                (state[:, :, scalar_indexs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5

            # merger context
            col_names = ['m_veh_speed','em_delta_y', 'delta_x_to_merge']
            scalar_indexs = self.names_to_index(col_names)
            state[:, :, scalar_indexs] = \
                (state[:, :, scalar_indexs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5

        elif state_type == 'env_state':
            col_names = ['e_veh_speed', 'f_veh_speed',
                            'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']

            scalar_indexs = self.names_to_index(col_names)
            state = \
                (state[:, :, scalar_indexs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5

        elif state_type == 'merger_c':
            col_names = ['m_veh_speed','em_delta_y', 'delta_x_to_merge']
            scalar_indexs = self.names_to_index(col_names)
            state = \
                (state[:, :, scalar_indexs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5

        return state

    def get_neur_att(self, att_context):
        # att_inputs = tf.concat([self.proj_latent, sdv_act], axis=-1)
        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                                    att_context, initial_state=[self.state_h, self.state_c])
        attention_temp = 5
        att_score = 1/(1+tf.exp(-attention_temp*self.model.forward_sim.attention_neu(lstm_output))).numpy()
        return att_score

    def act(self, obs):
        obs_t0, m_veh_exists, neighbours = obs
        if self.time_lapse_since_last_param_update % 20 == 0:
            obs_history = self.scale_state(self.obs_history.copy(), 'full')
            enc_h = self.model.h_seq_encoder(obs_history)
            latent_dis_param = self.model.belief_net(enc_h, dis_type='prior')
            sampled_z = self.model.belief_net.sample_z(latent_dis_param)
            proj_belief = self.model.belief_net.belief_proj(sampled_z)
            idm_params = self.model.idm_layer(proj_belief)
            self.driver_params_update(idm_params)
            self.belief_update(proj_belief)
            self.time_lapse_since_last_param_update = 0

        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        att_context = tf.concat([self.proj_latent, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)
        # att_score = self.get_neur_att(att_context)[0][0][0]
        att_score = 0
        print(att_score)
        # att_score = att_score*m_veh_exists
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

    def belief_update(self, sampled_z):
        latent_projection = self.model.forward_sim.projection(sampled_z)
        self.proj_latent = tf.reshape(latent_projection, [self.samples_n, 1, 50])
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
            self.proj_latent_update(sampled_z)
            self.time_lapse_since_last_param_update = 0

        sdv_act = np.array([[merger_c]])
        env_state = self.scale_features(env_state)

        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                             tf.concat([self.proj_latent, sdv_act, env_state], axis=-1)\
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
