from vehicles.idmmobil_vehicle import IDMMOBILVehicle
import numpy as np
import pickle
from importlib import reload
import tensorflow as tf

class NeuralIDMVehicle(IDMMOBILVehicle):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse = 0
        self.control_type = 'idmmobil'
        self.time_lapse_since_last_param_update = 0
        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])

        model_name = 'driver_model'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('./models/experiments/dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModel
        self.model = NeurIDMModel()
        self.model.load_weights(exp_dir).expect_partial()
        # self.model.idm_sim.attention_temp = 20

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def neur_observe(self, e_veh, f_veh, m_veh):
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
        m_veh_action_feature = [em_delta_y, m_veh_action, f_veh_exists, m_veh_exists]

        idm_s = [el_delta_v, el_delta_x, em_delta_v, em_delta_x, f_veh_exists, m_veh_exists]
        return [obs_t0, m_veh_action_feature, idm_s]

    def driver_params_update(self, sampled_idm_z):
        idm_params = self.model.idm_layer(sampled_idm_z).numpy()[0]
        self.driver_params['desired_v'] = idm_params[0]
        self.driver_params['desired_tgap'] = idm_params[1]
        self.driver_params['min_jamx'] = idm_params[2]
        self.driver_params['max_act'] = idm_params[3]
        self.driver_params['min_act'] = idm_params[4]

    def att_context_update(self, sampled_att_z):
        att_projection = self.model.idm_sim.linear_layer(sampled_att_z)
        self.att_context = tf.reshape(att_projection, [self.samples_n, 1, 100])
        # att_context = self.att_context([att_projection, enc_h], batch_size)
        self.state_h, self.state_c = att_projection, att_projection

    def prep_obs_seq(self, obs_history):
        obs_history = np.float32(obs_history)
        # print('obs_history', obs_history[:, -1, :])
        obs_history.shape = (self.samples_n*20, self.state_dim)
        obs_history[:, :-3] = self.scaler.transform(obs_history[:, :-3])
        obs_history.shape = (self.samples_n, 20, self.state_dim)
        return obs_history

    def get_neur_att(self, sdv_act):
        att_inputs = tf.concat([self.att_context, sdv_act], axis=-1)
        lstm_output, self.state_h, self.state_c = self.model.idm_sim.lstm_layer(\
                                    att_inputs, initial_state=[self.state_h, self.state_c])
        attention_temp = 20
        att_score = 1/(1+tf.exp(-attention_temp*self.model.idm_sim.attention_neu(lstm_output))).numpy()
        return att_score

    def act(self, obs):
        obs_t0, m_veh_action_feature, idm_s = obs
        # if self.time_lapse > 5:
        # print(self.obs_history)
        if self.time_lapse_since_last_param_update % 30 == 0:
        # if self.time_lapse_since_last_param_update == 0:
            obs_history = self.prep_obs_seq(self.obs_history.copy())
            # print(self.obs_history[:,-1,:])
            enc_h = self.model.h_seq_encoder(obs_history)
            prior_param = self.model.belief_net(enc_h, dis_type='prior')
            sampled_att_z, sampled_idm_z = self.model.belief_net.sample_z(prior_param)

            self.driver_params_update(sampled_idm_z)
            self.att_context_update(sampled_att_z)
            self.time_lapse_since_last_param_update = 0

        # actions = np.float32(np.array(actions))
        # sdv_act = np.repeat(np.array([[m_veh_action_feature]]), self.samples_n, axis=0)
        sdv_act = np.array([[m_veh_action_feature]])

        att_score = self.get_neur_att(sdv_act).tolist()[0][0][0]
        self.att = att_score

        el_delta_v, el_delta_x, em_delta_v, \
                            em_delta_x, f_veh_exists, m_veh_exists = idm_s

        print(sdv_act)
        ef_act = self.idm_action([el_delta_v, el_delta_x])
        em_act = self.idm_action([em_delta_v, em_delta_x])
        act_long = (1-att_score)*ef_act + att_score*em_act
        # print('att_score', att_score)
        # print('act_long', act_long)
        # print('sdv_act', sdv_act)
        self.time_lapse_since_last_param_update += 1
        # print(sdv_act)
        # print(self.att)
        return act_long

    def get_obs(self, ):
        """
        Returns an observation sequence conditioned on:
        - neighbouring cars
        - merger
        Note:
        """

class LSTMVehicle(NeuralIDMVehicle):
    def __init__(self):
        super().__init__()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        model_name = 'lstm_model'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        from models.core.lstm import Encoder
        self.model = Encoder()
        self.model.load_weights(exp_dir).expect_partial()

    def act(self, obs):
        obs_history = self.prep_obs_seq(self.obs_history.copy())
        pred_dis = self.model(obs_history)
        act_long = pred_dis.sample().numpy()[0][0]
        self.att = -1
        return act_long
#
# class LSTMVehicle(IDMVehicle):
#     def __init__(self):
#         super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
#         self.time_lapse = 0
#         self.control_type = 'idm'
#         self.initialize_agent()
#
#     def initialize_agent(self, config=None):
#         self.samples_n = 1
#         history_len = 20 # steps
#         self.state_dim = 4
#         self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
#         # self.action_history = [[0., 0.]]*20
#
#         model_name = 'lstm_keep_lane'
#         exp_dir = './models/experiments/'+model_name+'/model'
#         with open('./models/experiments/scaler.pickle', 'rb') as handle:
#             self.scaler = pickle.load(handle)
#
#         from models.core.lstm import Encoder
#         self.model = Encoder()
#         self.model.load_weights(exp_dir).expect_partial()
#         # self.model.idm_sim.attention_temp = 20
#
#     def update_obs_history(self, o_t):
#         # print(self.obs_history)
#         self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
#         self.obs_history[:, -1, :] = o_t
#
#     def observe(self, e_veh, f_veh):
#         obs_t0 = [e_veh.speed, f_veh.speed]
#
#         obs_t0.extend([
#                              e_veh.speed-f_veh.speed,
#                              f_veh.glob_x-e_veh.glob_x])
#         return obs_t0
#
#     def act(self, obs):
#         self.time_lapse += 1
#
#         # if self.time_lapse > 5:
#         self.update_obs_history(obs)
#         # print(self.obs_history)
#         if self.control_type == 'neural':
#             # x = self.scaler.transform(x)
#             obs_history = np.float32(self.obs_history.copy())
#             obs_history.shape = (self.samples_n*20, self.state_dim)
#             obs_history = self.scaler.transform(obs_history)
#             obs_history.shape = (self.samples_n, 20, self.state_dim)
#
#             pred_dis = self.model(obs_history)
#             act_long = pred_dis.sample().numpy()[0][0]
#             print(act_long)
#             return act_long
#         elif self.time_lapse > 25:
#             self.control_type = 'neural'
#
#         act_long = self.idm_action(obs[-2:])
#         return act_long
