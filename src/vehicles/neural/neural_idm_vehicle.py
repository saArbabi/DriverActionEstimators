from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
import numpy as np
import pickle
from importlib import reload
import tensorflow as tf
import json

class NeuralIDMVehicle(IDMMOBILVehicleMerge):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse_since_last_param_update = 0
        self.samples_n = 1
        self.history_len = 30 # steps
        self.state_dim = 10
        self.obs_history = np.zeros([self.samples_n, self.history_len, self.state_dim])

    def load_model(self, config, exp_path):
        from models.core import neural_idm
        reload(neural_idm)
        from models.core.neural_idm import  NeurIDMModel
        self.model = NeurIDMModel(config)
        self.model.load_weights(exp_path).expect_partial()

    def initialize_agent(self, model_name, epoch_count, data_id):
        exp_dir = './src/models/experiments/'+model_name
        exp_path = exp_dir+'/model_epo'+epoch_count
        dataset_name = 'sim_data_'+data_id
        data_files_dir = './src/models/experiments/data_files/'+dataset_name+'/'

        with open(data_files_dir+'env_scaler.pickle', 'rb') as handle:
            self.env_scaler = pickle.load(handle)

        with open(data_files_dir+'m_scaler.pickle', 'rb') as handle:
            self.m_scaler = pickle.load(handle)

        with open(data_files_dir+'dummy_value_set.pickle', 'rb') as handle:
            self.dummy_value_set = pickle.load(handle)

        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)
        self.load_model(config, exp_path)
        print(json.dumps(config, ensure_ascii=False, indent=4))
        self.create_state_indxs()

    def names_to_index(self, col_names):
        if type(col_names) == list:
            return [self.indxs[item] for item in col_names]
        else:
            return self.indxs[col_names]

    def create_state_indxs(self):
        self.indxs = {}
        feature_names = ['e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x',
                        'em_delta_v', 'em_delta_x',
                        'm_veh_speed','em_delta_y',
                        'delta_x_to_merge','m_veh_exists']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1
        col_names = ['e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']
        self.env_s_indxs = self.names_to_index(col_names)

        col_names = ['m_veh_speed', 'em_delta_y', 'delta_x_to_merge']
        self.merger_indxs = self.names_to_index(col_names)

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t[0, 0, :]

    def neur_observe(self):
        m_veh = self.neighbours['m']
        f_veh = self.neighbours['f']
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
            em_delta_x = m_veh.glob_x-self.glob_x
            em_delta_y = abs(m_veh.glob_y-self.glob_y)
            em_delta_v = self.speed-m_veh_speed
            delta_x_to_merge = m_veh.ramp_exit_start-m_veh.glob_x

        if not f_veh:
            f_veh_exists = 0
            f_veh_speed = self.dummy_value_set['f_veh_speed']
            el_delta_x = self.dummy_value_set['el_delta_x']
            el_delta_v = self.dummy_value_set['el_delta_v']
        else:
            f_veh_exists = 1
            f_veh_speed = f_veh.speed
            el_delta_x = f_veh.glob_x-self.glob_x
            el_delta_v = self.speed-f_veh_speed

        obs_t0 = [self.speed, f_veh_speed]

        obs_t0.extend([el_delta_v,
                             el_delta_x])

        obs_t0.extend([em_delta_v,
                             em_delta_x,
                             m_veh_speed,
                             em_delta_y,
                             delta_x_to_merge])

        obs_t0.append(m_veh_exists)
        self.m_veh_exists = m_veh_exists

        return [np.array([[obs_t0]]), [[[float(m_veh_exists)]]]]

    def driver_params_update(self, idm_params):
        idm_params = idm_params.numpy()[0]
        self.driver_params['desired_v'] = idm_params[0]
        self.driver_params['desired_tgap'] = idm_params[1]
        self.driver_params['min_jamx'] = idm_params[2]
        self.driver_params['max_act'] = idm_params[3]
        self.driver_params['min_act'] = idm_params[4]

    def belief_update(self, proj_belief):
        self.proj_latent = tf.reshape(proj_belief, [self.samples_n, 1, 64])
        # if self.time_lapse_since_last_param_update == 0:
            # self.state_h = self.state_c = tf.zeros([self.samples_n, 128])
        self.state_h = self.state_c = tf.zeros([self.samples_n, 128])

    def scale_state(self, state, state_type):
        if state_type == 'full':
            state[:, :, self.env_s_indxs] = \
                (state[:, :, self.env_s_indxs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5

            # merger context
            state[:, :, self.merger_indxs] = \
                (state[:, :, self.merger_indxs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5
            state[:,:,]
        elif state_type == 'env_state':
            state = \
                (state[:, :, self.env_s_indxs]-self.env_scaler.mean_)/self.env_scaler.var_**0.5

        elif state_type == 'merger_c':
            state = \
                (state[:, :, self.merger_indxs]-self.m_scaler.mean_)/self.m_scaler.var_**0.5

        return np.float32(state)

    def get_neur_att(self, att_context):
        lstm_output, self.state_h, self.state_c = self.model.forward_sim.lstm_layer(\
                                    att_context, initial_state=[self.state_h, self.state_c])
        attention_temp = 5
        att_score = 1/(1+tf.exp(-attention_temp*self.model.forward_sim.att_neu(lstm_output))).numpy()
        return att_score

    def action_clip(self, act_long):
        return max([-5.5, act_long])

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        # if self.time_lapse_since_last_param_update % 30 == 0:
        if self.time_lapse_since_last_param_update == 0:
            obs_history = self.scale_state(self.obs_history.copy(), 'full')
            enc_h = self.model.h_seq_encoder(obs_history)
            # self.enc_h = tf.reshape(enc_h, [self.samples_n, 1, 128])
            latent_dis_param = self.model.belief_net(enc_h, dis_type='prior')
            z_idm, z_att = self.model.belief_net.sample_z(latent_dis_param)

            proj_att = self.model.belief_net.z_proj_att(z_att)
            proj_idm = self.model.belief_net.z_proj_idm(z_idm)
            self.belief_update(proj_att)
            idm_params = self.model.idm_layer(proj_idm)
            self.driver_params_update(idm_params)
            # if self.id == 'neur_4':
            #     print('self.obs_history ', self.obs_history)
        self.time_lapse_since_last_param_update += 1

        ef_act = self.action_clip(self.idm_action(self, self.neighbours['f']))

        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        att_context = tf.concat([self.proj_latent, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)
        att_score = self.get_neur_att(att_context)
        att_score = att_score[0][0][0]
        # if att_score > 0.2 and self.id == 'neur_4':
        #     att_score = 1



        if self.neighbours['m'] and self.neighbours['m'].glob_x > self.glob_x:
            em_act = self.action_clip(self.idm_action(self, self.neighbours['m']))
        else:
            em_act = 0
            att_score = 0

        self.att = att_score
        act_long = (1-att_score)*ef_act + att_score*em_act

        # if self.id == 'neur_4':
        #     print('m_veh_exists', m_veh_exists)
        #     print('ef_act', ef_act)
        #     print('em_act', em_act)
            # print('obs_history ', obs_history)
            # print('nei ', self.neighbours)
            # print('att_context ', att_context)

        return act_long
