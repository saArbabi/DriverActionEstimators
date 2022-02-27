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
        self.state_dim = 13
        self.obs_history = np.zeros([self.samples_n, self.history_len, self.state_dim])

    def load_model(self, config, exp_path):
        from models.core import neural_idm
        reload(neural_idm)
        from models.core.neural_idm import  NeurIDMModel
        self.model = NeurIDMModel(config)
        # self.model.forward_sim.attention_temp = 3

        self.model.load_weights(exp_path).expect_partial()

    def initialize_agent(self, model_name, epoch_count, data_id):
        exp_dir = './src/models/experiments/'+model_name
        exp_path = exp_dir+'/model_epo'+epoch_count
        dataset_name = 'sim_data_'+data_id
        data_files_dir = './src/datasets/'+dataset_name+'/'

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
        feature_names = [
                        'e_veh_action_p', 'f_veh_action_p',
                        'e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x',
                        'em_delta_v', 'em_delta_x',
                        'm_veh_action_p', 'm_veh_speed','em_delta_y',
                        'delta_x_to_merge','m_veh_exists']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1
        col_names = ['e_veh_action_p', 'f_veh_action_p', 'e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']
        self.env_s_indxs = self.names_to_index(col_names)
        col_names = ['m_veh_action_p', 'm_veh_speed', 'em_delta_y', 'delta_x_to_merge']
        self.merger_indxs = self.names_to_index(col_names)

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t[0, 0, :]

    def neur_observe(self):
        m_veh = self.neighbours['m']
        f_veh = self.neighbours['f']
        if not m_veh:
            m_veh_exists = 0
            m_veh_action = self.dummy_value_set['m_veh_action_p']
            m_veh_speed = self.dummy_value_set['m_veh_speed']
            em_delta_x = self.dummy_value_set['em_delta_x']
            em_delta_v = self.dummy_value_set['em_delta_v']
            em_delta_y = self.dummy_value_set['em_delta_y']
            delta_x_to_merge = self.dummy_value_set['delta_x_to_merge']

        else:
            m_veh_exists = 1
            m_veh_action = m_veh.act_long_p
            m_veh_speed = m_veh.speed
            em_delta_x = m_veh.glob_x-self.glob_x
            em_delta_y = abs(m_veh.glob_y-self.glob_y)
            em_delta_v = self.speed-m_veh_speed
            delta_x_to_merge = m_veh.ramp_exit_start-m_veh.glob_x

        if not f_veh:
            f_veh_exists = 0
            f_veh_action = self.dummy_value_set['f_veh_action_p']
            f_veh_speed = self.dummy_value_set['f_veh_speed']
            el_delta_x = self.dummy_value_set['el_delta_x']
            el_delta_v = self.dummy_value_set['el_delta_v']
        else:
            f_veh_exists = 1
            f_veh_action = f_veh.act_long_p
            f_veh_speed = f_veh.speed
            el_delta_x = f_veh.glob_x-self.glob_x
            el_delta_v = self.speed-f_veh_speed

        obs_t0 = [self.act_long_p, f_veh_action, self.speed, f_veh_speed]

        obs_t0.extend([el_delta_v,
                             el_delta_x])

        obs_t0.extend([em_delta_v,
                             em_delta_x,
                             m_veh_action,
                             m_veh_speed,
                             em_delta_y,
                             delta_x_to_merge])
        obs_t0.append(m_veh_exists)
        self.m_veh_exists = m_veh_exists
        return [np.array([[obs_t0]]), [[[float(m_veh_exists)]]]]

    def driver_params_update(self, idm_params):
        idm_params = idm_params.numpy()[0, :]
        self.driver_params['desired_v'] = idm_params[0]
        self.driver_params['desired_tgap'] = idm_params[1]
        self.driver_params['min_jamx'] = idm_params[2]
        self.driver_params['max_act'] = idm_params[3]
        self.driver_params['min_act'] = idm_params[4]

    def belief_update(self, proj_latent):
        self.proj_latent = tf.reshape(proj_latent, [self.samples_n, 1, 128])

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
        f_att_score, m_att_score = self.model.forward_sim.get_att(att_context)
        f_att_score, m_att_score = f_att_score.numpy()[0][0][0], m_att_score.numpy()[0][0][0]
        f_att_score = (1 - self.m_veh_exists) + f_att_score*self.m_veh_exists
        m_att_score = m_att_score*self.m_veh_exists
        return f_att_score, m_att_score

    def action_clip(self, act_long):
        return min(max([-6, act_long]), 6)

    def act(self, obs):
        obs_t0, m_veh_exists = obs
        # if self.time_lapse_since_last_param_update % 20 == 0:
        if self.time_lapse_since_last_param_update == 0:
            obs_history = self.scale_state(self.obs_history.copy(), 'full')
            enc_h = self.model.h_seq_encoder(obs_history)
            latent_dis_param = self.model.belief_net(enc_h , dis_type='prior')
            z_idm, z_att = self.model.belief_net.sample_z(latent_dis_param)
            proj_idm = self.model.belief_net.z_proj_idm(z_idm)
            proj_att = self.model.belief_net.z_proj_att(z_att)
            self.belief_update(proj_att)
            self.enc_h = tf.reshape(enc_h, [self.samples_n, 1, 128])
            idm_params = self.model.idm_layer(proj_idm)
            self.driver_params_update(idm_params)
            # if self.id == 'neur_4':
            #     print('self.obs_history ', self.obs_history)


        self.time_lapse_since_last_param_update += 1


        env_state = self.scale_state(obs_t0, 'env_state')
        merger_c = self.scale_state(obs_t0, 'merger_c')
        att_context = tf.concat([self.proj_latent , self.enc_h, env_state, merger_c, \
                                                        m_veh_exists], axis=-1)
        f_att_score, m_att_score = self.get_neur_att(att_context)
        ef_act = self.action_clip(self.idm_action(self, self.neighbours['f']))

        if self.neighbours['m'] and self.neighbours['m'].glob_x > self.glob_x:
            em_act = self.idm_action(self, self.neighbours['m'])
            # if self.id == 'neur_2':
            #     print('em_act ', em_act)

            if em_act < -20:
                # not a feasible action
                em_act = 0
                m_att_score = 0
            else:
                em_act = self.action_clip(em_act)
        else:
            # no merger to attend to
            em_act = 0
            m_att_score = 0


        self.att = m_att_score
        act_long = f_att_score*ef_act + m_att_score*em_act



        return act_long
