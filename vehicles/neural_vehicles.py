from vehicles.idm_vehicle import IDMVehicle
import numpy as np
import pickle
from importlib import reload

class NeuralIDMVehicle(IDMVehicle):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse = 0
        self.control_type = 'idm'
        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 4
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        model_name = 'driver_model_kl'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        from models.core.driver_model import  NeurIDMModelLaneKeep
        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModelLaneKeep
        self.model = NeurIDMModelLaneKeep()
        self.model.load_weights(exp_dir).expect_partial()
        # self.model.idm_sim.attention_temp = 20

    def update_obs_history(self, o_t):
        # print(self.obs_history)
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def observe(self, e_veh, f_veh):
        step_feature = [e_veh.speed, f_veh.speed]

        step_feature.extend([
                             e_veh.speed-f_veh.speed,
                             f_veh.glob_x-e_veh.glob_x])
        return step_feature

    def predict_idm_params(self, obs_history):
        if self.time_lapse % 10 == 0:
            obs_history = np.float32(np.array(obs_history))
            obs_history.shape = (self.samples_n*20, self.state_dim)
            obs_history = self.scaler.transform(obs_history)
            obs_history.shape = (self.samples_n, 20, self.state_dim)

            # actions = np.float32(np.array(actions))
            # actions.shape = (1, 40, 2)
            # actions = np.repeat(actions, self.samples_n, axis=0)
            enc_h = self.model.h_seq_encoder(obs_history)
            enc_acts = self.model.act_encoder(obs_history[:,:, -1:])
            prior_param = self.model.belief_net([enc_h, enc_acts], dis_type='prior')
            sampled_att_z, sampled_idm_z = self.model.belief_net.sample_z(prior_param)
            # att_scores =  self.model.arbiter([sampled_att_z, enc_h, enc_acts])
            idm_params = self.model.idm_layer([sampled_idm_z, enc_h]).numpy()[0]
                    # print('self.elapsed_time: ', round(self.elapsed_time, 1))
            self.driver_params['desired_v'] = idm_params[0]
            self.driver_params['desired_tgap'] = idm_params[1]
            self.driver_params['min_jamx'] = idm_params[2]
            self.driver_params['max_act'] = idm_params[3]
            self.driver_params['min_act'] = idm_params[4]

        # return idm_params

    def act(self, obs):
        self.time_lapse += 1

        # if self.time_lapse > 5:
        self.update_obs_history(obs)
        # print(self.obs_history)
        if self.control_type == 'neural':
            self.predict_idm_params(self.obs_history.copy())

        elif self.time_lapse > 25:
            self.control_type = 'neural'

        act_long = self.idm_action(obs[-2:])
        return act_long

    def get_obs(self, ):
        """
        Returns an observation sequence conditioned on:
        - neighbouring cars
        - merger
        Note:
        """

class LSTMVehicle(IDMVehicle):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)
        self.time_lapse = 0
        self.control_type = 'idm'
        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.samples_n = 1
        history_len = 20 # steps
        self.state_dim = 4
        self.obs_history = np.zeros([self.samples_n, history_len, self.state_dim])
        # self.action_history = [[0., 0.]]*20

        model_name = 'lstm_kl'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        from models.core.lstm import Encoder
        self.model = Encoder()
        self.model.load_weights(exp_dir).expect_partial()
        # self.model.idm_sim.attention_temp = 20

    def update_obs_history(self, o_t):
        # print(self.obs_history)
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def observe(self, e_veh, f_veh):
        step_feature = [e_veh.speed, f_veh.speed]

        step_feature.extend([
                             e_veh.speed-f_veh.speed,
                             f_veh.glob_x-e_veh.glob_x])
        return step_feature

    def act(self, obs):
        self.time_lapse += 1

        # if self.time_lapse > 5:
        self.update_obs_history(obs)
        # print(self.obs_history)
        if self.control_type == 'neural':
            # x = self.scaler.transform(x)
            obs_history = np.float32(self.obs_history.copy())
            obs_history.shape = (self.samples_n*20, self.state_dim)
            obs_history = self.scaler.transform(obs_history)
            obs_history.shape = (self.samples_n, 20, self.state_dim)

            act_long = self.model(obs_history).numpy()[0][0]
            print(act_long)
            return act_long
        elif self.time_lapse > 25:
            self.control_type = 'neural'

        act_long = self.idm_action(obs[-2:])
        return act_long
