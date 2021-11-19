# from factory.vehicles import  *
import numpy as np
import pickle


class Vehicle(object):
    STEP_SIZE = 0.1 # s update rate - used for dynamics
    def __init__(self, id, lane_id, x, v):
        self.v = v # longitudinal speed [m/s]
        self.lane_id = lane_id # inner most right is 1
        self.y_lane = 0. # lane relative
        self.x = x # global coordinate
        self.id = id # 'sdv' or any other integer
        self.y = 2*lane_id*1.85-1.85

    def act(self):
        """
        :param high-lev decision of the car
        """
        pass

    def observe(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

class SDVehicle(Vehicle):
    OPTIONS = {
                0: ['LK', 'UP'],
                1: ['LK', 'DOWN'],
                2: ['LK', 'IDLE'],
                3: ['LCL', 'UP'],
                4: ['LCL', 'DOWN'],
                5: ['LCL', 'IDLE'],
                6: ['LCR', 'UP'],
                7: ['LCR', 'DOWN'],
                8: ['LCR', 'IDLE']
                }

    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        # self.id = id
        # self.lane_y = 3

        self.initialize_agent()

    def initialize_agent(self, config=None):
        self.time_budget = 0
        self.samples_n = 10 # number of samples. For filtering, this is only needed for visualising belief.
        history_len = 20 # steps
        state_dim = 9
        self.obs_history = np.zeros([self.samples_n, history_len, 9])
        self.action_history = [[0., 0.]]*20
        # load
        model_name = 'testing_car'
        exp_dir = './models/experiments/'+model_name+'/model'
        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        from models.core.driver_model import  NeurIDMModel
        model = NeurIDMModel()
        model.load_weights(exp_dir).expect_partial()
        self.h_seq_encoder = model.h_seq_encoder
        self.act_encoder = model.act_encoder
        self.belief_net = model.belief_net
        self.arbiter = model.arbiter
        self.idm_layer = model.idm_layer
        self.arbiter.attention_temp = 20

    def act(self, decision, obs):
        self.time_budget = 20 # steps

        if self.OPTIONS[decision][0] == 'LK':
            act_lat = 0.
        elif self.OPTIONS[decision][0] == 'LCL':
            act_lat = 0.7
        elif self.OPTIONS[decision][0] == 'LCR':
            act_lat = -0.7

        if self.OPTIONS[decision][1] == 'IDLE':
            act_long = 0.
        elif self.OPTIONS[decision][1] == 'UP':
            act_long = 1.
        elif self.OPTIONS[decision][1] == 'DOWN':
            act_long = -1.

        if self.time_budget >= 0:
            self.time_budget -= 0.1
        else:
            self.time_budget = 1


        return  [act_long, act_lat]

    def step(self, action):
        act_long, act_lat = action
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE **2

        self.v = self.v + act_long * self.STEP_SIZE

        if self.y_lane < -1.85:
            # self.lane_id = 1
            self.y_lane = 1.85

        self.y = self.y + act_lat*self.STEP_SIZE
        self.y_lane = self.y_lane + act_lat*self.STEP_SIZE


    def update_obs_history(self, o_t):
        # print(self.obs_history)
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def update_action_history(self, act_lat):
        # print(self.action_history)
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = [self.y_lane, act_lat]

    def get_action_plan(self, act_lat):
        action_plan = []
        y_lane = self.y_lane
        for i in range(20):
            y_lane = y_lane + act_lat*0.1
            action_plan.append([y_lane, act_lat])
        return action_plan

    def get_belief(self, obs_history, action_plan):
        """
        Belief(s_t+1) <== obs, actions
        """
        actions = self.action_history + action_plan
        # print(actions)
        # print('obs_history')
        # print(obs_history)
        obs_history = np.float32(np.array(obs_history))
        obs_history.shape = (self.samples_n*20, 9)
        obs_history[:, :-2] = self.scaler.transform(obs_history[:, :-2])
        obs_history.shape = (self.samples_n, 20, 9)

        actions = np.float32(np.array(actions))
        actions.shape = (1, 40, 2)
        actions = np.repeat(actions, self.samples_n, axis=0)

        enc_h = self.h_seq_encoder(obs_history)
        enc_acts = self.act_encoder(actions)
        prior_param = self.belief_net([enc_h, enc_acts], dis_type='prior')
        return prior_param, enc_h
