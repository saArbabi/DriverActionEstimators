# import numpy as np
# from tensorflow.keras import layers
# from tensorflow import keras
# import pandas as pd
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from model import Encoder
# from importlib import reload
# import tensorflow as tf
# import time
# %%
"""
lead vehicle
"""
class Viewer():
    def __init__(self, env_config):
        self.env_config  = env_config
        self.fig = plt.figure(figsize=(7, 7))
        self.env_ax = self.fig.add_subplot(211)
        self.tree_info = None
        self.belief_info = None
        self.decision_counts = None
        #TODO: option to record video

    def draw_road(self, ax, percept_origin):
        lane_cor = self.env_config['lane_width']*self.env_config['lane_count']
        ax.hlines(0, 0, self.env_config['lane_length'], colors='k', linestyles='solid')
        ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                                                    colors='k', linestyles='solid')

        if self.env_config['lane_count'] > 1:
            lane_cor = self.env_config['lane_width']
            for lane in range(self.env_config['lane_count']-1):
                ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                                                        colors='k', linestyles='--')
                lane_cor += self.env_config['lane_width']

        if percept_origin < self.env_config['percept_range']:
            ax.set_xlim(0, self.env_config['percept_range']*2)
        else:
            ax.set_xlim(percept_origin - self.env_config['percept_range'],
                                percept_origin + self.env_config['percept_range'])

        ax.set_yticks([])
        # plt.show()

    def draw_vehicles(self, ax, vehicles):
        xs = [veh.x for veh in vehicles if veh.id != 'sdv']
        ys = [veh.y for veh in vehicles if veh.id != 'sdv']
        ax.scatter(xs, ys, s=100, marker=">", color='grey')
        for veh in vehicles:
            vehicle_color = 'grey'
            if veh.id == 'neural':
                if veh.control_type != 'idm':
                    vehicle_color = 'green'

            ax.scatter(veh.x, veh.y, s=100, marker=">", color=vehicle_color)
            ax.annotate(round(veh.v, 1), (veh.x, veh.y+0.1))

    def draw_env(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax, percept_origin = vehicles[0].x)
        self.draw_vehicles(ax, vehicles)

    def update_plots(self, vehicles):
        self.draw_env(self.env_ax, vehicles)
        plt.pause(0.00001)
        # plt.show()

class Env():
    def __init__(self):
        self.viewer = None
        self.vehicles = [] # all vehicles
        self.env_clock = 0 # time past since the episode's start
        self.sdv = None
        self.default_config()
        self.seed()

    def seed(self, seed_value=2021):
        for veh in self.vehicles:
            if veh.id != 'sdv':
                veh.seed(seed_value)

    def reset(self):
        if self.sdv:
            return  self.sdv.observe(self.vehicles)

    def default_config(self):
        self.config = {'lane_count':2,
                        'lane_width':3.7, # m
                        'lane_length':10000, # m
                        'percept_range':70, # m, front and back
                        }

    def step(self, decision=None):
        # low-level actions currently not obs dependant
        for vehicle in self.vehicles:#
            action = vehicle.act()
            vehicle.step(action)

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.config)
            # self.viewer.PAUSE_CONTROL = PAUSE_CONTROL
        self.viewer.update_plots(self.vehicles)

    def random_value_gen(self, min, max):
        if min == max:
            return min
        val_range = range(min, max)
        return np.random.choice(val_range)


class Vehicle(object):
    STEP_SIZE = 0.1 # s update rate - used for dynamics
    def __init__(self, id, lane_id, x, v):
        self.v = v # longitudinal speed [m/s]
        self.lane_id = lane_id # inner most right is 1
        self.y = 0 # lane relative
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

    def step(self, action):
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * action * self.STEP_SIZE **2

        self.v = self.v + action * self.STEP_SIZE

class LeadVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)

    def act(self):
        # return 0
        return 0.7*np.sin(self.x*0.05)

class IDMVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        if not idm_param:
            self.default_config()
        else:
            self.idm_param = idm_param
        self.seed()
        # TODO params can also be learned

    def seed(self, seed_value=2021):
        self.rng = np.random.RandomState()
        self.rng.seed(seed_value)

    def default_config(self):
        # TODO nonstationary params
        self.idm_param = {
                        'desired_v':12, # m/s
                        'desired_tgap':3, # s
                        'min_jamx':2, # m
                        'max_acc':1.4, # m/s^2
                        'max_decc':2, # m/s^2
                        }

        self.desired_v = self.idm_param['desired_v']
        self.desired_tgap = self.idm_param['desired_tgap']
        self.min_jamx = self.idm_param['min_jamx']
        self.max_acc = self.idm_param['max_acc']
        self.max_decc = self.idm_param['max_decc']

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_acc*self.max_decc))
        return gap

    def act(self):
        obs = {'dv':self.lead_vehicle.v-self.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        acc = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        return acc
        # return [0, 0]
        # return [np.random.uniform(-2,2), 0]

class NeurVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        self.obs_history = []
        self.control_type = 'idm'
        config = {
         "model_config": {
             "learning_rate": 1e-3,
            "batch_size": 50,
            },
            "exp_id": "NA",
            "Note": ""}
        self.loadModel(config)
        self.default_config()

    def default_config(self):
        # TODO nonstationary params
        self.idm_param = {
                        'desired_v':12, # m/s
                        'desired_tgap':1.5, # s
                        'min_jamx':2, # m
                        'max_acc':1.4, # m/s^2
                        'max_decc':2, # m/s^2
                        }

        self.desired_v = self.idm_param['desired_v']
        self.desired_tgap = self.idm_param['desired_tgap']
        self.min_jamx = self.idm_param['min_jamx']
        self.max_acc = self.idm_param['max_acc']
        self.max_decc = self.idm_param['max_decc']

    def loadModel(self, config):
        checkpoint_dir = './sim/experiments/neural_2/model_dir'
        # checkpoint_dir = './experiments/model_dir'
        self.policy = Encoder(config)
        Checkpoint = tf.train.Checkpoint(net=self.policy)
        # Checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        Checkpoint.restore(checkpoint_dir+'/ckpt-1')

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_acc*self.max_decc))
        return gap

    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v-self.v, self.lead_vehicle.x-self.x])

        if len(self.obs_history) % 30 == 0:
            self.control_type = 'neural'
            input = np.array(self.obs_history)
            input.shape = (1, 30, 3)
            action = self.policy(input).numpy()[0][0]
            self.obs_history.pop(0)

        else:
            obs = {'dv':self.lead_vehicle.v-self.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            action = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)
        return action
        # return np.sin(self.x*0.05)

class NeurIDMVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        self.obs_history = []
        self.control_type = 'idm'
        config = {
         "model_config": {
             "learning_rate": 1e-3,
            "batch_size": 50,
            },
            "exp_id": "NA",
            "Note": ""}
        self.loadModel(config)
        self.default_config()

    def default_config(self):
        # TODO nonstationary params
        self.idm_param = {
                        'desired_v':12, # m/s
                        'desired_tgap':3, # s
                        'min_jamx':2, # m
                        'max_acc':1.4, # m/s^2
                        'max_decc':2, # m/s^2
                        }

        self.desired_v = self.idm_param['desired_v']
        self.desired_tgap = self.idm_param['desired_tgap']
        self.min_jamx = self.idm_param['min_jamx']
        self.max_acc = self.idm_param['max_acc']
        self.max_decc = self.idm_param['max_decc']

    def loadModel(self, config):
        checkpoint_dir = './sim/experiments/idm_neural_2/model_dir'
        # checkpoint_dir = './experiments/model_dir'
        self.policy = Encoder(config, model_use='inference')
        Checkpoint = tf.train.Checkpoint(net=self.policy)
        # Checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        Checkpoint.restore(checkpoint_dir+'/ckpt-1')

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_acc*self.max_decc))
        return gap

    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v-self.v, self.lead_vehicle.x-self.x])

        if len(self.obs_history) % 30 == 0:
        # if len(self.obs_history) == 30:
            self.control_type = 'neural'
            input = np.array(self.obs_history)
            input.shape = (1, 30, 3)
            param = self.policy(input).numpy()[0]
            self.obs_history.pop(0)

            self.desired_v = param[0]
            self.desired_tgap = param[1]
            self.min_jamx = param[2]
            self.max_acc = param[3]
            self.max_decc = param[4]

        obs = {'dv':self.lead_vehicle.v-self.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        action = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        return action
        # return np.sin(self.x*0.05)

"""
vis
"""
# from models.neural import  Encoder
from models.idm_neural import  Encoder

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

env = Env()
leader = LeadVehicle(id='leader', lane_id=1, x=100, v=10)
follower = IDMVehicle(id='follower', lane_id=1, x=40, v=10)
neural = NeurIDMVehicle(id='neural', lane_id=1, x=40, v=10)
# neural = NeurVehicle(id='neural', lane_id=1, x=40, v=10)
follower.lead_vehicle = leader
neural.lead_vehicle = leader
env.vehicles = [leader, follower, neural]
obs = env.reset()
env.render()

for i in range(500):
    env.render()
    env.step()
# %%
