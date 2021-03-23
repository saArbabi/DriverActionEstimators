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
        self.fig = plt.figure(figsize=(10, 3))
        self.env_ax = self.fig.add_subplot(211)
        self.tree_info = None
        self.belief_info = None
        self.decision_counts = None
        #TODO: option to record video

    def draw_road(self, ax, percept_origin, env_clock):
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
        plt.title('Elapsed time: '+str(round(env_clock, 1)))

    def draw_vehicles(self, ax, vehicles):
        xs = [veh.x for veh in vehicles if veh.id != 'sdv']
        ys = [veh.y for veh in vehicles if veh.id != 'sdv']
        ax.scatter(xs, ys, s=100, marker=">", color='grey')
        for veh in vehicles:
            vehicle_color = 'grey'
            if veh.id == 'neural':
                if veh.control_type != 'idm':
                    vehicle_color = 'green'
            if veh.id == 'idm':
                vehicle_color = 'purple'

            ax.scatter(veh.x, veh.y, s=100, marker=">", color=vehicle_color)
            ax.annotate(round(veh.v, 1), (veh.x, veh.y+0.1))
            # ax.annotate(round(veh.v, 1), (veh.x, veh.y+0.1))

    def draw_env(self, ax, vehicles, env_clock):
        ax.clear()
        self.draw_road(ax, percept_origin = vehicles[0].x, env_clock=env_clock)
        self.draw_vehicles(ax, vehicles)

    def update_plots(self, vehicles, env_clock):
        self.draw_env(self.env_ax, vehicles, env_clock)
        plt.pause(0.00000000000000000000001)
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
                        'percept_range':500, # m, front and back
                        }

    def step(self, decision=None):
        # low-level actions currently not obs dependant
        for vehicle in self.vehicles:#
            action = vehicle.act()
            vehicle.step(action)
        self.env_clock += 0.1

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.config)
            # self.viewer.PAUSE_CONTROL = PAUSE_CONTROL
        self.viewer.update_plots(self.vehicles, self.env_clock)

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
        return 1.5*np.sin(self.x*0.01)

class IDMVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, driver_type=None):
        super().__init__(id, lane_id, x, v)
        self.set_idm_params(driver_type)

    def set_idm_params(self, driver_type):
        normal_idm = {
                        'desired_v':25, # m/s
                        'desired_tgap':1.5, # s
                        'min_jamx':2, # m
                        'max_acc':1.4, # m/s^2
                        'max_decc':2, # m/s^2
                        }

        timid_idm = {
                        'desired_v':19.4, # m/s
                        'desired_tgap':2, # s
                        'min_jamx':4, # m
                        'max_acc':0.8, # m/s^2
                        'max_decc':1, # m/s^2
                        }

        aggressive_idm = {
                        'desired_v':30, # m/s
                        'desired_tgap':1, # s
                        'min_jamx':0, # m
                        'max_acc':2, # m/s^2
                        'max_decc':3, # m/s^2
                        }
        if not driver_type:
            raise ValueError('No driver_type specified')

        if driver_type == 'normal':
            idm_param = normal_idm
        if driver_type == 'timid':
            idm_param = timid_idm
        if driver_type == 'aggressive':
            idm_param = aggressive_idm

        self.desired_v = idm_param['desired_v']
        self.desired_tgap = idm_param['desired_tgap']
        self.min_jamx = idm_param['min_jamx']
        self.max_acc = idm_param['max_acc']
        self.max_decc = idm_param['max_decc']

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_acc*self.max_decc))
        return gap

    def act(self):
        obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        acc = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        return acc

class NeurVehicle(IDMVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type)
        self.obs_history = []
        self.control_type = 'idm'
        self.policy = model

    def act(self):
        pass

class DNNVehicle(NeurVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type, model)

    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v, self.v - self.lead_vehicle.v, self.lead_vehicle.x-self.x])

        if len(self.obs_history) % 30 == 0:
            self.control_type = 'neural'
            x = np.array([self.obs_history[-1]])
            x = self.scaler.transform(x)

            action = self.policy(x).numpy()[0][0]
            self.obs_history.pop(0)

        else:
            obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            action = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)
        return action

class LSTMIDMVehicle(NeurVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type, model)

    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v, self.v - self.lead_vehicle.v, self.lead_vehicle.x-self.x])

        # if len(self.obs_history) % 30 == 0:
        if len(self.obs_history) % 30 == 0 and self.control_type == 'idm':
            self.control_type = 'neural'
            x = np.array(self.obs_history)
            x_scaled = self.scaler.transform(x)

            x_scaled.shape = (1, 30, 4)
            x.shape = (1, 30, 4)
            param = self.policy([x_scaled, x]).numpy()[0]
            self.obs_history.pop(0)

            self.desired_v = param[0]
            self.desired_tgap = param[1]
            self.min_jamx = param[2]
            self.max_acc = param[3]
            self.max_decc = param[4]

        obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        action = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        return action

class LSTMVehicle(NeurVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type, model)

    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v, self.v - self.lead_vehicle.v, \
                                            self.lead_vehicle.x-self.x])

        if len(self.obs_history) % 30 == 0:
            self.control_type = 'neural'
            x = np.array(self.obs_history)
            x = self.scaler.transform(x)

            x.shape = (1, 30, 4)
            action = self.policy(x).numpy()[0][0]
            self.obs_history.pop(0)

        else:
            obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            action = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)
        return action
"""
vis
"""
# from models.neural import  Encoder

# from models.dnn import  Encoder
def set_follower(model_name, driver_type):
    config = {
             "model_config": {
                 "learning_rate": 1e-3,
                "batch_size": 50,
                },
                "exp_id": "NA",
                "Note": ""}

    model_type = model_name[:-6]
    exp_dir = './models/experiments/'+model_name+'/model_dir'

    with open('./models/experiments/scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)

    if model_type == 'dnn':
        from models.dnn import  Encoder
        model = Encoder(config)
        model.load_weights(exp_dir)
        follower = DNNVehicle(id='neural', lane_id=1, x=40, v=20,
                        driver_type=driver_type, model=model)

    if model_type == 'lstm':
        from models.lstm import  Encoder
        model = Encoder(config)
        # model = Encoder(config, model_use='inference')
        model.load_weights(exp_dir)
        follower = LSTMVehicle(id='neural', lane_id=1, x=40, v=20,
                        driver_type=driver_type, model=model)

    if model_type == 'lstmidm':
        from models.idm_neural import  Encoder
        model = Encoder(config, model_use='inference')
        # model = Encoder(config, model_use='inference')
        model.load_weights(exp_dir)
        follower = LSTMIDMVehicle(id='neural', lane_id=1, x=40, v=20,
                        driver_type=driver_type, model=model)

    follower.scaler = scaler
    return follower

# from models.dnn import  Encoder
from sklearn import preprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
os.chdir('./sim')
env = Env()
leader = LeadVehicle(id='leader', lane_id=1, x=100, v=20)
# follower_neural = set_follower(model_name='dnn_03')
# follower_neural = set_follower(model_name='lstm')
# follower_neural = set_follower(model_name='lstmidm')
driver_type = 'normal'
# driver_type = 'aggressive'
# follower_neural = set_follower(model_name='lstmidm_03', driver_type=driver_type)
# follower_neural = set_follower(model_name='dnn_01', driver_type=driver_type)
follower_neural = set_follower(model_name='lstmidm_sq_01', driver_type=driver_type)
# follower_neural = set_follower(model_name='lstmidm_01', driver_type=driver_type)
# follower_neural = set_follower(model_name='lstmidm_6s_03', driver_type=driver_type)
# follower_neural = set_follower(model_name='lstm_01')
follower_IDM = IDMVehicle(id='idm', lane_id=1, x=40, v=20, driver_type=driver_type)
# neural = NeurIDMVehicle(id='neural', lane_id=1, x=40, v=25)
# neural = NeurVehicle(id='neural', lane_id=1, x=40, v=25)
# neural = DNNVehicle(id='neural', lane_id=1, x=40, v=16)
follower_IDM.lead_vehicle = leader
follower_neural.lead_vehicle = leader
# env.vehicles = [leader, follower_IDM]
env.vehicles = [leader, follower_IDM, follower_neural]
obs = env.reset()
env.render()

for i in range(1000):
    env.render()
    env.step()
    if i % 100 == 0:
        answer = input('Continue?')
        if answer == 'n':
            plt.close()
            break
plt.show()
# %%
