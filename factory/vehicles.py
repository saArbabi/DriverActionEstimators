import numpy as np


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
        return 1.5*np.sin(self.x*0.04)

class MergeVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)

    def act(self):
        # return 0
        return 1.5*np.sin(self.x*0.04)

class IDMVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, driver_type=None):
        super().__init__(id, lane_id, x, v)
        self.set_idm_params(driver_type)

    def set_idm_params(self, driver_type):
        normal_idm = {
                        'desired_v':25, # m/s
                        'desired_tgap':1.5, # s
                        'min_jamx':2, # m
                        'max_act':1.4, # m/s^2
                        'min_act':2, # m/s^2
                        }

        timid_idm = {
                        'desired_v':19.4, # m/s
                        'desired_tgap':2, # s
                        'min_jamx':4, # m
                        'max_act':0.8, # m/s^2
                        'min_act':1, # m/s^2
                        }

        aggressive_idm = {
                        'desired_v':30, # m/s
                        'desired_tgap':1, # s
                        'min_jamx':0, # m
                        'max_act':2, # m/s^2
                        'min_act':3, # m/s^2
                        }
        if not driver_type:
            raise ValueError('No driver_type specified')

        if driver_type == 'normal_idm':
            idm_param = normal_idm
        if driver_type == 'timid_idm':
            idm_param = timid_idm
        if driver_type == 'aggressive_idm':
            idm_param = aggressive_idm

        self.desired_v = idm_param['desired_v']
        self.desired_tgap = idm_param['desired_tgap']
        self.min_jamx = idm_param['min_jamx']
        self.max_act = idm_param['max_act']
        self.min_act = idm_param['min_act']

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_act*self.min_act))
        return gap

    def act(self):
        obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        acc = self.max_act*(1-(self.v/self.desired_v)**4-\
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
            # x = self.scaler.transform(x)
            action = self.policy(x).numpy()[0][0]
            self.obs_history.pop(0)

        else:
            obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            action = self.max_act*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)
        return action

class LSTMIDMVehicle(NeurVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type, model)
        self.elapsed_time = 0


    def act(self):
        self.obs_history.append([self.v, self.lead_vehicle.v, \
        self.v - self.lead_vehicle.v, self.lead_vehicle.x-self.x])

        steps = 20
        if len(self.obs_history) % steps == 0:
        # if len(self.obs_history) % steps == 0 and self.control_type == 'idm':
            self.control_type = 'neural'
            x = np.array(self.obs_history)
            # x_scaled = self.scaler.transform(x)

            # x_scaled.shape = (1, steps, 5)
            x.shape = (1, steps, 4)
            self.obs_history.pop(0)

            # if round(self.elapsed_time, 1) % 10 == 0:
            param = self.policy([x, x]).numpy()[0]
            self.desired_v = param[0]
            self.desired_tgap = param[1]
            self.min_jamx = param[2]
            self.max_act = param[3]
            self.min_act = param[4]
                # print(param)

            self.elapsed_time += 0.1

        obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        action = self.max_act*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        self.action = action
        return action

class NeurIDM(NeurVehicle):
    def __init__(self, id, lane_id, x, v, driver_type, model):
        super().__init__(id, lane_id, x, v, driver_type, model)
        self.elapsed_time = 0
        self.action = 0
        self.time_switched = 0
        self.attention_score = 0.5
        self.attention = 1
        self.samples_n = 20
        self.steps_n = 20
        self.s_unscaled = np.zeros([self.samples_n, self.steps_n, 9])
        self.s_scaled = np.zeros([self.samples_n, self.steps_n, 9])
        self.obs = np.zeros([self.samples_n, 9])
        self.env_clock = 0
        self.y = np.ones([self.samples_n, 1])*self.y
        self.x = np.ones([self.samples_n, 1])*self.x
        self.v = np.ones([self.samples_n, 1])*self.v

    def observe(self, attention):
        lf_dx = self.lead_vehicle.x-self.x
        mf_dx = self.merge_vehicle.x-self.x
        # mf_dx = lf_dx - 5
        fl_dv = self.v - self.lead_vehicle.v
        fm_dv = self.v - self.merge_vehicle.v
        leader_feature = [self.lead_vehicle.v, fl_dv, lf_dx]
        # print('lf_dx: ', lf_dx)
        # print('mf_dx: ', mf_dx)
        merger_feature = [self.merge_vehicle.v, fm_dv, mf_dx]
        # merger_feature = [self.merge_vehicle.v, fm_dv, mf_dx]
        m_exists = 1
        self.obs[:, 0:2] = [m_exists, attention]
        collection = [self.v]
        collection.extend(leader_feature)
        collection.extend(merger_feature)
        # print(self.x)

        for i in range(7):
            # print(i)
            self.obs[:, 2+i:3+i] = collection[i]


        # print('OBS:  ', self.obs)

        # return obs

    def act(self):
        self.observe(self.attention)
        # self.obs_history.append(latest_obs)
        # if self.elapsed_time == 0:

        self.s_unscaled[:, :-1, :] = self.s_unscaled[:, 1:, :]
        self.s_unscaled[:, -1, :] = self.obs[:, :]
        # print(self.obs_history)

        self.s_scaled[:, :-1, :] = self.s_scaled[:, 1:, :]
        scaled_obs = self.obs.copy()
        scaled_obs[:, 2:] = self.scaler.transform(scaled_obs[:, 2:])
        self.s_scaled[:, -1, :] = scaled_obs[:, :]
        print(scaled_obs)
        self.env_clock += 1

        if self.env_clock > 20:
            self.control_type = 'neural'
            x_scaled = self.s_unscaled.copy()
            print(x_scaled.shape)
            param, alpha = self.policy([self.s_scaled, self.s_scaled[:,-1:,:], self.s_unscaled])

            self.attention_score = alpha.numpy()
            print('alpha-shape: ', self.attention_score.shape)
            # # for item in param:
            # #     print(item.numpy()[0])
            # # print(latest_obs)
            print('param-shape  ', param[0].numpy().shape)
            param = [item.numpy() for item in param]
            #
            # if env_clock % 1 == 0:
                # print('self.elapsed_time: ', round(self.elapsed_time, 1))
            self.desired_v = param[0]
            self.desired_tgap = param[1]
            self.min_jamx = param[2]
            self.max_act = param[3]
            self.min_act = param[4]

            print('desired_v: ', self.desired_v.mean())
            # print('desired_tgap: ', self.desired_tgap)
            # print('min_jamx: ', self.min_jamx)
            # print('max_act: ', self.max_act)
            # print('min_act: ', self.min_act)

            obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            fl_act = self.max_act*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)

            obs = {'dv':self.v-self.merge_vehicle.v, 'dx':self.merge_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            fm_act = self.max_act*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)


            self.action = fl_act*self.attention_score + fm_act*(1-self.attention_score)
            # print('act-shapefl  ', fl_act.shape)
            # print('act-shape  ', self.action.shape)
            # print(self.action)
            print(self.attention_score.mean())
            print('1', fl_act.mean())
            print('2', fm_act.mean())
            # print(self.obs)
            # self.elapsed_time += 0.1
            return self.action

        obs = {'dv':self.v-self.attend_veh.v, 'dx':self.attend_veh.x-self.x}
        desired_gap = self.get_desired_gap(obs['dv'])
        action = self.max_act*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)

        self.action = action

        # print(action)
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
            # x = self.scaler.transform(x)
            x.shape = (1, 30, 4)
            action = self.policy(x).numpy()[0][0]
            self.obs_history.pop(0)

        else:
            obs = {'dv':self.v-self.lead_vehicle.v, 'dx':self.lead_vehicle.x-self.x}
            desired_gap = self.get_desired_gap(obs['dv'])
            action = self.max_act*(1-(self.v/self.desired_v)**4-\
                                                (desired_gap/obs['dx'])**2)
        return action
