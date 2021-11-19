import numpy as np
np.random.seed(2020)

class Vehicle(object):
    STEP_SIZE = 0.1
    def __init__(self, id, lane_id, glob_x, speed):
        self.id = id
        self.lane_id = lane_id
        self.glob_x = glob_x
        # self.glob_y = glob_y
        self.speed = speed
        self.lane_y = 0
        self.lane_width = 3.75

    def observe(self):
        raise NotImplementedError

    def step(self, actions):
        act_long, act_lat = actions
        self.glob_x +=  self.speed * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE**2
        self.speed +=  act_long * self.STEP_SIZE
        if act_lat != 0:
            self.glob_y +=  act_lat*self.STEP_SIZE
            self.lane_y +=  act_lat*self.STEP_SIZE

            if self.lane_y <= -self.lane_width/2:
                # just stepped into right lane
                self.lane_id  += 1
                self.lane_y += self.lane_width
            elif self.lane_y >= self.lane_width/2:
                # just stepped into left lane
                self.lane_id  -= 1
                self.lane_y -= self.lane_width

    def act(self):
        raise NotImplementedError


class IDMVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        # self.capability = 'IDM'
        self.lane_id = lane_id
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.neighbours = {'att': None}
        self.perception_range = 1000 #m
        self.lane_width = 3.75
        self.act_long = 0

        if aggressiveness != None:
            self.set_idm_params(aggressiveness)

    def set_idm_params(self, aggressiveness):
        Parameter_range = {'most_aggressive': {
                                        'desired_v':30, # m/s
                                        'desired_tgap':1, # s
                                        'min_jamx':0, # m
                                        'max_act':2, # m/s^2
                                        'min_act':3, # m/s^2
                                        'politeness':0,
                                        'safe_braking':-3,
                                        'act_threshold':0
                                        },
                            'least_aggressvie': {
                                        'desired_v':19.4, # m/s
                                        'desired_tgap':2, # s
                                        'min_jamx':4, # m
                                        'max_act':0.8, # m/s^2
                                        'min_act':1, # m/s^2
                                        'politeness':1,
                                        'safe_braking':-1,
                                        'act_threshold':0.2
                                         }}

        self.driver_params = {}
        self.driver_params['aggressiveness'] = aggressiveness  # in range [0, 1]
        # IDM params
        self.driver_params['desired_v'] = self.get_idm_param(Parameter_range, 'desired_v')
        self.driver_params['desired_v'] += np.random.normal()
        self.driver_params['desired_tgap'] = self.get_idm_param(Parameter_range, 'desired_tgap')
        self.driver_params['min_jamx'] = self.get_idm_param(Parameter_range, 'min_jamx')
        self.driver_params['max_act'] = self.get_idm_param(Parameter_range, 'max_act')
        self.driver_params['min_act'] = self.get_idm_param(Parameter_range, 'min_act')

    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act']:
            # the larger the param, the more aggressive the driver
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.driver_params['aggressiveness']*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx']:
            # the larger the param, the more timid the driver
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  max_value - self.driver_params['aggressiveness']*(max_value-min_value)

    def my_neighbours(self, vehicles):
        neighbours = {}
        delta_xs_f = [self.perception_range]
        candidate_f = None

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = abs(vehicle.glob_x-self.glob_x)
                if vehicle.lane_id == self.lane_id and vehicle.glob_x > self.glob_x \
                                                    and delta_x < self.perception_range:
                    if delta_x < min(delta_xs_f):
                        delta_xs_f.append(delta_x)
                        candidate_f = vehicle

        neighbours['att'] = candidate_f
        return neighbours

    def get_desired_gap(self, delta_v):
        desired_gap = self.driver_params['min_jamx'] + \
                        max(0,
                        self.driver_params['desired_tgap']*\
                        self.speed+(self.speed*delta_v)/ \
                        (2*np.sqrt(self.driver_params['max_act']*\
                        self.driver_params['min_act']))
                        )

        return desired_gap

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 1000]
        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        assert delta_x > 0
        return [delta_v, delta_x]

    def idm_action(self, obs):
        delta_v, delta_x = obs
        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.driver_params['max_act']*(1-(self.speed/self.driver_params['desired_v'])**4-\
                                            (desired_gap/(delta_x))**2)

        return act_long
