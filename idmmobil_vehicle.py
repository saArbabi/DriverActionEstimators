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
        self.lane_width = 3.7

    def observe(self):
        raise NotImplementedError

    def step(self, actions):
        act_long, act_lat = actions
        self.glob_x +=  self.speed * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE **2
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


class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        # self.capability = 'IDM'
        self.lane_id = lane_id
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.neighbours = {}
        self.perception_range = 100 #m
        self.lane_width = 3.8

        self.lateral_actions = {'move_left':0.7,
                                'move_right':-0.7,
                                'keep_lane':0
                                }
        self.set_idm_params(aggressiveness)


    def set_idm_params(self, aggressiveness):
        if aggressiveness == None:
            raise ValueError('No aggressiveness specified!')

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
        self.driver_params['desired_tgap'] = self.get_idm_param(Parameter_range, 'desired_tgap')
        self.driver_params['min_jamx'] = self.get_idm_param(Parameter_range, 'min_jamx')
        self.driver_params['max_act'] = self.get_idm_param(Parameter_range, 'max_act')
        self.driver_params['min_act'] = self.get_idm_param(Parameter_range, 'min_act')
        # MOBIL params
        self.driver_params['politeness'] = self.get_idm_param(Parameter_range, 'politeness')
        self.driver_params['safe_braking'] = self.get_idm_param(Parameter_range, 'safe_braking')
        self.driver_params['act_threshold'] = self.get_idm_param(Parameter_range, 'act_threshold')

        if 0 <= self.driver_params['aggressiveness'] < 0.33:
            # timid driver
            attentiveness = 0.5*self.lane_width*np.random.beta(2, 10)
        elif 0.33 <= self.driver_params['aggressiveness'] <= 0.66:
            # normal driver
            attentiveness = 0.5*self.lane_width*np.random.beta(3, 3)
        elif 0.66 < self.driver_params['aggressiveness']:
            # aggressive driver
            attentiveness = 0.5*self.lane_width*np.random.beta(10, 2)
        self.driver_params['attentiveness'] = round(attentiveness, 1)

        # self.driver_params['desired_v'] += np.random.normal(0, 1)
    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act', 'safe_braking']:
            # the larger the param, the more aggressive the driver
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.driver_params['aggressiveness']*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness', 'act_threshold']:
            # the larger the param, the more timid the driver
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  max_value - self.driver_params['aggressiveness']*(max_value-min_value)


    def get_desired_gap(self, delta_v):
        gap = self.driver_params['min_jamx'] + \
                                            self.driver_params['desired_tgap']*\
                                            self.speed+(self.speed*delta_v)/ \
                                            (2*np.sqrt(self.driver_params['max_act']*\
                                            self.driver_params['min_act']))
        return max([0, gap])

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 1000]
        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        return [delta_v, delta_x]

    def idm_actions(self, obs):
        delta_v, delta_x = obs
        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.driver_params['max_act']*(1-(self.speed/self.driver_params['desired_v'])**4-\
                                            (desired_gap/(delta_x+1e-5))**2)

        return round(act_long, 2)

    def check_reservations(self, target_lane, reservations):
        """To ensure two cars do not simultaneously move into the same lane.
        """
        if not reservations:
            return True
        else:
            for reserved in reservations.values():
                reserved_lane, max_glob_x, min_glob_x = reserved
                if target_lane == reserved_lane and min_glob_x < self.glob_x < max_glob_x:
                    return False
            return True

    def check_neighbours(self, neighbours):
        """To ensure neighbours keep lane while merger is changing lane.
        """
        for vehicle in neighbours.values():
            if vehicle and vehicle.lane_decision != 'keep_lane':
                return False
        return True

    def mobil_condition(self, actions_gains):
        """To decide if changing lane is worthwhile.
        """
        ego_gain, new_follower_gain, old_follower_gain = actions_gains
        lc_condition = ego_gain+self.driver_params['politeness']*(new_follower_gain+\
                                                                old_follower_gain )
        return lc_condition


    def act(self, neighbours, reservations):
        act_long = self.idm_actions(self.observe(self, neighbours['f']))
        if self.lane_decision == 'move_left':
            if self.lane_id == self.target_lane :
                if self.lane_y >= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'move_right':
            if self.lane_id == self.target_lane :
                if self.lane_y <= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'keep_lane' and self.check_neighbours(neighbours):
            # if not :
            # # if not self.check_neighbours(neighbours) or self.glob_x < 100:
            #     # Keep lane if neighbours are chanigng lane or if you have just entered
            #     # highway
            #     pass

            lc_left_condition = 0
            lc_right_condition = 0

            act_rl_lc = self.idm_actions(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_actions(self.observe(neighbours['rr'], self))
            act_r_lc = self.idm_actions(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_actions(self.observe(neighbours['r'], self))
            old_follower_gain = act_r_lc-act_r_lk

            if self.lane_id > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_actions(self.observe(neighbours['rl'], neighbours['fl']))
                act_ego_lc_l = self.idm_actions(self.observe(self, neighbours['fl']))
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if self.lane_id < self.lanes_n and \
                                                self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_actions(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_actions(self.observe(neighbours['rr'], neighbours['fr']))

                ego_gain = act_ego_lc_r-act_long
                new_follower_gain = act_rr_lc-act_rr_lk
                lc_right_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if max([lc_left_condition, lc_right_condition]) > self.driver_params['act_threshold']:
                if lc_left_condition > lc_right_condition:
                    target_lane = self.target_lane - 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_left'
                        self.target_lane -= 1
                        return [act_ego_lc_l, self.lateral_actions[self.lane_decision]]

                elif lc_left_condition < lc_right_condition:
                    target_lane = self.target_lane + 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_right'
                        self.target_lane += 1
                        return [act_ego_lc_r, self.lateral_actions[self.lane_decision]]

        return [act_long, self.lateral_actions[self.lane_decision]]