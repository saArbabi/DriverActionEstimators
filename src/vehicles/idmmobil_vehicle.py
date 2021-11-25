import numpy as np
from vehicles.vehicle import Vehicle

class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        if id:
            super().__init__(id, lane_id, glob_x, speed)
            self.beta_precision = 15
            self.lane_id = lane_id
            self.lane_width = 3.75
            self.lanes_n = 2
            self.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
            self.target_lane = lane_id
            self.lane_decision = 'keep_lane'
            self.neighbours = {veh_name: None for veh_name in\
                                ['f', 'fl', 'rl', 'r', 'rr', 'fr', 'm', 'att']}
            self.perception_range = 500 #m
            self.act_long = 0
            self.time_lapse = 0 # since vehicle came to existance
            self.vehicle_type = 'idmmobil'
            self.control_type = 'idmmobil'
            self.lateral_actions = {'move_left':0.75,
                                    'move_right':-0.75,
                                    'keep_lane':0}
            self.steps_since_lc_initiation = 0
            self.steps_prior_lc = 20 # steps
            self.steps_to_new_lane_entry = self.steps_prior_lc + \
                            (0.5*self.lane_width)/(0.1*self.lateral_actions['move_left'])

            self.steps_to_lc_completion = self.steps_prior_lc + \
                            (self.lane_width)/(0.1*self.lateral_actions['move_left'])

            self.parameter_range = {'most_aggressive': {
                                            'desired_v':25, # m/s
                                            'desired_tgap': 0.5, # s
                                            'min_jamx':1, # m
                                            'max_act':4, # m/s^2
                                            'min_act':4, # m/s^2
                                            'politeness':0.,
                                            'safe_braking':-5,
                                            'act_threshold':0
                                            },
                             'least_aggressvie': {
                                            'desired_v':15, # m/s
                                            'desired_tgap':2, # s
                                            'min_jamx':5, # m
                                            'max_act':2, # m/s^2
                                            'min_act':2, # m/s^2
                                            'politeness':1,
                                            'safe_braking':-3,
                                            'act_threshold':0.2
                                             }}

            self.driver_params = {}
            self.driver_params['aggressiveness'] = aggressiveness  # in range [0, 1]
            if aggressiveness != None:
                self.set_driver_params()

    def set_attentiveness(self):
        self.driver_params['attentiveness'] = \
                            self.steps_to_new_lane_entry*self.sample_driver_param()

    def set_driver_params(self):
        self.set_attentiveness()
        # IDM params
        self.driver_params['desired_v'] = self.get_driver_param('desired_v')
        # self.driver_params['desired_v'] += np.random.normal()
        self.driver_params['desired_tgap'] = self.get_driver_param('desired_tgap')
        self.driver_params['min_jamx'] = self.get_driver_param('min_jamx')
        self.driver_params['max_act'] = self.get_driver_param('max_act')
        self.driver_params['min_act'] = self.get_driver_param('min_act')
        # MOBIL params
        self.driver_params['politeness'] = self.get_driver_param('politeness')
        self.driver_params['safe_braking'] = self.get_driver_param('safe_braking')
        self.driver_params['act_threshold'] = self.get_driver_param('act_threshold')

    def sample_driver_param(self):
        # return self.driver_params['aggressiveness']
        # return np.random.triangular(0, self.driver_params['aggressiveness'], 1)
        alpha_param = self.beta_precision*self.driver_params['aggressiveness']
        beta_param = self.beta_precision*(1-self.driver_params['aggressiveness'])
        return np.random.beta(alpha_param, beta_param)

    def get_driver_param(self, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act']:
            # the larger the param, the more aggressive the driver
            min_value = self.parameter_range['least_aggressvie'][param_name]
            max_value = self.parameter_range['most_aggressive'][param_name]
            return  min_value + self.sample_driver_param()*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness',
                                                'act_threshold', 'safe_braking']:
            # the larger the param, the more timid the driver
            min_value = self.parameter_range['most_aggressive'][param_name]
            max_value = self.parameter_range['least_aggressvie'][param_name]
            return  max_value - self.sample_driver_param()*(max_value-min_value)

    def my_neighbours(self, vehicles):
        """
        Ego can be in 3 states:
        - decided to change lane - indicaiton
        - decided to change lane - in progress
        - decided to keep lane

        Neighbours can be in 3 states:
        - decided to change lane - indicaiton
        - decided to change lane - in progress
        - decided to keep lane

        Who is ego's neighbour depends on ego's and a given neighbour's state.
        """
        neighbours = {}
        delta_xs_f, delta_xs_fl, delta_xs_rl, delta_xs_r, \
        delta_xs_rr, delta_xs_fr, delta_xs_m, delta_xs_att = ([self.perception_range] for i in range(8))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
        candidate_rr, candidate_fr, candidate_m, candidate_att = (None for i in range(8))

        right_lane_id = self.lane_id + 1
        left_lane_id = self.lane_id - 1

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = abs(vehicle.glob_x-self.glob_x)
                vehicle_lane_y = round(vehicle.lane_y, 2)
                if vehicle.lane_id in [self.lane_id, right_lane_id, left_lane_id] and \
                                                    delta_x < self.perception_range:

                    if self.lane_decision != 'keep_lane':
                        # ego performing a lane change
                        if vehicle.lane_id == self.target_lane:
                            if self.target_lane == vehicle.target_lane and \
                                                vehicle.glob_x > self.glob_x and \
                                                delta_x < min(delta_xs_att):

                                if delta_x < min(delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                if delta_x < min(delta_xs_att):
                                    delta_xs_att.append(delta_x)
                                    candidate_att = vehicle

                            if vehicle_lane_y == 0:
                                if vehicle.glob_x < self.glob_x:
                                    if self.lane_decision == 'move_right' and \
                                                            delta_x < min(delta_xs_rr):
                                        delta_xs_rr.append(delta_x)
                                        candidate_rr = vehicle

                                    elif self.lane_decision == 'move_left' and \
                                                            delta_x < min(delta_xs_rl):
                                        delta_xs_rl.append(delta_x)
                                        candidate_rl = vehicle
                                else:
                                    if self.lane_decision == 'move_right' and \
                                                            delta_x < min(delta_xs_fr):
                                        delta_xs_fr.append(delta_x)
                                        candidate_fr = vehicle

                                    elif self.lane_decision == 'move_left' and \
                                                            delta_x < min(delta_xs_fl):
                                        delta_xs_fl.append(delta_x)
                                        candidate_fl = vehicle

                                    if delta_x < min(delta_xs_f):
                                        delta_xs_f.append(delta_x)
                                        candidate_f = vehicle

                                    if delta_x < min(delta_xs_att):
                                        delta_xs_att.append(delta_x)
                                        candidate_att = vehicle

                        if self.lane_id == self.target_lane == vehicle.target_lane and \
                                vehicle.lane_decision != 'keep_lane' and \
                                        round(self.lane_y, 2) == 0 and \
                                        vehicle.glob_x > self.glob_x and \
                                        delta_x < min(delta_xs_m):
                            delta_xs_m.append(delta_x)
                            candidate_m = vehicle
                            if self.am_i_attending(vehicle, delta_x, delta_xs_att):
                                # for merging cars
                                delta_xs_att.append(delta_x)
                                candidate_att = vehicle
                    else:
                        if vehicle.glob_x > self.glob_x:
                            # front neibouring cars
                            if vehicle.target_lane == self.lane_id:
                                if self.am_i_attending(vehicle, delta_x, delta_xs_att):
                                    # for merging cars
                                    delta_xs_att.append(delta_x)
                                    candidate_att = vehicle

                                if vehicle.lane_decision != 'keep_lane':
                                    if delta_x < min(delta_xs_m):
                                        delta_xs_m.append(delta_x)
                                        candidate_m = vehicle
                                else:
                                    if delta_x < min(delta_xs_f):
                                        delta_xs_f.append(delta_x)
                                        candidate_f = vehicle

                                if vehicle.lane_id == self.lane_id:
                                    if delta_x < min(delta_xs_att):
                                        delta_xs_att.append(delta_x)
                                        candidate_att = vehicle

                            if vehicle_lane_y == 0:
                                if vehicle.lane_id == right_lane_id:
                                    if delta_x < min(delta_xs_fr):
                                        # neighbour keeping lane
                                        delta_xs_fr.append(delta_x)
                                        candidate_fr = vehicle

                                elif vehicle.lane_id == left_lane_id:
                                    if delta_x < min(delta_xs_fl):
                                        delta_xs_fl.append(delta_x)
                                        candidate_fl = vehicle

                        else:
                            if vehicle_lane_y == 0:
                                if vehicle.lane_id == right_lane_id:
                                    if delta_x < min(delta_xs_rr):
                                        # neighbour keeping lane
                                        delta_xs_rr.append(delta_x)
                                        candidate_rr = vehicle

                                elif vehicle.lane_id == left_lane_id:
                                    if delta_x < min(delta_xs_rl):
                                        delta_xs_rl.append(delta_x)
                                        candidate_rl = vehicle

                            if vehicle.lane_id == self.lane_id == vehicle.target_lane:
                                # same lane
                                if delta_x < min(delta_xs_r):
                                    delta_xs_r.append(delta_x)
                                    candidate_r = vehicle

        neighbours['f'] = candidate_f
        neighbours['fl'] = candidate_fl
        neighbours['rl'] = candidate_rl
        neighbours['r'] = candidate_r
        neighbours['rr'] = candidate_rr
        neighbours['fr'] = candidate_fr
        if candidate_m and candidate_f and candidate_m.glob_x <= candidate_f.glob_x:
            neighbours['m'] = candidate_m
        elif candidate_m and not candidate_f:
            neighbours['m'] = candidate_m
        else:
            neighbours['m'] = None
        # neighbours['m'] = candidate_m
        neighbours['att'] = candidate_att
        return neighbours

    def update_desired_speed(self, att_vehicle):
        """More aggressive drivers always want to go faster than their leader.
        """
        if att_vehicle:
            agg_diff = self.driver_params['aggressiveness']-\
                                            att_vehicle.driver_params['aggressiveness']

            desired_v_diff = self.driver_params['desired_v']-\
                                            att_vehicle.driver_params['desired_v']
            if agg_diff > 0 and desired_v_diff < 0:
                self.driver_params['desired_v'] = att_vehicle.driver_params['desired_v']*1.1

    def am_i_attending(self, vehicle, delta_x, delta_xs):
        """Am I attending to the vehicle?
            There are x3 scenarios:
            - I am alreading attending to a merger and there is no closer merger
            - I am alreading attending to a merger and there is a new, closer merger
            - There is a new merger
                - happens either due to attentiveness or to avoid dangerous scenarios
        """
        # am I already attending to a merge car?
        if self.neighbours['m']:
            if self.neighbours['m'] == self.neighbours['att'] == vehicle:
                return True
            elif self.neighbours['m'] == self.neighbours['att'] != vehicle:
                if vehicle.glob_x <= self.neighbours['att'].glob_x and delta_x < min(delta_xs):
                    return True
                else:
                    return False

        act_long = self.idm_action(self, vehicle)
        if  delta_x < min(delta_xs) and \
                vehicle.steps_since_lc_initiation >= self.driver_params['attentiveness']:
            return True
        return False

    def will_i_lead(self, vehicle, delta_x, delta_xs):
        """Will I be leading 'vehicle' at some point?
        """
        if vehicle.target_lane == self.target_lane and \
                                    self.glob_x > vehicle.glob_x and \
                                       delta_x < min(delta_xs):
            return True
        return False

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 1000]
        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        assert delta_x > 0, 'leader {l_id} is behind follower {f_id}'.format(\
                                        l_id=leader.id, f_id=follower.id)
        return [delta_v, delta_x]

    def idm_action(self, follower, leader):
        """
        Note: Follower is not always the ego, it can be the ego's neighbour(use:MOBIL)
        """
        if not follower:
            return 0
        delta_v, delta_x = self.observe(follower, leader)
        desired_gap = follower.driver_params['min_jamx'] + \
                        max(0,
                        follower.driver_params['desired_tgap']*\
                        follower.speed+(follower.speed*delta_v)/ \
                        (2*np.sqrt(follower.driver_params['max_act']*\
                        follower.driver_params['min_act'])))

        act_long = follower.driver_params['max_act']*\
                    (1-(follower.speed/follower.driver_params['desired_v'])**4-\
                                                        (desired_gap/(delta_x))**2)

        return act_long
        # return max(-5, min(act_long, 5))

    def check_reservations(self, target_lane, reservations):
        """To ensure:
        - two cars do not simultaneously move into the same lane.
        - ego does not perform a lane change as an other car is merging nearby into
        its lane.
        """
        # print(reservations)
        if not reservations:
            return True
        else:
            for reserved in reservations.values():
                reserved_lane, max_glob_x, min_glob_x = reserved
                if target_lane == reserved_lane and min_glob_x <= self.glob_x <= max_glob_x:
                    return False
            return True

    def check_neighbours(self, neighbours):
        """To ensure neighbours keep lane while merger is changing lane.
        """
        if neighbours['m'] and neighbours['m'].lane_decision != 'keep_lane':
            return False
        return True

    def mobil_condition(self, actions_gains):
        """To decide if changing lane is worthwhile.
        """
        ego_gain, new_follower_gain, old_follower_gain = actions_gains
        lc_condition = ego_gain+self.driver_params['politeness']*(new_follower_gain+\
                                                                old_follower_gain )
        return lc_condition

    def act(self, reservations):
        act_long, act_lat = self.idm_mobil_act(reservations)
        return [act_long, act_lat]

    def lateral_action(self):
        if self.lane_decision == 'keep_lane':
            return 0

        if self.steps_since_lc_initiation >= self.steps_prior_lc:
            self.steps_since_lc_initiation += 1
            return self.lateral_actions[self.lane_decision]
        else:
            self.steps_since_lc_initiation += 1
            return 0

    def is_lc_complete(self):
        if self.steps_since_lc_initiation == self.steps_to_lc_completion:
            if self.lane_decision == 'move_left' and self.neighbours['rl']:
                self.neighbours['rl'].neighbours['f'] = self
                self.neighbours['rl'].neighbours['m'] = None
            elif self.lane_decision == 'move_right' and self.neighbours['rr']:
                self.neighbours['rr'].neighbours['f'] = self
                self.neighbours['rr'].neighbours['m'] = None

            self.lane_decision = 'keep_lane'
            self.steps_since_lc_initiation = 0

    def idm_mobil_act(self, reservations):
        neighbours = self.neighbours
        act_long = self.idm_action(self, self.neighbours['att'])
        # return [act_long, self.lateral_action()]
        if self.lane_decision != 'keep_lane':
            self.is_lc_complete()

        elif self.lane_decision == 'keep_lane' and self.glob_x > 50 and \
                                        self.check_neighbours(self.neighbours):
            lc_left_condition = 0
            lc_right_condition = 0

            act_rl_lc = self.idm_action(self.neighbours['rl'], self)
            act_rr_lc = self.idm_action(self.neighbours['rr'], self)
            act_r_lc = self.idm_action(self.neighbours['r'], self.neighbours['f'])
            act_r_lk = self.idm_action(self.neighbours['r'], self)
            old_follower_gain = act_r_lc-act_r_lk

            if self.lane_id > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_ego_lc_l = self.idm_action(self, self.neighbours['fl'])
                act_rl_lk = self.idm_action(self.neighbours['rl'], self.neighbours['fl'])
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if self.lane_id < self.lanes_n and self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_action(self, self.neighbours['fr'])
                act_rr_lk = self.idm_action(self.neighbours['rr'], self.neighbours['fr'])

                ego_gain = act_ego_lc_r-act_long
                new_follower_gain = act_rr_lc-act_rr_lk
                lc_right_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if max([lc_left_condition, lc_right_condition]) > self.driver_params['act_threshold']:

                if lc_left_condition > lc_right_condition:
                    target_lane = self.target_lane - 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_left'
                        self.neighbours['att'] = self.neighbours['fl']
                        self.neighbours['f'] = self.neighbours['fl']
                        # self.set_driver_params()
                        self.target_lane -= 1
                        if self.neighbours['rl']:
                            self.neighbours['rl'].set_attentiveness()
                        return [act_ego_lc_l, self.lateral_action()]

                elif lc_left_condition < lc_right_condition:
                    target_lane = self.target_lane + 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_right'
                        self.neighbours['att'] = self.neighbours['fr']
                        self.neighbours['f'] = self.neighbours['fr']
                        # self.set_driver_params()
                        self.target_lane += 1
                        if self.neighbours['rr']:
                            self.neighbours['rr'].set_attentiveness()

                        return [act_ego_lc_r, self.lateral_action()]

        return [act_long, self.lateral_action()]
