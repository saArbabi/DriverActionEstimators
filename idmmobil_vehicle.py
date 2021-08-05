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

class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        # self.capability = 'IDM'
        self.lane_id = lane_id
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.neighbours = {veh_name: None for veh_name in ['f', 'fl', 'rl', 'r', 'rr', 'fr', 'm', 'att']}
        self.perception_range = 200 #m
        self.lane_width = 3.75
        self.act_long = 0
        self.steps_since_indicators_on = 0
        self.steps_since_arrival = 0
        self.lateral_actions = {'move_left':0.75,
                                'move_right':-0.75,
                                'keep_lane':0
                                }

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
        self.driver_params['desired_v'] = self.get_idm_param(\
                                Parameter_range, 'desired_v')
        self.driver_params['desired_tgap'] = self.get_idm_param(Parameter_range, 'desired_tgap')
        self.driver_params['min_jamx'] = self.get_idm_param(Parameter_range, 'min_jamx')
        self.driver_params['max_act'] = self.get_idm_param(Parameter_range, 'max_act')
        self.driver_params['min_act'] = self.get_idm_param(Parameter_range, 'min_act')
        # MOBIL params
        self.driver_params['politeness'] = self.get_idm_param(Parameter_range, 'politeness')
        self.driver_params['safe_braking'] = self.get_idm_param(Parameter_range, 'safe_braking')
        self.driver_params['act_threshold'] = self.get_idm_param(Parameter_range, 'act_threshold')

        steps_to_complete_lc = 10 + (0.5*self.lane_width)/(0.1*self.lateral_actions['move_left'])
        if 0 <= self.driver_params['aggressiveness'] < 0.33:
            # timid driver
            attentiveness = steps_to_complete_lc*np.random.beta(2, 10)
        elif 0.33 <= self.driver_params['aggressiveness'] <= 0.66:
            # normal driver
            attentiveness = steps_to_complete_lc*np.random.beta(3, 3)
        elif 0.66 < self.driver_params['aggressiveness']:
            # aggressive driver
            attentiveness = steps_to_complete_lc*np.random.beta(10, 2)

        self.driver_params['attentiveness'] = attentiveness
        # self.driver_params['attentiveness'] = aggressiveness*0.5*self.lane_width

    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act']:
            # the larger the param, the more aggressive the driver
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.driver_params['aggressiveness']*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness', 'act_threshold', 'safe_braking']:
            # the larger the param, the more timid the driver
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  max_value - self.driver_params['aggressiveness']*(max_value-min_value)

    def my_neighbours(self, vehicles):
        """Returns list of current neighbouring vehicles.
            Note:
            - When performing a merge, ego will follow the vehicle in the
            target lane if one exists.
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
                delta_x = vehicle.glob_x-self.glob_x
                if vehicle.lane_id in [self.lane_id, right_lane_id, left_lane_id] and \
                                  abs(delta_x) < self.perception_range:

                    if self.lane_decision != 'keep_lane':
                        if self.am_i_following(vehicle, delta_x, delta_xs_att):
                            delta_xs_att.append(delta_x)
                            candidate_att = vehicle

                            delta_xs_f.append(delta_x)
                            candidate_f = vehicle

                        elif self.will_i_lead(vehicle, delta_x, delta_xs_r):
                            delta_xs_r.append(abs(delta_x))
                            candidate_r = vehicle

                    elif self.lane_decision == 'keep_lane':
                        if delta_x > 0:
                            if vehicle.lane_id == right_lane_id:
                                if vehicle.target_lane == self.lane_id:
                                    if delta_x < delta_xs_m[-1]:
                                        delta_xs_m.append(delta_x)
                                        candidate_m = vehicle

                                    if self.am_i_attending(vehicle, delta_x, delta_xs_att):
                                        # neighbour merging
                                        delta_xs_att.append(delta_x)
                                        candidate_att = vehicle

                                elif delta_x < delta_xs_fr[-1] and \
                                        (vehicle.lane_decision == 'keep_lane' or \
                                         vehicle.target_lane == self.lane_id):
                                    # neighbour keeping lane
                                    delta_xs_fr.append(delta_x)
                                    candidate_fr = vehicle

                            if vehicle.target_lane == right_lane_id:
                                    # neighbour leavering lane
                                if delta_x < delta_xs_fr[-1]:
                                    delta_xs_fr.append(delta_x)
                                    candidate_fr = vehicle

                            if vehicle.lane_id == left_lane_id:
                                if vehicle.target_lane == self.lane_id:
                                    if delta_x < delta_xs_m[-1]:
                                        delta_xs_m.append(delta_x)
                                        candidate_m = vehicle

                                    if self.am_i_attending(vehicle, delta_x, delta_xs_att):
                                        # neighbour merging
                                        delta_xs_att.append(delta_x)
                                        candidate_att = vehicle

                                elif delta_x < delta_xs_fl[-1] and \
                                            (vehicle.lane_decision == 'keep_lane' or \
                                             vehicle.target_lane == self.lane_id):
                                    delta_xs_fl.append(delta_x)
                                    candidate_fl = vehicle

                            if vehicle.target_lane == left_lane_id:
                                if delta_x < delta_xs_fl[-1]:
                                    delta_xs_fl.append(delta_x)
                                    candidate_fl = vehicle

                            if vehicle.lane_id == self.lane_id and \
                                        vehicle.target_lane == self.lane_id:
                                if vehicle.lane_decision != 'keep_lane':
                                    if delta_x < delta_xs_m[-1]:
                                        delta_xs_m.append(delta_x)
                                        candidate_m = vehicle
                                else:
                                    if delta_x < delta_xs_f[-1]:
                                        delta_xs_f.append(delta_x)
                                        candidate_f = vehicle

                                if delta_x < delta_xs_att[-1]:
                                    delta_xs_att.append(delta_x)
                                    candidate_att = vehicle

                        elif delta_x < 0:
                            delta_x = abs(delta_x)
                            if vehicle.lane_id == right_lane_id:
                                if delta_x < delta_xs_rr[-1] and \
                                            vehicle.lane_decision == 'keep_lane':
                                    # neighbour keeping lane
                                    delta_xs_rr.append(delta_x)
                                    candidate_rr = vehicle

                            if vehicle.target_lane == right_lane_id:
                                    # neighbour leavering lane
                                if delta_x < delta_xs_rr[-1]:
                                    delta_xs_rr.append(delta_x)
                                    candidate_rr = vehicle

                            if vehicle.lane_id == left_lane_id:
                                if delta_x < delta_xs_rl[-1]  and \
                                            vehicle.lane_decision == 'keep_lane':
                                    delta_xs_rl.append(delta_x)
                                    candidate_rl = vehicle

                            if vehicle.target_lane == left_lane_id:
                                if delta_x < delta_xs_rl[-1]:
                                    delta_xs_rl.append(delta_x)
                                    candidate_rl = vehicle

                            if vehicle.lane_id == self.lane_id and \
                                    vehicle.target_lane == self.lane_id:
                                # same lane
                                if delta_x < delta_xs_r[-1]:
                                    delta_xs_r.append(delta_x)
                                    candidate_r = vehicle

        neighbours['f'] = candidate_f
        neighbours['fl'] = candidate_fl
        neighbours['rl'] = candidate_rl
        neighbours['r'] = candidate_r
        neighbours['rr'] = candidate_rr
        neighbours['fr'] = candidate_fr

        if candidate_m and candidate_f and candidate_m.glob_x < candidate_f.glob_x:
            neighbours['m'] = candidate_m
        elif candidate_m and not candidate_f:
            neighbours['m'] = candidate_m
        else:
            neighbours['m'] = None

        neighbours['att'] = candidate_att

        return neighbours

    def am_i_attending(self, vehicle, delta_x, delta_xs):
        """Am I attending to the vehicle?
        """
        if vehicle.steps_since_indicators_on >= self.driver_params['attentiveness'] \
                                            and delta_x < delta_xs[-1]:
            return True
        return False

    def am_i_following(self, vehicle, delta_x, delta_xs):
        """Am I following 'vehicle' in my target lane?
        """
        if vehicle.target_lane == self.target_lane and  delta_x > 0 \
                and delta_x < delta_xs[-1]:
            return True
        return False

    def will_i_lead(self, vehicle, delta_x, delta_xs):
        """Will I be leading 'vehicle' at some point?
        """
        if vehicle.target_lane == self.target_lane and \
                                        delta_x < 0 and \
                                        abs(delta_x) < delta_xs[-1]:
            return True
        return False

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
        return [delta_v, delta_x]

    def idm_action(self, obs):
        delta_v, delta_x = obs
        assert delta_x > 0

        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.driver_params['max_act']*(1-(self.speed/self.driver_params['desired_v'])**4-\
                                            (desired_gap/(delta_x))**2)

        return act_long

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
                if (target_lane == reserved_lane or self.lane_id == reserved_lane)\
                                    and min_glob_x < self.glob_x < max_glob_x:

                    return False
            return True

    def check_neighbours(self, neighbours):
        """To ensure neighbours keep lane while merger is changing lane.
        """
        if neighbours['att'] and neighbours['att'].lane_decision != 'keep_lane':
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
        return [max(-3, min(act_long, 3)), act_lat]

    def lateral_action(self):
        if self.lane_decision == 'keep_lane':
            return 0

        if self.lane_id == self.target_lane and round(self.lane_y, 1) == 0:
            self.steps_since_arrival += 1
            return 0

        if self.steps_since_indicators_on >= 10:
            return self.lateral_actions[self.lane_decision]
        else:
            self.steps_since_indicators_on += 1
            return 0

    def is_lane_change_complete(self):
        if self.steps_since_arrival >= 30:
            # manoeuvre completed
            self.lane_decision = 'keep_lane'
            self.lane_y = 0
            self.steps_since_indicators_on = 0
            self.steps_since_arrival = 0

    def idm_mobil_act(self, reservations):
        neighbours = self.neighbours
        act_long = self.idm_action(self.observe(self, neighbours['att']))
        # return [act_long, self.lateral_action()]
        if self.lane_decision != 'keep_lane':
            self.is_lane_change_complete()

        elif self.lane_decision == 'keep_lane' and self.glob_x > 50 and \
                                            self.check_neighbours(neighbours):
            lc_left_condition = 0
            lc_right_condition = 0

            act_rl_lc = self.idm_action(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_action(self.observe(neighbours['rr'], self))
            act_r_lc = self.idm_action(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_action(self.observe(neighbours['r'], self))
            old_follower_gain = act_r_lc-act_r_lk

            if self.lane_id > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_action(self.observe(neighbours['rl'], neighbours['fl']))
                act_ego_lc_l = self.idm_action(self.observe(self, neighbours['fl']))
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if self.lane_id < self.lanes_n and \
                                                self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_action(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_action(self.observe(neighbours['rr'], neighbours['fr']))

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
                        self.target_lane -= 1
                        return [act_ego_lc_l, self.lateral_action()]

                elif lc_left_condition < lc_right_condition:
                    target_lane = self.target_lane + 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_right'
                        self.neighbours['att'] = self.neighbours['fr']
                        self.neighbours['f'] = self.neighbours['fr']
                        self.target_lane += 1
                        return [act_ego_lc_r, self.lateral_action()]

        return [act_long, self.lateral_action()]
