from importlib import reload
from vehicles import idmmobil_vehicle
reload(idmmobil_vehicle)
from vehicles.idmmobil_vehicle import IDMMOBILVehicle

class IDMMOBILVehicleMerge(IDMMOBILVehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed, aggressiveness)
        self.STM_m_veh = self.steps_prior_lc + \
                        (0.5*self.lane_width)/(0.1*self.lateral_actions['move_left']) # steps to merge
        self.TTM_m_veh = self.STM_m_veh*0.1 # time to merge

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
        delta_xs_fr, delta_xs_m, delta_xs_att = \
                                            ([self.perception_range] for i in range(7))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
        candidate_fr, candidate_m, candidate_att = (None for i in range(7))

        right_lane_id = self.lane_id + 1
        left_lane_id = self.lane_id - 1

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = abs(vehicle.glob_x-self.glob_x)
                if delta_x < self.perception_range:
                    if vehicle.glob_x > self.glob_x:
                        # front neibouring cars
                        if self.is_it_merger(vehicle):
                            if min(delta_xs_f) > delta_x < min(delta_xs_m):
                                delta_xs_m.append(delta_x)
                                candidate_m = vehicle

                        if vehicle.target_lane == self.target_lane and \
                                vehicle.lane_decision == 'keep_lane':
                                if delta_x < min(delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                if delta_x < min(delta_xs_att):
                                    delta_xs_att.append(delta_x)
                                    candidate_att = vehicle

                        if vehicle.target_lane == right_lane_id or \
                            (self.lane_decision != 'keep_lane' and vehicle.lane_id == 2):
                            if delta_x < min(delta_xs_fr):
                                # neighbour keeping lane
                                delta_xs_fr.append(delta_x)
                                candidate_fr = vehicle

                        if vehicle.target_lane == left_lane_id:
                            if delta_x < min(delta_xs_fl):
                                delta_xs_fl.append(delta_x)
                                candidate_fl = vehicle
                    else:
                        if vehicle.target_lane == left_lane_id:
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
        neighbours['fr'] = candidate_fr
        neighbours['r'] = candidate_r
        if not candidate_m and candidate_fr and candidate_fr.lane_decision == 'keep_lane' \
                                                and candidate_fr.glob_x-self.glob_x > 10 \
                                                        and candidate_fr.id != 'dummy':

            neighbours['m'] = candidate_fr
        elif (candidate_m and not candidate_f) or \
                (candidate_m and candidate_f and candidate_m.glob_x < candidate_f.glob_x):
            neighbours['m'] = candidate_m
        else:
            neighbours['m'] = None


        if self.am_i_attending(neighbours['m'], candidate_f):
            neighbours['att'] = neighbours['m']
        else:
            neighbours['att'] = candidate_att
        # if neighbours['m'] and self.id == 7:
        #     print(self)
        return neighbours

    def is_it_merger(self, vehicle):
        """Several scenarios are possible:
        (1) two vehicles merging simultaneously
        (2) one vehicle mering infront of another keeping lane
        """
        if vehicle.id == 'dummy':
            return False
        elif vehicle.target_lane == self.lane_id and \
                                    vehicle.lane_decision != 'keep_lane':
            return True
        return False

    def am_i_attending(self, m_veh, f_veh):
        if not m_veh or (f_veh and m_veh.glob_x > f_veh.glob_x) or m_veh.glob_x < 150:
            return False
        if m_veh == self.neighbours['att'] == m_veh or \
                                                    m_veh.lane_id == self.lane_id:
            return True

        gap_to_merge = self.TTM_m_veh*m_veh.speed + m_veh.glob_x-self.glob_x
        TTM_e_veh = gap_to_merge/self.speed
        act_long = self.idm_action(self, m_veh)
        # print('TTM_e_veh + polit', self.driver_params['politeness']*TTM_e_veh)
        # print('TTM_m_veh ', self.TTM_m_veh)
        # print('TTM_e_veh ', TTM_e_veh)
        # print('act_long ', act_long)
        if m_veh.lane_decision !='keep_lane' and \
                                act_long < self.driver_params['safe_braking']:
            # emergency situation
            return True
        elif self.TTM_m_veh < self.driver_params['politeness']*TTM_e_veh and \
                                act_long > -self.driver_params['min_act']:
            # print('hi')
            return True
        elif self.TTM_m_veh >= self.driver_params['politeness']*TTM_e_veh:
            return False

    def act(self):
        act_long, act_lat = self.idm_mobil_act()
        return [act_long, act_lat]

    def can_lc_be_considered(self, act_rl_lc):
        # return False
        if self.lane_id > 1 and self.glob_x > 150 and \
                self.driver_params['safe_braking'] <= act_rl_lc:
            return True

    def idm_mobil_act(self):
        act_long = self.idm_action(self, self.neighbours['att'])
        if self.lane_decision != 'keep_lane':
            self.is_lc_complete()

        elif self.lane_decision == 'keep_lane':
            lc_left_condition = 0
            lc_right_condition = 0
            act_ego_lc_l = self.idm_action(self, self.neighbours['fl'])
            act_rl_lc = self.idm_action(self.neighbours['rl'], self)

            if self.can_lc_be_considered(act_rl_lc):
                # consider moving left
                act_r_lc = self.idm_action(self.neighbours['r'], self.neighbours['f'])
                act_r_lk = self.idm_action(self.neighbours['r'], self)
                old_follower_gain = act_r_lc-act_r_lk

                act_rl_lk = self.idm_action(self.neighbours['rl'], self.neighbours['fl'])
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk

                lc_left_condition = self.mobil_condition([ego_gain, \
                                        new_follower_gain, old_follower_gain])

                # if self.id == 7:
                #     print('ego_gain ', ego_gain)
                #     print('old_follower_gain ', old_follower_gain)
                #     print('new_follower_gain ', new_follower_gain)
                #     print('lc_left_condition ', lc_left_condition)
            if lc_left_condition > self.driver_params['act_threshold']:
                target_lane = self.target_lane - 1
                self.lane_decision = 'move_left'
                self.neighbours['att'] = self.neighbours['fl']
                self.neighbours['f'] = self.neighbours['fl']
                # self.set_driver_params()
                self.target_lane -= 1
                # if self.neighbours['rl']:
                    # self.neighbours['rl'].neighbours['att'] = self
                return [act_ego_lc_l, self.lateral_action()]

        return [act_long, self.lateral_action()]
