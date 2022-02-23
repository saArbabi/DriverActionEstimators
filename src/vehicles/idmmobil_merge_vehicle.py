from importlib import reload
from vehicles import idmmobil_vehicle
reload(idmmobil_vehicle)
from vehicles.idmmobil_vehicle import IDMMOBILVehicle
import json

class IDMMOBILVehicleMerge(IDMMOBILVehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed, aggressiveness)
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)

        self.act_long_c = None # current action
        self.act_long_p = None # last action
        self.merge_lane_start = config['merge_lane_start']
        self.ramp_exit_start = config['ramp_exit_start']

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
        delta_xs_fr, delta_xs_att = ([self.perception_range] for i in range(6))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
        candidate_fr, candidate_m, candidate_att = (None for i in range(7))

        right_lane_id = self.lane_id + 1
        left_lane_id = self.lane_id - 1

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = abs(vehicle.glob_x-self.glob_x)
                if delta_x < self.perception_range:
                    if self.is_it_merger(vehicle):
                        candidate_m = vehicle

                    if vehicle.glob_x > self.glob_x:
                        # front neibouring cars
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

        if candidate_m and candidate_f and \
                    candidate_m.glob_x > candidate_f.glob_x and \
                    candidate_m.lane_decision != 'keep_lane':
            neighbours['m'] = None
        else:
            neighbours['m'] = candidate_m

        if neighbours['m'] and self.am_i_attending(neighbours['m'], candidate_f):
            neighbours['att'] = neighbours['m']
        else:
            neighbours['att'] = candidate_att
        return neighbours

    def is_it_merger(self, vehicle):
        if vehicle.id == 'dummy':
            return False
        elif self.glob_x > self.ramp_exit_start and vehicle.glob_x < self.glob_x:
            return False
        elif (vehicle.lane_decision != 'keep_lane' and vehicle.glob_x > self.glob_x) \
            or (vehicle.lane_decision == 'keep_lane' and vehicle.lane_id == 2):
            return True
        return False

    def get_ttm(self, m_veh):
        delta_y = self.lane_width*1.5 - m_veh.glob_y# merger lateral distance to main road
        ttm_m_veh = delta_y/self.lateral_actions['move_left'] # steps to merge
        return ttm_m_veh

    def is_cidm_att(self, act_long, m_veh, f_veh):
        ttm_m_veh = self.get_ttm(m_veh)
        gap_to_merge = ttm_m_veh*m_veh.speed + m_veh.glob_x-self.glob_x
        ttm_e_veh = gap_to_merge/self.speed
        # print('ttm_e_veh + polit', self.driver_params['politeness']*ttm_e_veh)
        # print('ttm_e_veh ', ttm_e_veh)
        # print('ttm_m_veh ', ttm_m_veh)
        # print('act_long ', act_long)
        if ttm_m_veh < self.driver_params['politeness']*ttm_e_veh and \
                                act_long > -self.driver_params['min_act']:
            return True

    def am_i_attending(self, m_veh, f_veh):
        """Several scenarios are possible:
        (1) Ego attends because merger has entered its lane
        (2) Ego attends following the cooperative idm
        (3) Ego attends for safety
        """
        if (self.glob_x > m_veh.glob_x) or (f_veh.glob_x < m_veh.glob_x) \
                            or (m_veh.glob_x < self.merge_lane_start) \
                            or m_veh.lane_decision == 'keep_lane':
            return False
        elif m_veh == self.neighbours['att']:
            return True
        elif m_veh.lane_id == self.lane_id:
            # print('lane-based ########')
            return True

        act_long = self.idm_action(self, m_veh)
        if act_long < self.driver_params['safe_braking']:
            # emergency situation
            # print('collisio- avoidance based ########')
            return True
        elif self.is_cidm_att(act_long, m_veh, f_veh):
            # print('cidm-based ########')
            return True
        else:
            return False

    def act(self):
        act_long, act_lat = self.idm_mobil_act()
        return [act_long, act_lat]

    def is_merge_complete(self):
        if self.glob_y >= 1.5*self.lane_width:
            if self.neighbours['rl']:
                self.neighbours['rl'].neighbours['f'] = self
                self.neighbours['rl'].neighbours['m'] = None
            self.lane_decision = 'keep_lane'
            self.glob_y = 1.5*self.lane_width

    def lateral_action(self):
        if self.lane_decision == 'keep_lane':
            return 0

        if self.glob_x >= self.ramp_exit_start:
            if self.target_lane != 1:
                self.target_lane = 1
            return self.lateral_actions[self.lane_decision]
        else:
            return 0

    def is_merge_possible(self, act_rl_lc):
        # return False
        # if self.id == 4:
        #     print(act_rl_lc)
        if self.lane_id > 1 and self.glob_x > self.merge_lane_start and \
                self.driver_params['safe_braking'] <= act_rl_lc:
            return True

    def idm_mobil_act(self):
        act_long = self.idm_action(self, self.neighbours['att'])
        if self.lane_decision != 'keep_lane':
            self.is_merge_complete()

        elif self.lane_decision == 'keep_lane' and self.lane_id == 2:
            lc_left_condition = 0
            act_ego_lc_l = self.idm_action(self, self.neighbours['fl'])
            act_rl_lc = self.idm_action(self.neighbours['rl'], self)
            # print('act_rl_lc ', act_rl_lc)
            if self.is_merge_possible(act_rl_lc):
                # consider moving left
                act_r_lc = self.idm_action(self.neighbours['r'], self.neighbours['f'])
                act_r_lk = self.idm_action(self.neighbours['r'], self)
                old_follower_gain = act_r_lc-act_r_lk

                act_rl_lk = self.idm_action(self.neighbours['rl'], self.neighbours['fl'])
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk

                lc_left_condition = self.mobil_condition([ego_gain, \
                                        new_follower_gain, old_follower_gain])

                # print('ego_gain ', ego_gain)
                # print('old_follower_gain ', old_follower_gain)
                # print('new_follower_gain ', new_follower_gain)
                # print('lc_left_condition ', lc_left_condition)

            if lc_left_condition > self.driver_params['act_threshold']:
                self.lane_decision = 'move_left'

        return [act_long, self.lateral_action()]
