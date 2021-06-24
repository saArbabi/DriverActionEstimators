from idmmobil_vehicle import IDMMOBILVehicle
import numpy as np
np.random.seed(2020)

class VehicleHandler:
    def __init__(self, config=None):
        # self.entry_vehicles =
        # self.queuing_vehicles =
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.next_vehicle_id = 0
        self.lane_width = 3.7
        self.reservations = {}

    def create_vehicle(self, lane_id):
        """Creates a new vehicle.
        """
        id = self.next_vehicle_id
        glob_x = np.random.uniform(-30, 0)
        # aggressiveness = np.random.uniform(0, 1)
        aggressiveness = np.random.choice([0, 0.5, 1])
        speed = 20
        new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)
        new_vehicle.lanes_n = self.lanes_n
        new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
        self.next_vehicle_id += 1
        return new_vehicle

    def handle_vehicle_entries(self, queuing_entries, last_entries):
        new_entries = []
        if not last_entries:
            for lane_id in range(1, self.lanes_n+1):
                new_vehicle = self.create_vehicle(lane_id)
                queuing_entries[lane_id] = None
                last_entries[lane_id] = new_vehicle
                new_entries.append(new_vehicle)
            return new_entries

        for lane_id in range(1, self.lanes_n+1):
            # print(last_entries[lane_id].glob_x)
            if not queuing_entries[lane_id]:
                queuing_entries[lane_id] = self.create_vehicle(lane_id)
            else:
                leader = last_entries[lane_id]
                follower = queuing_entries[lane_id]
                act_long = follower.idm_actions(follower.observe(follower, leader))
                if act_long > -.5:
                    # check if cars are not too close
                    new_entries.append(queuing_entries[lane_id])
                    last_entries[lane_id] = queuing_entries[lane_id]
                    queuing_entries[lane_id] = None
        return new_entries

    def my_neighbours(self, ego, vehicles):
        """Returns list of current neighbouring vehicles.
            Note:
            - When performing a merge, ego will follow the vehicle in the
            target lane if one exists.
        """
        neighbours = {}
        delta_xs_f, delta_xs_fl, delta_xs_rl, delta_xs_r, \
                        delta_xs_rr, delta_xs_fr = ([ego.perception_range] for i in range(6))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
                        candidate_rr, candidate_fr = (None for i in range(6))

        for vehicle in vehicles:
            if vehicle.id != ego.id:
                delta_x = vehicle.glob_x-ego.glob_x
                if vehicle.lane_id in [ego.lane_id, ego.lane_id+1, ego.lane_id-1] and \
                                  abs(delta_x) < ego.perception_range:

                    if ego.lane_decision != 'keep_lane':
                        if self.am_i_following(ego.target_lane, vehicle.lane_id, delta_x, delta_xs_f):
                            delta_xs_f.append(delta_x)
                            candidate_f = vehicle

                        elif self.will_i_lead(ego.target_lane, vehicle.lane_id, delta_x, delta_xs_r):
                            delta_xs_r.append(abs(delta_x))
                            candidate_r = vehicle
                        continue

                    elif ego.lane_decision == 'keep_lane':
                        if delta_x >= 0:
                            if vehicle.lane_id == ego.lane_id+1:
                                # right lane
                                if self.am_i_attending(ego, vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fr[-1]:
                                    delta_xs_fr.append(delta_x)
                                    candidate_fr = vehicle

                            elif vehicle.lane_id == ego.lane_id-1:
                                # left lane
                                if self.am_i_attending(ego, vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fl[-1]:
                                    delta_xs_fl.append(delta_x)
                                    candidate_fl = vehicle

                            elif vehicle.lane_id == ego.lane_id:
                                # same lane
                                if delta_x < delta_xs_f[-1]:
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                        elif delta_x < 0:
                            delta_x = abs(delta_x)
                            if vehicle.lane_id == ego.lane_id+1:
                                # right lane
                                if delta_x < delta_xs_rr[-1]:
                                    delta_xs_rr.append(delta_x)
                                    candidate_rr = vehicle
                            elif vehicle.lane_id == ego.lane_id-1:
                                # left lane
                                if delta_x < delta_xs_rl[-1]:
                                    delta_xs_rl.append(delta_x)
                                    candidate_rl = vehicle
                            elif vehicle.lane_id == ego.lane_id:
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

        return neighbours

    def am_i_attending(self, ego, vehicle, delta_x, delta_xs):
        """Am I attending to the merging car?
        """
        if vehicle.target_lane == ego.lane_id and \
                abs(vehicle.lane_y) > ego.driver_params['attentiveness'] \
                and delta_x < delta_xs[-1]:
            return True
        return False

    def am_i_following(self, ego_target_lane, vehicle_lane_id, delta_x, delta_xs):
        """Am I following 'vehicle' in my target lane?
        """
        if vehicle_lane_id == ego_target_lane and delta_x > 0 and delta_x < delta_xs[-1]:
            return True
        return False

    def will_i_lead(self, ego_target_lane, vehicle_lane_id, delta_x, delta_xs):
        """Will I be leading 'vehicle' at some point?
        """
        if vehicle_lane_id == ego_target_lane and delta_x < 0 \
                                            and abs(delta_x) < delta_xs[-1]:
            return True
        return False

    def update_reservations(self, vehicle):
        """
        Note: lane reservations have to be updated after each vehicle decides an action.
        """
        if vehicle.id in self.reservations and vehicle.lane_decision == 'keep_lane':
            del self.reservations[vehicle.id]
        elif vehicle.lane_decision != 'keep_lane':
            max_glob_x, min_glob_x = round(vehicle.glob_x) + 100, round(vehicle.glob_x) - 100
            self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]
