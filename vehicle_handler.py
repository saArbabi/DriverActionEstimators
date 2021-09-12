from importlib import reload
from vehicles import idmmobil_vehicle
reload(idmmobil_vehicle)
from vehicles.idmmobil_vehicle import IDMMOBILVehicle
from vehicles.idm_vehicle import IDMVehicle
import numpy as np


class VehicleHandler:
    def __init__(self, config=None):
        # self.entry_vehicles =
        # self.queuing_vehicles =
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.next_vehicle_id = 1
        self.lane_width = 3.7
        self.reservations = {}

    def create_vehicle(self, lane_id):
        """Creates a new vehicle.
        """
        id = self.next_vehicle_id
        glob_x = np.random.uniform(-30, 0)
        # if lane_id in [1, 2]:
        #     aggressiveness = np.random.uniform(0.7, 0.99)
        # else:
        #     aggressiveness = np.random.uniform(0.01, 0.99)
        # aggressiveness = np.random.uniform(0.2, 0.8)
        aggressiveness = np.random.uniform(0.01, 0.99)

        # aggressiveness = np.random.choice([0, 0.5, 1])
        speed = 30 - (lane_id-1)*3 + np.random.normal()
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

            leader = last_entries[lane_id]
            follower = queuing_entries[lane_id]

            # ttc = delta_x/follower.speed
            # max_delta_x = follower.driver_params['aggressiveness']*50 + \
            #                                 (20 + 10*np.random.normal(0, 1))
            delta_x = leader.glob_x - follower.glob_x
            if delta_x > 80:
                # check if cars are not too close
                new_entries.append(queuing_entries[lane_id])
                last_entries[lane_id] = queuing_entries[lane_id]
                queuing_entries[lane_id] = None

        return new_entries

    def update_reservations(self, vehicle):
        """
        Note: lane reservations have to be updated after each vehicle decides an action.
        """
        if vehicle.id in self.reservations and vehicle.lane_decision == 'keep_lane':
            del self.reservations[vehicle.id]
        elif vehicle.lane_decision != 'keep_lane':
            if vehicle.lane_decision == 'move_left':
                leader_neighbour = vehicle.neighbours['fl']
                follower_neighbour = vehicle.neighbours['rl']
            elif vehicle.lane_decision == 'move_right':
                leader_neighbour = vehicle.neighbours['fr']
                follower_neighbour = vehicle.neighbours['rr']

            if leader_neighbour:
                max_glob_x = round(leader_neighbour.glob_x) + 5
            else:
                max_glob_x = vehicle.glob_x + 30

            if follower_neighbour:
                min_glob_x = round(follower_neighbour.glob_x) - 5
            else:
                min_glob_x = vehicle.glob_x - 30

            self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]

# class VehicleHandlerMC(VehicleHandler):
#     def __init__(self, config=None):
#         super().__init__(config)
#
#     def create_vehicle(self, lane_id):
#         """Creates a new vehicle.
#         """
#         id = self.next_vehicle_id
#         glob_x = np.random.uniform(-30, 0)
#         aggressiveness = np.random.uniform(0, 1)
#
#         speed = 25 + np.random.normal(0, 2)
#         new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)
#         new_vehicle.lanes_n = self.lanes_n
#         new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
#         self.next_vehicle_id += 1
#         return new_vehicle


# class VehicleHandlerLaneKeep:
#     def __init__(self, config=None):
#         # self.entry_vehicles =
#         # self.queuing_vehicles =
#         self.lane_length = config['lane_length']
#         self.next_vehicle_id = 1
#         self.lane_width = 3.7
#         self.reservations = {}
#
#     def create_vehicle(self, lane_id):
#         """Creates a new vehicle.
#         """
#         id = self.next_vehicle_id
#         glob_x = np.random.uniform(-30, 0)
#         aggressiveness = np.random.choice([0, 0.5, 1])
#         speed = 25 + np.random.normal(0, 2)
#         # new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)
#         new_vehicle = IDMVehicle(id, lane_id, glob_x, speed, aggressiveness)
#         new_vehicle.glob_y = self.lane_width/2
#         self.next_vehicle_id += 1
#         return new_vehicle
#
#     def handle_vehicle_entries(self, queuing_entries, last_entries):
#         new_entries = []
#         lane_id = 1
#         if not last_entries:
#             new_vehicle = self.create_vehicle(lane_id)
#             queuing_entries[lane_id] = None
#             last_entries[lane_id] = new_vehicle
#             new_entries.append(new_vehicle)
#             return new_entries
#
#         if not queuing_entries[lane_id]:
#             queuing_entries[lane_id] = self.create_vehicle(lane_id)
#
#         leader = last_entries[lane_id]
#         follower = queuing_entries[lane_id]
#         delta_x = delta_x
#         speed = 25 + np.random.normal(0, 2)
#
#         if follower.driver_params['aggressiveness'] > 0.7:
#             max_delta_x = 130
#         else:
#             max_delta_x = 75
#
#         if delta_x > max_delta_x:
#             new_entries.append(queuing_entries[lane_id])
#             last_entries[lane_id] = queuing_entries[lane_id]
#             queuing_entries[lane_id] = None
#
#         return new_entries
