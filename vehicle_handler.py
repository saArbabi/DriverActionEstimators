from importlib import reload
import idmmobil_vehicle
reload(idmmobil_vehicle)
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
        # aggressiveness = 1
        aggressiveness = np.random.choice([0.7, 1])
        # aggressiveness = np.random.choice([0, 0.5, 1])
        speed = aggressiveness*10 + 20 + np.random.normal(0, 1)
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
            delta_x = leader.glob_x - follower.glob_x
            # coin_flip = np.random.random()
            if delta_x > 60:
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
            max_glob_x, min_glob_x = round(vehicle.glob_x) + 50, round(vehicle.glob_x) - 50
            self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]
