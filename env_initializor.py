from vehicles.idmmobil_vehicle import IDMMOBILVehicleMerge
import numpy as np

class EnvInitializor():
    def __init__(self, config):
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.next_vehicle_id = 1
        self.lane_width = 3.7

    def create_vehicle(self, lead_vehicle, position_range, lane_id):
        """
        Returns a vehicle with random temprements and initial
        states (position+velocity).
        """
        aggressiveness = np.random.uniform(0.01, 0.99)
        speed = 24 + np.random.normal(0, 1)

        max_glob_x = position_range[1]
        min_glob_x = position_range[0]
        init_action = -1.
        while init_action <= -1. and max_glob_x > min_glob_x:
            glob_x = np.random.uniform(min_glob_x, max_glob_x)
            new_vehicle = IDMMOBILVehicleMerge(\
                            self.next_vehicle_id, lane_id, glob_x, speed, aggressiveness)

            init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
            max_glob_x -= 10

        new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
        self.next_vehicle_id += 1
        return new_vehicle

    def init_env(self, episode_id):
        np.random.seed(episode_id)
        # main road vehicles
        lane_id = 1
        vehicles = []
        pos_ranges = np.arange(700,-1, -100) 
        for i in range(len(pos_ranges)-1):
            if not vehicles:
                lead_vehicle = None
            else:
                lead_vehicle = vehicles[-1]

            position_range = [pos_ranges[i+1], pos_ranges[i]]
            new_vehicle = self.create_vehicle(lead_vehicle, position_range, lane_id)
            vehicles.append(new_vehicle)

        # ramp vehicle
        lane_id = 2
        position_range = [100, 200]
        new_vehicle = self.create_vehicle(None, position_range, lane_id)
        vehicles.append(new_vehicle)
        return vehicles
