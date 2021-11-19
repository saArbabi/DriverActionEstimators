import numpy as np
from importlib import reload
from vehicles import idmmobil_merge_vehicle
reload(idmmobil_merge_vehicle)
from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge
# import time

class EnvInitializor():
    def __init__(self, config):
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.lane_width = 3.7

    def create_main_lane_vehicle(self, lead_vehicle, lane_id, glob_x):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = 15 + aggressiveness*(10) + np.random.uniform(0, 1)
        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, aggressiveness)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        if init_action >= -5:
            self.next_vehicle_id += 1
            return new_vehicle

    def create_ramp_merge_vehicle(self, lead_vehicle, lane_id):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = 15 + aggressiveness*(10)
        if not lead_vehicle:
            lead_vehicle = self.dummy_stationary_car

        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, 100,\
                                                init_speed, aggressiveness)
        self.next_vehicle_id += 1
        return new_vehicle

    def init_env(self, episode_id):
        """Operations for creating a scenario
        (1) Create a traffic leader with random speed.
        (2) Create series of followers with similar speeds. The follower positions
            are set to comply with a random initial action value.
        """
        print(episode_id)

        # time.sleep(1)
        np.random.seed(episode_id)
        # main road vehicles
        lane_id = 1
        vehicles = []
        # traffic_density = 30
        traffic_density = np.random.randint(3, 6)
        # print('traffic_density ', traffic_density)

        glob_x = 150
        for n in range(traffic_density):
            if not vehicles:
                lead_vehicle = None
            else:
                lead_vehicle = vehicles[-1]
            new_vehicle = self.create_main_lane_vehicle(lead_vehicle, \
                                                            lane_id, glob_x)
            if new_vehicle:
                vehicles.append(new_vehicle)
            if glob_x == 0:
                break
            glob_x -= np.random.uniform(20, 50)
            glob_x = max([0, glob_x])

        # ramp vehicles
        lane_id = 2
        new_vehicle = self.create_ramp_merge_vehicle(None, lane_id)
        if new_vehicle:
            vehicles.append(new_vehicle)
        return vehicles
