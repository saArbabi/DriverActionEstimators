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
        self.lane_width = config['lane_width']
        self.min_desired_v = 15
        self.max_desired_v = 25
        self.desired_v_range = self.max_desired_v-self.min_desired_v

    def get_init_speed(self, aggressiveness):
        init_speed = self.min_desired_v + \
                    aggressiveness*(self.desired_v_range) + \
                    np.random.normal(0, 1)
        return init_speed

    def create_main_lane_vehicle(self, lead_vehicle, lane_id, glob_x):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = self.get_init_speed(aggressiveness)
        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, aggressiveness)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        if init_action >= -new_vehicle.driver_params['min_act']:
            self.next_vehicle_id += 1
            return new_vehicle

    def create_ramp_merge_vehicle(self, lane_id, glob_x):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = self.get_init_speed(aggressiveness)
        lead_vehicle = self.dummy_stationary_car

        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, aggressiveness)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        if init_action >= -new_vehicle.driver_params['min_act']:
            self.next_vehicle_id += 1
            return new_vehicle


    def init_env(self, episode_id):
        """Operations for creating a scenario
        (1) Create a traffic leader with random speed.
        (2) Create series of followers with similar speeds. The follower positions
            are set to comply with a random initial action value.
        """
        # time.sleep(1)
        np.random.seed(episode_id)
        # main road vehicles
        lane_id = 1
        vehicles = []
        # traffic_density = 30
        traffic_density = np.random.randint(3, 6) # number of vehicles
        # print('traffic_density ', traffic_density)

        glob_x = 150
        while True and len(vehicles) < traffic_density:
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
        while True:
            glob_x = np.random.uniform(50,  150)
            new_vehicle = self.create_ramp_merge_vehicle(lane_id=lane_id, glob_x=glob_x)
            if new_vehicle:
                vehicles.append(new_vehicle)
                break
        return vehicles
