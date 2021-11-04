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
        self.init_speed = 15

    def create_main_lane_vehicle(self, lead_vehicle, lane_id):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = np.random.normal(self.init_speed, 1)
        spacing_bound = np.random.uniform(30, 100) # sets the traffic density
        if not lead_vehicle:
            init_speed = np.random.uniform(5, 20)
            init_x = (2/3)*self.lane_length
            new_vehicle = IDMMOBILVehicleMerge(\
                        self.next_vehicle_id, lane_id, init_x, init_speed, aggressiveness)
            self.next_vehicle_id += 1
            new_vehicle.driver_params['desired_v'] = init_speed
            return new_vehicle
        else:
            new_vehicle = IDMMOBILVehicleMerge(\
                        self.next_vehicle_id, lane_id, 0,\
                                                    init_speed, aggressiveness)
            min_glob_x = max([0, lead_vehicle.glob_x-spacing_bound])
            init_action = -new_vehicle.driver_params['min_act']
            while init_action <= -new_vehicle.driver_params['min_act'] \
                                                    and min_glob_x >= 0:
                new_vehicle.glob_x = np.random.uniform(min_glob_x,\
                                                        lead_vehicle.glob_x)
                min_glob_x -= 20
                init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
                if init_action >= -new_vehicle.driver_params['min_act']:
                    self.next_vehicle_id += 1
                    return new_vehicle

    def create_ramp_merge_vehicle(self, lead_vehicle, lane_id):
        aggressiveness = np.random.uniform(0.01, 0.99)
        init_speed = np.random.normal(self.init_speed, 1)
        spacing_bound = np.random.uniform(50, 150) # sets the traffic density

        if not lead_vehicle:
            lead_vehicle = self.dummy_stationary_car

        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, 0,\
                                                init_speed, aggressiveness)
        min_glob_x = max([0, lead_vehicle.glob_x-spacing_bound])
        init_action = -new_vehicle.driver_params['min_act']

        while init_action <= -new_vehicle.driver_params['min_act']\
                                    and min_glob_x >= 0:
            new_vehicle.glob_x = np.random.uniform(min_glob_x,\
                                                    lead_vehicle.glob_x)
            init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
            min_glob_x -= 20
            if init_action >= -new_vehicle.driver_params['min_act']:
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
        new_vehicle = self.create_main_lane_vehicle(None, lane_id)
        vehicles.append(new_vehicle)

        while new_vehicle:
            new_vehicle = self.create_main_lane_vehicle(vehicles[-1], lane_id)
            if new_vehicle:
                vehicles.append(new_vehicle)

        # ramp vehicles
        lane_id = 2
        new_vehicle = self.create_ramp_merge_vehicle(None, lane_id)
        if new_vehicle:
            vehicles.append(new_vehicle)
        #
        # while new_vehicle:
        #     new_vehicle = self.create_ramp_merge_vehicle(vehicles[-1], lane_id, spacing_bound)
        #     if new_vehicle:
        #         vehicles.append(new_vehicle)
        return vehicles
