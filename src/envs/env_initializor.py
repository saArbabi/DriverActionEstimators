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
        self.lane_length = config['lane_length']
        self.merge_lane_start = config['merge_lane_start']

        self.min_v = 10
        self.max_v = 25
        self.desired_v_range = self.max_v-self.min_v

    def get_init_speed(self, aggressiveness):
        init_speed = np.random.uniform(self.min_v, \
                            self.min_v+aggressiveness*self.desired_v_range)
        return init_speed

    def create_main_lane_vehicle(self, lead_vehicle, lane_id, glob_x, agg):
        init_speed = self.get_init_speed(agg)
        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, agg)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        while init_action < -new_vehicle.driver_params['min_act']:
            new_vehicle.glob_x -= 10
            init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
            if new_vehicle.glob_x < 0:
                return
        else:
            self.next_vehicle_id += 1
            return new_vehicle

    def create_ramp_merge_vehicle(self, lane_id, glob_x, agg):
        lead_vehicle = self.dummy_stationary_car
        init_speed = self.get_init_speed(agg)
        new_vehicle = IDMMOBILVehicleMerge(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, agg)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        if init_action >= new_vehicle.driver_params['safe_braking']:
            self.next_vehicle_id += 1
            return new_vehicle

    def init_env(self, episode_id):
        print('episode_id: ', episode_id)
        np.random.seed(episode_id)
        # main road vehicles
        lane_id = 1
        vehicles = []
        traffic_density = np.random.randint(3, 6) # number of vehicles
        # print('traffic_density ', traffic_density)

        available_spacing = self.lane_length-self.merge_lane_start-50
        glob_x = available_spacing
        avg_spacing = available_spacing/traffic_density
        aggs = np.random.uniform(0.01, 0.99, traffic_density) # aggressiveness
        vehicle_count = 0
        while len(vehicles) < traffic_density:
            if not vehicles:
                lead_vehicle = None
            else:
                lead_vehicle = vehicles[-1]
            new_vehicle = self.create_main_lane_vehicle(lead_vehicle, \
                                            lane_id, glob_x, aggs[vehicle_count])
            if new_vehicle:
                vehicles.append(new_vehicle)
                glob_x = vehicles[-1].glob_x - avg_spacing
                glob_x = max([0, glob_x])
                vehicle_count += 1
            if glob_x == 0:
                break
        # vehicles[-1].glob_x = 0
        lead_car_speed = np.random.uniform(self.min_v, self.max_v)
        vehicles[0].speed = vehicles[0].driver_params['desired_v'] = lead_car_speed

        # ramp vehicles
        lane_id = 2
        aggs = np.random.uniform(0.2, 0.99) # aggressiveness
        while True:
            glob_x = np.random.uniform(150,  200)
            new_vehicle = self.create_ramp_merge_vehicle(lane_id, \
                                                         glob_x, aggs)
            if new_vehicle:
                vehicles.append(new_vehicle)
                break
        return vehicles
