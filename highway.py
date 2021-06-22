# from factory.viewer import Viewer
import copy
import matplotlib.pyplot as plt
from vehicle_handler import VehicleHandler
# import matplotlib.pyplot as plt

class Env:
    def __init__(self, config):
        self.config = config
        self.elapsed_time = 0
        self.handler = VehicleHandler(config)
        self.sdv = None
        self.vehicles = []
        self.usage = None
        self.initiate_environment()
        # self.vehicles = []

    def initiate_environment(self):
        self.lane_length = self.config['lane_length']

    def recorder(self, ego, neighbours):
        """For recording vehicle trajectories. Used for:
        - model training
        - perfromance validations # TODO
        """
        if ego.glob_x < 100:
            return
        if ego.lane_decision == 'keep_lane':
            lane_decision = 0
        elif ego.lane_decision == 'move_left':
            lane_decision = 1
        elif ego.lane_decision == 'move_right':
            lane_decision = -1

        act_long, act_lat = ego.actions
        state = neighbours.copy()
        state['ego'] = [ego.speed, ego.glob_x, act_long, act_lat, ego.lane_y]
        for key, neighbour in neighbours.items():
            if neighbour:
                act_long, _ = neighbour.actions
                state[key] = [neighbour.speed, neighbour.glob_x, act_long]
            else:
                state[key] = None


        if not ego.id in self.recordings['info']:
            self.recordings['info'][ego.id] = ego.driver_params

        if not ego.id in self.recordings['states']:
            self.recordings['states'][ego.id] = []
            self.recordings['decisions'][ego.id] = []
            self.recordings['elapsed_time'][ego.id] = []

        self.recordings['states'][ego.id].append(state)
        self.recordings['decisions'][ego.id].append(lane_decision)
        self.recordings['elapsed_time'][ego.id].append(self.elapsed_time)

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        vehicles = [] # list for vehicles with new states
        for vehicle_i in self.vehicles:
            if vehicle_i.glob_x > self.lane_length:
                # vehicle has left the highway
                if vehicle_i.id in self.handler.reservations:
                    del self.handler.reservations[vehicle_i.id]
                continue
            neighbours = self.handler.my_neighbours(vehicle_i, self.vehicles)
            actions = vehicle_i.act(neighbours, self.handler.reservations)
            vehicle_i.neighbours = neighbours
            vehicle_i.actions = actions

            self.handler.update_reservations(vehicle_i)
            vehicle_ii = copy.copy(vehicle_i)
            vehicle_ii.step(actions)
            vehicles.append(vehicle_ii)

            if self.usage == 'data generation':
                self.recorder(vehicle_i, neighbours)


        self.vehicles = vehicles
        new_vehicle_entries = self.handler.place_vehicle(len(vehicles), self.elapsed_time)
        if new_vehicle_entries:
            self.vehicles.extend(new_vehicle_entries)

        self.elapsed_time += 1
