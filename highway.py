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
        self.queuing_entries = {}
        self.last_entries = {}

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

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            neighbours = self.handler.my_neighbours(vehicle, self.vehicles)
            actions = vehicle.act(neighbours, self.handler.reservations)
            vehicle.neighbours = neighbours
            joint_action.append(actions)
            self.handler.update_reservations(vehicle)
            if self.usage == 'data generation':
                self.recorder(vehicle, neighbours)
        return joint_action

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        vehicles = []
        joint_action = self.get_joint_action()
        for vehicle, actions in zip(self.vehicles, joint_action):
            if vehicle.glob_x > self.lane_length:
                # vehicle has left the highway
                if vehicle.id in self.handler.reservations:
                    del self.handler.reservations[vehicle.id]
                continue

            vehicle.step(actions)
            vehicles.append(vehicle)

        self.vehicles = vehicles
        new_entries = self.handler.handle_vehicle_entries(
                                                          self.queuing_entries,
                                                          self.last_entries)
        if new_entries:
            self.vehicles.extend(new_entries)
        self.elapsed_time += 1
