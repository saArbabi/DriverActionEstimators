# from factory.viewer import Viewer
from importlib import reload
import copy
import vehicle_handler
reload(vehicle_handler)
from vehicle_handler import VehicleHandler

class Env:
    def __init__(self, config):
        self.config = config
        self.time_step = 0
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


    def recorder(self, ego, actions):
        """For recording vehicle trajectories. Used for:
        - model training
        - perfromance validations # TODO
        """
        if self.usage != 'data generation':
            return
        # if ego.glob_x < 100:
        #     return

        if not ego.id in self.recordings:
            self.recordings[ego.id] = {}
        log = {attrname: getattr(ego, attrname) for attrname in self.veh_log}
        log['att_veh_id'] = None if not ego.neighbours['f'] else ego.neighbours['f'].id
        log['aggressiveness'] = ego.driver_params['aggressiveness']
        log['act_long'], log['act_lat'] = actions
        self.recordings[ego.id][self.time_step] = log

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            neighbours = vehicle.my_neighbours(self.vehicles)
            vehicle.neighbours = neighbours
            actions = vehicle.act(neighbours, self.handler.reservations)
            joint_action.append(actions)
            self.handler.update_reservations(vehicle)
            self.recorder(vehicle, actions)

        return joint_action

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        vehicles = []
        joint_action = self.get_joint_action()
        self.time_step += 1

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
