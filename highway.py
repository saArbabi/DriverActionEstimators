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


    def recorder(self, ego):
        """For recording vehicle trajectories. Used for:
        - model training
        - perfromance validations # TODO
        """

        if self.usage != 'data generation':
            return
        # if ego.glob_x < 100:
        #     return
        if ego.lane_decision == 'keep_lane':
            lane_decision = 0
        elif ego.lane_decision == 'move_left':
            lane_decision = 1
        elif ego.lane_decision == 'move_right':
            lane_decision = -1

        if not ego.id in self.recordings:
            self.recordings[ego.id] = {}
        # self.recordings[ego.id][self.time_step] = copy.deepcopy(ego) # snapshot of ego
        state = {attrname: getattr(ego, attrname) for attrname in self.fetch_states}
        state['att_veh_id'] = None if not ego.neighbours['f'] else ego.neighbours['f'].id
        self.recordings[ego.id][self.time_step] = state
        # copy.deepcopy(ego) # snapshot of ego

        # act_long, act_lat = ego.actions
        # state = {n: None for n in ego.neighbours.keys()}
        # state['ego'] = [ego.speed, ego.glob_x, act_long, act_lat, ego.lane_y]
        # for key, neighbour in ego.neighbours.items():
        #     if neighbour:
        #         act_long, _ = neighbour.actions
        #         follower_aggress = neighbour.driver_params['aggressiveness']
        #         if neighbour.neighbours['f']:
        #             follower_atten = neighbour.neighbours['f'].id
        #         else:
        #             follower_atten = 0
        #         state[key] = [neighbour.speed, neighbour.glob_x, act_long, \
        #                       follower_aggress, follower_atten, neighbour.id]
        #     else:
        #         state[key] = None
        #
        #
        # if not ego.id in self.recordings['info']:
        #     self.recordings['info'][ego.id] = ego.driver_params
        #
        # if not ego.id in self.recordings['states']:
        #     self.recordings['states'][ego.id] = []
        #     self.recordings['decisions'][ego.id] = []
        #     self.recordings['time_step'][ego.id] = []
        #
        # self.recordings['states'][ego.id].append(state)
        # self.recordings['decisions'][ego.id].append(lane_decision)
        # self.recordings['time_step'][ego.id].append(self.time_step)

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            neighbours = vehicle.my_neighbours(self.vehicles)
            vehicle.neighbours = neighbours
            self.recorder(vehicle)
            actions = vehicle.act(neighbours, self.handler.reservations)
            vehicle.actions = actions
            joint_action.append(actions)
            self.handler.update_reservations(vehicle)

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
