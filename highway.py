# from factory.viewer import Viewer
from importlib import reload
import copy
import vehicle_handler
reload(vehicle_handler)
from vehicle_handler import VehicleHandler
import copy
from vehicles.neural_vehicles import NeuralIDMVehicle, LSTMVehicle
import types

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


    def recorder(self):
        """For recording vehicle trajectories. Used for:
        - model training
        - perfromance validations # TODO
        """
        for ego in self.vehicles:
            if ego.glob_x < 0:
                continue
            if not ego.id in self.recordings:
                self.recordings[ego.id] = {}
            log = {attrname: getattr(ego, attrname) for attrname in self.veh_log}
            log['f_veh_id'] = None if not ego.neighbours['f'] else ego.neighbours['f'].id
            log['m_veh_id'] = None if not ego.neighbours['m'] else ego.neighbours['m'].id
            log['att_veh_id'] = None if not ego.neighbours['att'] else ego.neighbours['att'].id
            log['aggressiveness'] = ego.driver_params['aggressiveness']
            log['act_long'] = ego.act_long
            self.recordings[ego.id][self.time_step] = log
            # if ego.id == 14:
            #     print('##### sim ####')
            #     print(self.time_step)
            #     print(ego.lane_decision)
            #     print(ego.neighbours['f'].glob_x - ego.glob_x)
            #     print('front_id ', ego.neighbours['f'].id)
            #     print('##### sim ####')
                # print(log)
                # print(ego.glob_x - ego.neighbours['f'].glob_x)

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(self.vehicles)
            actions = vehicle.act(self.handler.reservations)
            joint_action.append(actions)
            vehicle.act_long = actions[0]
            self.handler.update_reservations(vehicle)
        return joint_action

    def remove_vehicles_outside_bound(self):
        vehicles = []
        for vehicle in self.vehicles:
            if vehicle.glob_x > self.lane_length:
                # vehicle has left the highway
                if vehicle.id in self.handler.reservations:
                    del self.handler.reservations[vehicle.id]
                continue
            vehicles.append(vehicle)
        self.vehicles = vehicles

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        self.remove_vehicles_outside_bound()
        joint_action = self.get_joint_action()
        if self.usage == 'data generation':
            self.recorder()
        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)

        new_entries = self.handler.handle_vehicle_entries(
                                                          self.queuing_entries,
                                                          self.last_entries)
        if new_entries:
            self.vehicles.extend(new_entries)
        self.time_step += 1

class EnvLaneKeep(Env):
    def __init__(self, config):
        super().__init__(config)
        self.vehicles = []
        self.handler = VehicleHandlerLaneKeep(config)

    def add_new_vehicles(self):
        new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
                                                          self.last_entries)
        for vehicle in new_entries:
            self.vehicles.append(vehicle)

    def recorder(self):
        for ego in self.vehicles:
            if ego.glob_x < 0:
                continue
            if not ego.id in self.recordings:
                self.recordings[ego.id] = {}
            log = {attrname: getattr(ego, attrname) for attrname in self.veh_log}
            log['att_veh_id'] = None if not ego.neighbours['att'] else ego.neighbours['att'].id
            log['aggressiveness'] = ego.driver_params['aggressiveness']
            log['act_long'] = ego.act_long
            self.recordings[ego.id][self.time_step] = log

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        acts_real = []
        acts_ima = []
        for veh_real in self.vehicles:
            veh_real.neighbours = veh_real.my_neighbours(self.vehicles)
            obs = veh_real.observe(veh_real, veh_real.neighbours['att'])
            act_long = veh_real.idm_action(obs)
            acts_real.append([act_long, 0])
            veh_real.act_long = act_long

        return acts_real

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        self.remove_vehicles_outside_bound()
        acts_real = self.get_joint_action()
        if self.usage == 'data generation':
            self.recorder()

        for veh_real, act_real in zip(self.vehicles, acts_real):
            veh_real.step(act_real)

        self.add_new_vehicles()
        self.time_step += 1

class EnvLaneKeepMC(Env):
    def __init__(self, config):
        self.config = config
        self.time_step = 0
        self.handler = VehicleHandlerLaneKeep(config)
        self.usage = None
        self.real_vehicles = []
        self.ima_vehicles = []
        self.initiate_environment()

    def idm_to_neural_vehicle(self, vehicle):
        # neural_vehicle = NeuralIDMVehicle()
        neural_vehicle = LSTMVehicle()
        for attrname, attrvalue in list(vehicle.__dict__.items()):
            setattr(neural_vehicle, attrname, attrvalue)
        return neural_vehicle

    def add_new_vehicles(self):
        new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
                                                          self.last_entries)
        for vehicle in new_entries:
            if vehicle.id > 1:
                neural_vehicle = self.idm_to_neural_vehicle(vehicle)
                neural_vehicle.id = 'neur_'+str(vehicle.id)
                imagined_vehicle = neural_vehicle
                imagined_vehicle.vehicle_type = 'neural'
            else:
                imagined_vehicle = copy.copy(vehicle)
                imagined_vehicle.vehicle_type = 'idm'

            self.ima_vehicles.append(imagined_vehicle)
            self.real_vehicles.append(vehicle)

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        acts_real = []
        acts_ima = []
        for veh_real, veh_ima in zip(self.real_vehicles, self.ima_vehicles):
            veh_real.neighbours = veh_real.my_neighbours(self.real_vehicles)
            obs = veh_real.observe(veh_real, veh_real.neighbours['att'])
            act_long = veh_real.idm_action(obs)
            acts_real.append([act_long, 0])
            veh_real.act_long = act_long
            if veh_ima.vehicle_type == 'neural':
                veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles)
                obs = veh_ima.observe(veh_ima, veh_ima.neighbours['att'])
                act_long = veh_ima.act(obs)
                acts_ima.append([act_long, 0])
                veh_ima.act_long = act_long

            elif veh_ima.vehicle_type == 'idm':
                veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles)
                obs = veh_ima.observe(veh_ima, veh_ima.neighbours['att'])
                act_long = veh_ima.idm_action(obs)
                acts_ima.append([act_long, 0])
                veh_ima.act_long = act_long

        return acts_real, acts_ima

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        # self.remove_vehicles_outside_bound()
        acts_real, acts_ima = self.get_joint_action()
        for veh_real, veh_ima, act_real, act_ima in zip(
                                                    self.real_vehicles,
                                                    self.ima_vehicles,
                                                    acts_real,
                                                    acts_ima):

            veh_real.step(act_real)
            veh_ima.step(act_ima)

        self.add_new_vehicles()
        self.time_step += 1

#
# class EnvMCLC(Env):
#     def __init__(self, config):
#         super().__init__(config)
#         self.real_vehicles = []
#         self.ima_vehicles = []
#
#     def prohibit_lane_change(self, vehicle):
#         """
#         For cars to be modelled for MC
#         """
#         def _act(self, reservations):
#             act_long = self.idm_action(self.observe(self, self.neighbours['f']))
#             return [max(-3, min(act_long, 3)), 0]
#         vehicle.act = types.MethodType(_act, vehicle)
#
#     def idm_to_neural_vehicle(self, vehicle):
#         neural_vehicle = NeuralIDMVehicle()
#         for attrname, attrvalue in list(vehicle.__dict__.items()):
#             setattr(neural_vehicle, attrname, attrvalue)
#         return neural_vehicle
#
#     def add_new_vehicles(self):
#         new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
#                                                           self.last_entries)
#         for vehicle in new_entries:
#             if vehicle.id in [1, 15]:
#                 self.prohibit_lane_change(vehicle)
#                 neural_vehicle = self.idm_to_neural_vehicle(vehicle)
#                 neural_vehicle.id = 'neur_'+str(vehicle.id)
#                 imagined_vehicle = neural_vehicle
#                 imagined_vehicle.vehicle_type = 'neural'
#             else:
#                 imagined_vehicle = copy.copy(vehicle)
#                 imagined_vehicle.vehicle_type = 'idmmobil'
#
#             self.ima_vehicles.append(imagined_vehicle)
#             self.real_vehicles.append(vehicle)
#
#
#     def set_ima_veh_decision(self, veh_real, veh_ima):
#         for attrname in ['lane_decision', 'lane_y', 'target_lane']:
#             attrvalue = getattr(veh_real, attrname)
#             setattr(veh_ima, attrname, attrvalue)
#
#     def get_joint_action(self):
#         """
#         Returns the joint action of all vehicles on the road
#         """
#         acts_real = []
#         acts_ima = []
#         for veh_real, veh_ima in zip(self.real_vehicles, self.ima_vehicles):
#             veh_real.neighbours = veh_real.my_neighbours(self.real_vehicles)
#             act_long, act_lat = veh_real.act(self.handler.reservations)
#             acts_real.append([act_long, act_lat])
#             veh_real.act_long = act_long
#             self.handler.update_reservations(veh_real)
#             if veh_ima.vehicle_type == 'neural':
#                 acts_ima.append([0, 0])
#
#             elif veh_ima.vehicle_type == 'idmmobil':
#                 veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles)
#                 self.set_ima_veh_decision(veh_real, veh_ima)
#                 act_long, _ = veh_ima.act(self.handler.reservations)
#                 acts_ima.append([act_long, act_lat]) # act lat is true
#                 veh_ima.act_long = act_long
#
#         return acts_real, acts_ima
#
#
#     def step(self, actions=None):
#         """ steps the environment forward in time.
#         """
#         # self.remove_vehicles_outside_bound()
#         # joint_action = self.get_joint_action(self.real_vehicles+self.ima_vehicles)
#
#         acts_real, acts_ima = self.get_joint_action()
#         for veh_real, veh_ima, act_real, act_ima in zip(
#                                                     self.real_vehicles,
#                                                     self.ima_vehicles,
#                                                     acts_real,
#                                                     acts_ima):
#
#             veh_real.step(act_real)
#             veh_ima.step(act_ima)
#
#         self.add_new_vehicles()
#         self.time_step += 1
