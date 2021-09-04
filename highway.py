# from factory.viewer import Viewer
from importlib import reload
import copy
import vehicle_handler
reload(vehicle_handler)
from vehicle_handler import VehicleHandler
import copy
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
            log['desired_v'] = ego.driver_params['desired_v']
            log['desired_tgap'] = ego.driver_params['desired_tgap']
            log['min_jamx'] = ego.driver_params['min_jamx']
            log['max_act'] = ego.driver_params['max_act']
            log['min_act'] = ego.driver_params['min_act']
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
            vehicle.time_lapse += 1

        new_entries = self.handler.handle_vehicle_entries(
                                                          self.queuing_entries,
                                                          self.last_entries)
        if new_entries:
            self.vehicles.extend(new_entries)
        self.time_step += 1

class EnvMC(Env):
    def __init__(self, config):
        super().__init__(config)
        self.real_vehicles = []
        self.ima_vehicles = []
        self.real_mc_log = {}
        self.ima_mc_log = {}
        self.collision_detected = False
        self.debugging_mode = False
        self.metric_collection_mode = False


    def prohibit_lane_change(self, vehicle):
        """
        For cars to be modelled for MC
        """
        def _act(self, reservations):
            act_long = self.idm_action(self, self.neighbours['att'])
            return [max(-3, min(act_long, 3)), 0]
        vehicle.act = types.MethodType(_act, vehicle)

    def idm_to_neural_vehicle(self, vehicle):
        neural_vehicle = copy.deepcopy(self.neural_vehicle)
        for attrname, attrvalue in list(vehicle.__dict__.items()):
            if attrname != 'act':
                setattr(neural_vehicle, attrname, copy.copy(attrvalue))
        return neural_vehicle

    def add_new_vehicles(self):
        new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
                                                          self.last_entries)
        for vehicle in new_entries:
            imagined_vehicle = copy.deepcopy(vehicle)
            imagined_vehicle.collision_detected = False
            self.ima_vehicles.append(imagined_vehicle)
            self.real_vehicles.append(vehicle)

    def set_ima_veh_decision(self, veh_real, veh_ima):
        for attrname in ['lane_decision', 'lane_y', 'target_lane']:
            attrvalue = getattr(veh_real, attrname)
            setattr(veh_ima, attrname, attrvalue)

    def set_ima_veh_neighbours(self, veh_real, veh_ima):
        for key, neighbour in veh_real.neighbours.items():
            if neighbour:
                for veh in self.ima_vehicles:
                    if veh.id == neighbour.id:
                        veh_ima.neighbours[key] = veh
            else:
                veh_ima.neighbours[key] = None

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        acts_real = []
        acts_ima = []
        for veh_real, veh_ima in zip(self.real_vehicles, self.ima_vehicles):
            # real vehicles
            veh_real.neighbours = veh_real.my_neighbours(self.real_vehicles)
            act_long, act_lat = veh_real.act(self.handler.reservations)
            acts_real.append([act_long, act_lat])
            veh_real.act_long = act_long
            self.handler.update_reservations(veh_real)

            # imagined vehicles
            self.set_ima_veh_neighbours(veh_real, veh_ima)
            if veh_ima.vehicle_type == 'neural':
                obs = veh_ima.neur_observe(veh_ima, veh_ima.neighbours['f'], \
                                                        veh_ima.neighbours['m'])
                if not veh_ima.collision_detected:
                    veh_ima.update_obs_history(obs[0])
                    if veh_ima.time_lapse > 35 and veh_ima.control_type != 'neural':
                        veh_ima.control_type = 'neural'

                    if veh_ima.control_type == 'neural':
                        # act_long = 2.5
                        act_long = veh_ima.act(obs)
                        act_long = max(-3, min(act_long, 3))
                        if self.metric_collection_mode:
                            veh_ima.act_long = act_long
                            self.mc_log_info(veh_real, veh_ima)
                    else:
                        act_long = veh_ima.idm_action(veh_ima, veh_ima.neighbours['att'])
                else:
                    act_long = 0

            elif veh_ima.vehicle_type == 'idmmobil' and not veh_ima.collision_detected:
                try:
                    act_long = veh_ima.idm_action(veh_ima, veh_ima.neighbours['att'])
                except:
                    veh_ima.collision_detected = True
            else:
                act_long = 0


            acts_ima.append([act_long, act_lat]) # lateral action is from veh_real

            if self.debugging_mode:
                veh_ima.act_long = act_long
                self.vis_log_info(veh_real, veh_ima)
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
            veh_real.time_lapse += 1
            veh_ima.time_lapse += 1

        if self.time_step > 300:
            ima_vehicles = []
            for vehicle in self.ima_vehicles:
                if 50 < vehicle.time_lapse < 200 and vehicle.vehicle_type != 'neural':
                    neural_vehicle = self.idm_to_neural_vehicle(vehicle)
                    # neural_vehicle.id = 'neur_'+str(vehicle.id)
                    imagined_vehicle = neural_vehicle
                    imagined_vehicle.vehicle_type = 'neural'
                    imagined_vehicle.time_lapse = 0
                    ima_vehicles.append(imagined_vehicle)
                else:
                    ima_vehicles.append(vehicle)
            self.ima_vehicles = ima_vehicles

        self.add_new_vehicles()
        self.time_step += 1

    def vis_log_info(self, veh_real, veh_ima):
        """
        For on off visualisation and debugging.
        """

        if veh_ima.id in self.vis_vehicles and veh_ima.vehicle_type == 'neural':
            veh_id =  veh_real.id
            if veh_real.neighbours['m'] and\
                                veh_real.neighbours['att'] == veh_real.neighbours['m']:
                att_real = 1
            else:
                att_real = 0

            if veh_id not in self.real_mc_log:
                self.real_mc_log[veh_id] = {}
                self.ima_mc_log[veh_id] = {}

                self.real_mc_log[veh_id]['act'] = []
                self.real_mc_log[veh_id]['speed'] = []
                self.real_mc_log[veh_id]['att'] = []
                self.real_mc_log[veh_id]['desvel'] = []

                self.ima_mc_log[veh_id]['act'] = []
                self.ima_mc_log[veh_id]['speed'] = []
                self.ima_mc_log[veh_id]['att'] = []
                self.ima_mc_log[veh_id]['desvel'] = []
                self.ima_mc_log[veh_id]['m_veh_exists'] = []

            self.real_mc_log[veh_id]['act'].append(veh_real.act_long)
            self.real_mc_log[veh_id]['speed'].append(veh_real.speed)
            self.real_mc_log[veh_id]['att'].append(att_real)
            self.real_mc_log[veh_id]['desvel'].append(\
                                               veh_real.driver_params['desired_v'])

            # Imagined vehicle
            self.ima_mc_log[veh_id]['act'].append(veh_ima.act_long)
            self.ima_mc_log[veh_id]['speed'].append(veh_ima.speed)

            if not hasattr(self, 'att'):
                self.ima_mc_log[veh_id]['att'].append(att_real)
            else:
                self.ima_mc_log[veh_id]['att'].append(veh_ima.att)

            self.ima_mc_log[veh_id]['desvel'].append(\
                                                veh_ima.driver_params['desired_v'])
            self.ima_mc_log[veh_id]['m_veh_exists'].append(\
                                                veh_ima.m_veh_exists)


    def mc_log_info(self, veh_real, veh_ima):
        """
        Informatin to be logged:
        - ego (real and imagined) global_x for rwse and collision detection
        - ego (real and imagined) speed for rwse
        - ego (real and imagined) action for comparing action distributions
        """
        veh_id =  veh_real.id
        if veh_id not in self.real_mc_log:
            self.real_mc_log[veh_id] = {}
            self.ima_mc_log[veh_id] = {}

            self.real_mc_log[veh_id] = []
            self.ima_mc_log[veh_id] = []

        if veh_real.neighbours['m'] and\
                            veh_real.neighbours['att'] == veh_real.neighbours['m']:
            att_real = 1
        else:
            att_real = 0

        if veh_real.neighbours['att']:
            min_delta_x = veh_real.neighbours['att'].glob_x - veh_real.glob_x
        else:
            min_delta_x = 100
        real_mc_log = [self.time_step, veh_real.glob_x, \
                            veh_real.speed, veh_real.act_long, min_delta_x, att_real]
        self.real_mc_log[veh_id].append(real_mc_log)

        # Imagined vehicle
        if veh_ima.neighbours['att']:
            min_delta_x = veh_ima.neighbours['att'].glob_x - veh_ima.glob_x
        else:
            min_delta_x = 100

        # ima_mc_log = [self.time_step, veh_ima.glob_x, \
        #                         veh_ima.speed, veh_ima.act_long, min_delta_x, veh_ima.att]
        ima_mc_log = [self.time_step, veh_ima.glob_x, \
                                veh_ima.speed, veh_ima.act_long, min_delta_x]
        self.ima_mc_log[veh_id].append(ima_mc_log)
# class EnvLaneKeep(Env):
#     def __init__(self, config):
#         super().__init__(config)
#         self.vehicles = []
#         self.handler = VehicleHandlerLaneKeep(config)
#
#     def add_new_vehicles(self):
#         new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
#                                                           self.last_entries)
#         for vehicle in new_entries:
#             self.vehicles.append(vehicle)
#
#     def recorder(self):
#         for ego in self.vehicles:
#             if ego.glob_x < 0:
#                 continue
#             if not ego.id in self.recordings:
#                 self.recordings[ego.id] = {}
#             log = {attrname: getattr(ego, attrname) for attrname in self.veh_log}
#             log['att_veh_id'] = None if not ego.neighbours['att'] else ego.neighbours['att'].id
#             log['aggressiveness'] = ego.driver_params['aggressiveness']
#             log['act_long'] = ego.act_long
#             self.recordings[ego.id][self.time_step] = log
#
#     def get_joint_action(self):
#         """
#         Returns the joint action of all vehicles on the road
#         """
#         acts_real = []
#         acts_ima = []
#         for veh_real in self.vehicles:
#             veh_real.neighbours = veh_real.my_neighbours(self.vehicles)
#             obs = veh_real.observe(veh_real, veh_real.neighbours['att'])
#             act_long = veh_real.idm_action(obs)
#             acts_real.append([act_long, 0])
#             veh_real.act_long = act_long
#
#         return acts_real
#
#     def step(self, actions=None):
#         """ steps the environment forward in time.
#         """
#         self.remove_vehicles_outside_bound()
#         acts_real = self.get_joint_action()
#         if self.usage == 'data generation':
#             self.recorder()
#
#         for veh_real, act_real in zip(self.vehicles, acts_real):
#             veh_real.step(act_real)
#
#         self.add_new_vehicles()
#         self.time_step += 1

# class EnvLaneKeepMC(Env):
#     def __init__(self, config):
#         self.config = config
#         self.time_step = 0
#         self.handler = VehicleHandlerLaneKeep(config)
#         self.usage = None
#         self.real_vehicles = []
#         self.ima_vehicles = []
#         self.initiate_environment()
#
#     def idm_to_neural_vehicle(self, vehicle):
#         # neural_vehicle = NeuralIDMVehicle()
#         neural_vehicle = LSTMVehicle()
#         for attrname, attrvalue in list(vehicle.__dict__.items()):
#             setattr(neural_vehicle, attrname, attrvalue)
#         return neural_vehicle
#
#     def add_new_vehicles(self):
#         new_entries = self.handler.handle_vehicle_entries(self.queuing_entries,
#                                                           self.last_entries)
#         for vehicle in new_entries:
#             if vehicle.id > 1:
#                 neural_vehicle = self.idm_to_neural_vehicle(vehicle)
#                 neural_vehicle.id = 'neur_'+str(vehicle.id)
#                 imagined_vehicle = neural_vehicle
#                 imagined_vehicle.vehicle_type = 'neural'
#             else:
#                 imagined_vehicle = copy.copy(vehicle)
#                 imagined_vehicle.vehicle_type = 'idm'
#
#             self.ima_vehicles.append(imagined_vehicle)
#             self.real_vehicles.append(vehicle)
#
#     def get_joint_action(self):
#         """
#         Returns the joint action of all vehicles on the road
#         """
#         acts_real = []
#         acts_ima = []
#         for veh_real, veh_ima in zip(self.real_vehicles, self.ima_vehicles):
#             veh_real.neighbours = veh_real.my_neighbours(self.real_vehicles)
#             obs = veh_real.observe(veh_real, veh_real.neighbours['att'])
#             act_long = veh_real.idm_action(obs)
#             acts_real.append([act_long, 0])
#             veh_real.act_long = act_long
#             if veh_ima.vehicle_type == 'neural':
#                 veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles)
#                 obs = veh_ima.observe(veh_ima, veh_ima.neighbours['att'])
#                 act_long = veh_ima.act(obs)
#                 acts_ima.append([act_long, 0])
#                 veh_ima.act_long = act_long
#
#             elif veh_ima.vehicle_type == 'idm':
#                 veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles)
#                 obs = veh_ima.observe(veh_ima, veh_ima.neighbours['att'])
#                 act_long = veh_ima.idm_action(obs)
#                 acts_ima.append([act_long, 0])
#                 veh_ima.act_long = act_long
#
#         return acts_real, acts_ima
#
#     def step(self, actions=None):
#         """ steps the environment forward in time.
#         """
#         # self.remove_vehicles_outside_bound()
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
