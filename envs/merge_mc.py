from envs.merge import EnvMerge
from vehicle_handler import VehicleHandlerMergeMC

import copy

class EnvMergeMC(EnvMerge):
    def __init__(self, config):
        super().__init__(config)
        self.real_vehicles = []
        self.ima_vehicles = []
        self.real_mc_log = {}
        self.ima_mc_log = {}

        self.collision_detected = False
        self.debugging_mode = False
        self.metric_collection_mode = False

    def initialize_env(self, episode_id):
        """Initates the environment, but with a real and imagined (to be predicted)
            copy of traffic state.
        """
        self.time_step = 0
        self.env_initializor.next_vehicle_id = 1
        self.env_initializor.dummy_stationary_car = self.dummy_stationary_car
        vehicles = self.env_initializor.init_env(episode_id)
        for vehicle in vehicles:
            imagined_vehicle = copy.deepcopy(vehicle)
            imagined_vehicle.collision_detected = False
            if imagined_vehicle.lane_id == 2:
                imagined_vehicle.vehicle_type = 'idmmobil_merge'
            self.ima_vehicles.append(imagined_vehicle)
            self.real_vehicles.append(vehicle)

    def idm_to_neural_vehicle(self, vehicle):
        neural_vehicle = copy.deepcopy(self.neural_vehicle)

        for attrname, attrvalue in list(vehicle.__dict__.items()):
            if attrname != 'act':
                setattr(neural_vehicle, attrname, copy.copy(attrvalue))
        return neural_vehicle

    def set_ima_veh_decision(self, veh_real, veh_ima):
        for attrname in ['lane_decision', 'lane_y', 'target_lane']:
            attrvalue = getattr(veh_real, attrname)
            setattr(veh_ima, attrname, attrvalue)

    def set_ima_veh_neighbours(self, veh_real, veh_ima):
        for key, neighbour in veh_real.neighbours.items():
            if neighbour:
                for veh in self.ima_vehicles:
                    if veh.id == neighbour.id or veh.id == 'neur_'+str(neighbour.id):
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
            veh_real.neighbours = veh_real.my_neighbours(self.real_vehicles+[self.dummy_stationary_car])
            act_long, act_lat = veh_real.act()
            acts_real.append([act_long, act_lat])
            veh_real.act_long = act_long

            # imagined vehicles
            self.set_ima_veh_neighbours(veh_real, veh_ima)
            if veh_ima.vehicle_type == 'neural':
                obs = veh_ima.neur_observe(veh_ima, veh_ima.neighbours['f'], \
                                                        veh_ima.neighbours['m'])
                # if veh_real.id == 3:
                #     print(veh_ima.obs_history)
                if not veh_ima.collision_detected:
                    veh_ima.update_obs_history(obs[0])

                    if not (veh_ima.obs_history[0,0,:] == 0).all() and \
                                                        veh_ima.control_type != 'neural':
                        # controller change
                        veh_ima.control_type = 'neural'

                    if veh_ima.control_type == 'neural':
                        print('act_long')
                        # _act_long = veh_ima.act(obs)
                        # act_long = veh_ima.act(obs)
                        act_long = -0.5
                        # _ = veh_ima.act(obs)
                        veh_ima.act_long = act_long
                        if self.metric_collection_mode:
                            self.mc_log_info(veh_real, veh_ima)
                    else:
                        act_long = veh_ima.idm_action(veh_ima, veh_ima.neighbours['att'])
                else:
                    act_long = 0
                    print(act_long)

            elif veh_ima.vehicle_type == 'idmmobil' and not veh_ima.collision_detected:
                try:
                    act_long = veh_ima.idm_action(veh_ima, veh_ima.neighbours['att'])
                except:
                    veh_ima.collision_detected = True

            acts_ima.append([act_long, act_lat]) # lateral action is from veh_real


            if self.debugging_mode:
                if veh_ima.vehicle_type == 'neural':
                    if veh_ima.control_type == 'neural':
                        # veh_ima.act_long = _act_long
                        veh_ima.act_long = act_long
                    else:
                        veh_ima.act_long = act_long
                else:
                    veh_ima.act_long = act_long

                self.vis_log_info(veh_real, veh_ima)

        return acts_real, acts_ima

    def neuralize_vehicle_type(self):
        """Given some conditionals, it neuralizes the vehicle type.
        Note: the controller only changes when sufficient histroy is collected.
        """
        ima_vehicles = []
        for vehicle in self.ima_vehicles:
            if vehicle.vehicle_type != 'idmmobil_merge' \
                    and vehicle.neighbours['att'] and vehicle.time_lapse > 0:
                neural_vehicle = self.idm_to_neural_vehicle(vehicle)
                neural_vehicle.id = 'neur_'+str(vehicle.id)
                imagined_vehicle = neural_vehicle
                imagined_vehicle.vehicle_type = 'neural'
                ima_vehicles.append(imagined_vehicle)
            else:
                ima_vehicles.append(vehicle)
        self.ima_vehicles = ima_vehicles

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

        if self.time_step == 10:
            self.neuralize_vehicle_type()
        self.time_step += 1

    def vis_log_info(self, veh_real, veh_ima):
        """
        For one off visualisation and debugging.
        """
        if veh_ima.vehicle_type == 'neural':
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
                for key in veh_ima.driver_params.keys():
                    self.real_mc_log[veh_id][key] = []

                self.ima_mc_log[veh_id]['act'] = []
                self.ima_mc_log[veh_id]['speed'] = []
                self.ima_mc_log[veh_id]['att'] = []
                self.ima_mc_log[veh_id]['m_veh_exists'] = []
                for key in veh_ima.driver_params.keys():
                    self.ima_mc_log[veh_id][key] = []

            self.real_mc_log[veh_id]['act'].append(veh_real.act_long)
            self.real_mc_log[veh_id]['speed'].append(veh_real.speed)
            self.real_mc_log[veh_id]['att'].append(att_real)
            for key in veh_ima.driver_params.keys():
                self.real_mc_log[veh_id][key].append(veh_real.driver_params[key])

            # Imagined vehicle
            self.ima_mc_log[veh_id]['act'].append(veh_ima.act_long)
            self.ima_mc_log[veh_id]['speed'].append(veh_ima.speed)

            if not hasattr(veh_ima, 'att'):
                self.ima_mc_log[veh_id]['att'].append(att_real)
            else:
                self.ima_mc_log[veh_id]['att'].append(veh_ima.att)


            for key in veh_ima.driver_params.keys():
                self.ima_mc_log[veh_id][key].append(veh_ima.driver_params[key])

            # self.ima_mc_log[veh_id]['m_veh_exists'].append(\
            #                                     veh_ima.m_veh_exists)


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
