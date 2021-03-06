from envs import highway
from importlib import reload
from envs import env_initializor
reload(env_initializor)
from envs.env_initializor import EnvInitializor
from vehicles.idmmobil_merge_vehicle import IDMMOBILVehicleMerge

class EnvMerge(highway.Env):
    def __init__(self, config):
        super().__init__(config)

        self.dummy_stationary_car = IDMMOBILVehicleMerge(\
                                'dummy', 2, config['merge_zone_end'], 0, None)
        self.env_initializor = EnvInitializor(config)

    def initialize_env(self, episode_id):
        self.time_step = 0
        self.env_initializor.next_vehicle_id = 1
        self.env_initializor.dummy_stationary_car = self.dummy_stationary_car
        self.vehicles = self.env_initializor.init_env(episode_id)
        self.lane_length = self.config['lane_length']

    def recorder(self, vehicles):
        """For recording vehicle trajectories. Used for:
        - model training
        """
        for ego in vehicles:
            if not self.episode_id in self.recordings:
                self.recordings[self.episode_id] = {}
            if not ego.id in self.recordings[self.episode_id]:
                self.recordings[self.episode_id][ego.id] = {}

            log = {attrname: getattr(ego, attrname) for attrname in self.veh_log}
            log['f_veh_id'] = None if not ego.neighbours['f'] else ego.neighbours['f'].id
            m_veh = ego.neighbours['m']
            log['m_veh_id'] = None if not m_veh else m_veh.id
            log['att_veh_id'] = None if not ego.neighbours['att'] else ego.neighbours['att'].id
            if ego.id != 'dummy':
                log['aggressiveness'] = ego.driver_params['aggressiveness']
                log['desired_v'] = ego.driver_params['desired_v']
                log['desired_tgap'] = ego.driver_params['desired_tgap']
                log['min_jamx'] = ego.driver_params['min_jamx']
                log['max_act'] = ego.driver_params['max_act']
                log['min_act'] = ego.driver_params['min_act']
                log['act_long_p'] = ego.act_long_p
                log['act_long_c'] = ego.act_long_c
            self.recordings[self.episode_id][ego.id][self.time_step] = log

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(self.vehicles+[self.dummy_stationary_car])
            actions = vehicle.act()
            joint_action.append(actions)
            if self.time_step > 0:
                vehicle.act_long_p = vehicle.act_long_c
            vehicle.act_long_c = actions[0]
        return joint_action

    def remove_unwanted_vehicles(self):
        """vehicles are removed if:
        - they exit highway
        """
        vehicles = []
        for vehicle in self.vehicles:
            if vehicle.glob_x > self.lane_length:
                continue
            vehicles.append(vehicle)
        self.vehicles = vehicles

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        assert self.vehicles, 'Environment not yet initialized'
        # self.remove_unwanted_vehicles()
        joint_action = self.get_joint_action()
        if self.usage == 'data generation':
            self.recorder(self.vehicles+[self.dummy_stationary_car])
        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)
            vehicle.time_lapse += 1

        self.time_step += 1
