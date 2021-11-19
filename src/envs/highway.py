class Env:
    def __init__(self, config):
        self.config = config
        self.time_step = 0
        self.sdv = None
        self.vehicles = []
        self.usage = None
        # self.initiate_environment()

    def initiate_environment(self):
        self.handler = VehicleHandler(config)
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
