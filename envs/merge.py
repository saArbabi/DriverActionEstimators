from envs.highway import Env
from vehicles.idmmobil_vehicle import IDMMOBILVehicle
from importlib import reload
from env_initializor import EnvInitializor


class EnvMerge(Env):
    def __init__(self, config):
        super().__init__(config)
        self.dummy_stationary_car = IDMMOBILVehicle('dummy', 2, 500, 0, None)
        self.vehicles = EnvInitializor(config).init_env(100)

    def get_joint_action(self):
        """
        Returns the joint action of all vehicles on the road
        """
        joint_action = []
        for vehicle in self.vehicles:
            vehicle.neighbours = vehicle.my_neighbours(self.vehicles+[self.dummy_stationary_car])
            actions = vehicle.act()
            joint_action.append(actions)
            vehicle.act_long = actions[0]
            # self.handler.update_reservations(vehicle)
        return joint_action

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        vehicle_stuck = False
        joint_action = self.get_joint_action()
        if self.usage == 'data generation':
            self.recorder()
        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)
            vehicle.time_lapse += 1
            if vehicle.speed < 15 and vehicle.lane_decision == 'keep_lane':
                vehicle_stuck = True

        self.time_step += 1
