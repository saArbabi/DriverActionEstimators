
from envs.highway import Env
from vehicle_handler import VehicleHandlerMerge
from vehicles.idmmobil_vehicle import IDMMOBILVehicle
from importlib import reload
import vehicle_handler
reload(vehicle_handler)
from vehicle_handler import VehicleHandler

class EnvMerge(Env):
    def __init__(self, config):
        super().__init__(config)
        self.dummy_stationary_car = IDMMOBILVehicle('dummy', 2, 400, 0, None)
        self.handler = VehicleHandlerMerge(config)

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
        self.remove_vehicles_outside_bound()
        joint_action = self.get_joint_action()
        if self.usage == 'data generation':
            self.recorder()
        for vehicle, actions in zip(self.vehicles, joint_action):
            vehicle.step(actions)
            vehicle.time_lapse += 1
            if vehicle.speed < 15 and vehicle.lane_decision == 'keep_lane':
                vehicle_stuck = True


        if not vehicle_stuck:
            new_entries = self.handler.handle_vehicle_entries(
                                                              self.queuing_entries,
                                                              self.last_entries)
            if new_entries:
                self.vehicles.extend(new_entries)
        self.time_step += 1
