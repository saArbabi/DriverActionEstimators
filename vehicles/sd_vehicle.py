from vehicles.vehicle import Vehicle
import numpy as np
from mcts import MCTSDPW

class SDVehicle(Vehicle):
    OPTIONS = {
                0: ['LK', 'UP'],
                1: ['LK', 'DOWN'],
                2: ['LK', 'IDLE'],
                3: ['LCL', 'UP'],
                4: ['LCL', 'DOWN'],
                5: ['LCL', 'IDLE'],
                6: ['LCR', 'UP'],
                7: ['LCR', 'DOWN'],
                8: ['LCR', 'IDLE']
                }

    def __init__(self, id, lane_id, x, v, env):
        super().__init__(id, lane_id, x, v)
        self.env = env
        self.planner = MCTSDPW()
        self.time_budget = 1 # s


    def act(self, decision, obs):
        if self.OPTIONS[decision][0] == 'LK':
            act_lat = 0
        elif self.OPTIONS[decision][0] == 'LCL':
            act_lat = 0.75
        elif self.OPTIONS[decision][0] == 'LCR':
            act_lat = -0.75

        if self.OPTIONS[decision][1] == 'IDLE':
            act_long = 0
        elif self.OPTIONS[decision][1] == 'UP':
            act_long = 1
        elif self.OPTIONS[decision][1] == 'DOWN':
            act_long = -1

        return  [act_long, act_lat]

    def observe(self, vehicles):
        """
        return: the ego-centric observation based on which a vehicle can act
        """
        obs = {'front_dx':0, 'rear_dx':0}
        candid_veh = None
        candid_dx = None
        for vehicle in vehicles:
            if vehicle.id != self.id and vehicle.lane_id == self.lane_id and \
                                                            vehicle.x >= self.x:

                if not candid_veh:
                    candid_veh = vehicle
                    candid_dx = vehicle.x - self.x

                elif vehicle.x - self.x < candid_dx:
                    candid_veh = vehicle
                    candid_dx = vehicle.x - self.x

        if not candid_veh:
            obs['front_dx'] = 30
        else:
            obs['front_dx'] = candid_dx

        candid_veh = None
        candid_dx = None
        for vehicle in vehicles:
            if vehicle.id != self.id and vehicle.lane_id == self.lane_id and \
                                                            vehicle.x <= self.x:

                if not candid_veh:
                    candid_veh = vehicle
                    candid_dx = -(vehicle.x - self.x)

                elif vehicle.x - self.x < candid_dx:
                        candid_veh = vehicle
                        candid_dx = -(vehicle.x - self.x)

        if not candid_veh:
            obs['rear_dx'] = 30
        else:
            obs['rear_dx'] = candid_dx

        return obs
