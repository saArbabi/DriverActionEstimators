from vehicles.vehicle import Vehicle
import numpy as np

class IDMVehicle(Vehicle):
    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        if not idm_param:
            self.default_config()
        else:
            self.idm_param = idm_param
        self.seed()
        # TODO params can also be learned

    def seed(self, seed_value=2021):
        self.rng = np.random.RandomState()
        self.rng.seed(seed_value)

    def default_config(self):
        # TODO nonstationary params
        self.idm_param = {
                        'desired_v':self.v, # m/s
                        'desired_tgap':2.8, # s
                        'min_jamx':0, # m
                        'max_acc':3, # m/s^2
                        'max_decc':3, # m/s^2
                        }

        self.desired_v = self.idm_param['desired_v']
        self.desired_tgap = self.idm_param['desired_tgap']
        self.min_jamx = self.idm_param['min_jamx']
        self.max_acc = self.idm_param['max_acc']
        self.max_decc = self.idm_param['max_decc']

    def get_desired_gap(self, dv):
        gap = self.min_jamx + self.desired_tgap*self.v+(self.v*dv)/ \
                                        (2*np.sqrt(self.max_acc*self.max_decc))
        return gap

    def act(self, obs):
        desired_gap = self.get_desired_gap(obs['dv'])
        acc = self.max_acc*(1-(self.v/self.desired_v)**4-\
                                            (desired_gap/obs['dx'])**2)
        acc += self.rng.normal(0, 1, 1)
        return [sorted([-3, acc, 3])[1], 0]
        # return [0, 0]
        # return [np.random.uniform(-2,2), 0]

    def observe(self, vehicles):
        """
        return: the ego-centric observation based on which a vehicle can act
        """
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
            obs = {'dv':0 , 'dx':30}
        else:
            obs = {'dv':candid_veh.v - self.v , 'dx':candid_dx}

        return obs
