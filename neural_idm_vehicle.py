from idmmobil_vehicle import IDMMOBILVehicle

class NeuralIDMVehicle(IDMMOBILVehicle):
    def __init__(self):
        super().__init__(id=None, lane_id=None, glob_x=None, speed=None, aggressiveness=None)

    def act(self, reservations):
        act_long = self.idm_action(self.observe(self, self.neighbours['f']))
        return [max(-3, min(act_long, 3)), 0]

    def get_obs(self, ):
        """
        Returns an observation sequence conditioned on:
        - neighbouring cars
        - merger
        Note:
        """
