
class Vehicle(object):
    STEP_SIZE = 0.1 # s update rate - used for dynamics
    def __init__(self, id, lane_id, x, v):
        self.v = v # longitudinal speed [m/s]
        self.lane_id = lane_id # inner most right is 1
        self.y = 0 # lane relative
        self.x = x # global coordinate
        self.id = id # 'sdv' or any other integer
        self.y = 2*lane_id*1.85-1.85

    def act(self):
        """
        :param high-lev decision of the car
        """
        pass

    def observe(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def step(self, action):
        """Defines simple vehicle dynamics.
        param: action: [long_acc, lat_speed]
        """
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * action[0] * self.STEP_SIZE **2


        self.v = self.v + action[0] * self.STEP_SIZE

        if action[1] != 0:
            left_lane_bound = 2*self.lane_id*1.85
            righ_lane_bound = 2*self.lane_id*1.85-3.7

            y_delta = action[1]*self.STEP_SIZE
            self.y += y_delta

            if self.y >= left_lane_bound and action[1] > 0:
                self.lane_id += 1

            if self.y <= righ_lane_bound and action[1] < 0:
                self.lane_id -= 1
