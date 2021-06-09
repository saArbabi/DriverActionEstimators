class Vehicle(object):
    STEP_SIZE = 0.1 # s update rate - used for dynamics
    def __init__(self, id, lane_id, x, v):
        self.v = v # longitudinal speed [m/s]
        self.lane_id = lane_id # inner most right is 1
        self.y_lane = 0 # lane relative
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

    def __init__(self, id, lane_id, x, v, idm_param=None):
        super().__init__(id, lane_id, x, v)
        self.id = id
        self.lane_y = 3
        self.time_budget = 0


    def act(self, decision, obs):
        self.time_budget = 1

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

        if self.time_budget >= 0:
            self.time_budget -= 0.1
        else:
            self.time_budget = 1


        return  [act_long, act_lat]

    def step(self, action):
        act_long, act_lat = action
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE **2

        self.v = self.v + act_long * self.STEP_SIZE
        self.y = self.y + act_lat * self.STEP_SIZE
