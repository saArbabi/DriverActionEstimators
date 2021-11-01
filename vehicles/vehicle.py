import numpy as np

class Vehicle(object):
    STEP_SIZE = 0.1
    def __init__(self, id, lane_id, glob_x, speed):
        self.id = id
        self.lane_id = lane_id
        self.glob_x = glob_x
        self.speed = speed
        self.lane_y = 0
        self.lane_width = 3.75

    def observe(self):
        raise NotImplementedError

    def step(self, actions):
        act_long, act_lat = actions
        self.glob_x +=  self.speed * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE**2
        self.speed +=  act_long * self.STEP_SIZE
        if act_lat != 0:
            self.glob_y +=  act_lat*self.STEP_SIZE
            self.lane_y +=  act_lat*self.STEP_SIZE

            if self.lane_y <= -self.lane_width/2:
                # just stepped into right lane
                self.lane_id  += 1
                self.lane_y += self.lane_width
            elif self.lane_y >= self.lane_width/2:
                # just stepped into left lane
                self.lane_id  -= 1
                self.lane_y -= self.lane_width

    def act(self):
        raise NotImplementedError
