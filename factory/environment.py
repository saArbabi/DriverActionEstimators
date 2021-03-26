from factory.viewer import Viewer

class Env():
    def __init__(self):
        self.viewer = None
        self.vehicles = [] # all vehicles
        self.env_clock = 0 # time past since the episode's start
        self.default_config()

    def default_config(self):
        self.config = {'lane_count':3,
                        'lane_width':3.7, # m
                        'lane_length':10000, # m
                        'percept_range':500, # m, front and back
                        }

    def step(self, decision=None):
        # low-level actions currently not obs dependant
        for vehicle in self.vehicles:#
            action = vehicle.act()
            vehicle.step(action)
        self.env_clock += 0.1

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.config)
            # self.viewer.PAUSE_CONTROL = PAUSE_CONTROL
        self.viewer.update_plots(self.vehicles, self.env_clock)
