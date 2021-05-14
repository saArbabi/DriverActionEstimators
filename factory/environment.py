from factory.viewer import Viewer

class Env():
    def __init__(self):
        self.viewer = None
        self.vehicles = [] # all vehicles
        self.env_clock = 0 # number of steps past
        self.default_config()

    def default_config(self):
        self.config = {'lane_count':3,
                        'lane_width':3.7, # m
                        'lane_length':500, # m
                        'percept_range':200, # m, front and back
                        }

    def step(self, decision=None):
        # low-level actions currently not obs dependant
        for vehicle in self.vehicles:#
            action = vehicle.act()
            vehicle.step(action)

        self.env_clock += 1

    def render(self, model_type=None):
        if self.viewer is None:
            self.viewer = Viewer(model_type, self.config)
            # self.viewer.PAUSE_CONTROL = PAUSE_CONTROL
        self.viewer.env_clock = self.env_clock
        self.viewer.update_plots(self.vehicles)
