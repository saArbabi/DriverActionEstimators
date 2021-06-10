from factory.viewer import Viewer

class Env():
    def __init__(self):
        self.viewer = None
        self.vehicles = [] # all vehicles
        self.elapsed_time = 0 # number of steps past
        self.default_config()

    def default_config(self):
        self.config = {'lane_count':3,
                        'lane_width':3.7, # m
                        'lane_length':5000, # m
                        'percept_range':70, # m, front and back
                        }

    def step(self, sdv_action=None):
        if not sdv_action:
            # first observation
            return self.observe(0)

        # sdv_obs = self.sdv.observe(self.vehicles)
        # sdv_action = self.sdv.act(action, None)

        # while self.sdv.time_budget >= 0:
        self.vehicles['sdv'].step(sdv_action)
        for vehicle_id in self.vehicle_ids:#
            if vehicle_id != 'sdv':
                action = self.vehicles[vehicle_id].act()
                self.vehicles[vehicle_id].step(action)

            if vehicle_id == 'follower':
                follower_action = action
            # for vehicle in self.vehicles:#
            #     if vehicle.id == 'sdv':
            #         vehicle.step(sdv_action)
            #     else:
            #         action = vehicle.act()
            #         vehicle.step(action)

            # self.sdv.time_budget -= 0.1
        self.observe(follower_action)
        self.elapsed_time += 1
        return self.observe(follower_action)

    def observe(self, follower_action):
        lf_dx = self.vehicles['leader'].x-self.vehicles['follower'].x
        mf_dx = self.vehicles['sdv'].x-self.vehicles['follower'].x
        fl_dv = self.vehicles['follower'].v - self.vehicles['leader'].v
        fm_dv = self.vehicles['follower'].v - self.vehicles['sdv'].v
        leader_feature = [self.vehicles['leader'].v, fl_dv, lf_dx]
        # print('lf_dx: ', lf_dx)
        # print('mf_dx: ', mf_dx)
        merger_feature = [self.vehicles['sdv'].v, fm_dv, mf_dx, self.vehicles['sdv'].y_lane]
        o_t = [self.vehicles['follower'].v]+leader_feature+merger_feature+[follower_action]
        return o_t

    def render(self, model_type=None):
        if self.viewer is None:
            self.viewer = Viewer(model_type, self.config)
            # self.viewer.PAUSE_CONTROL = PAUSE_CONTROL
        self.viewer.elapsed_time = self.elapsed_time
        self.viewer.update_plots(self.vehicles)
