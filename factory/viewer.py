import matplotlib.pyplot as plt

class Viewer():
    def __init__(self, env_config):
        self.env_config  = env_config
        self.fig = plt.figure(figsize=(10, 3))
        self.env_ax = self.fig.add_subplot(211)
        self.tree_info = None
        self.belief_info = None
        self.decision_counts = None
        #TODO: option to record video

    def draw_road(self, ax, percept_origin, env_clock):
        lane_cor = self.env_config['lane_width']*self.env_config['lane_count']
        ax.hlines(0, 0, self.env_config['lane_length'], colors='k', linestyles='solid')
        ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                                                    colors='k', linestyles='solid')

        if self.env_config['lane_count'] > 1:
            lane_cor = self.env_config['lane_width']
            for lane in range(self.env_config['lane_count']-1):
                ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                                                        colors='k', linestyles='--')
                lane_cor += self.env_config['lane_width']

        if percept_origin < self.env_config['percept_range']:
            ax.set_xlim(0, self.env_config['percept_range']*2)
        else:
            ax.set_xlim(percept_origin - self.env_config['percept_range'],
                                percept_origin + self.env_config['percept_range'])

        ax.set_yticks([])
        # plt.show()
        plt.title('Elapsed time: '+str(round(env_clock, 1)))

    def draw_vehicles(self, ax, vehicles):
        xs = [veh.x for veh in vehicles if veh.id != 'neural']
        ys = [veh.y for veh in vehicles if veh.id != 'neural']
        for veh in vehicles:
            vehicle_color = 'grey'
            edgecolors = 'black'

            if veh.id == 'neural':
                vehicle_color = 'none'

                if veh.control_type == 'neural':
                    edgecolors = 'green'
                if veh.control_type == 'normal_idm':
                    edgecolors = 'orange'
                if veh.control_type == 'timid_idm':
                    edgecolors = 'yellow'
                if veh.control_type == 'aggressive_idm':
                    edgecolors = 'red'

            if veh.id == 'normal_idm':
                vehicle_color = 'orange'
            if veh.id == 'timid_idm':
                vehicle_color = 'yellow'
            if veh.id == 'aggressive_idm':
                vehicle_color = 'red'

            ax.scatter(veh.x, veh.y, s=100, marker=">", \
                                            facecolors=vehicle_color, edgecolors=edgecolors)

            ax.annotate(round(veh.v, 1), (veh.x, veh.y+0.1))
            # ax.annotate(round(veh.v, 1), (veh.x, veh.y+0.1))

    def draw_env(self, ax, vehicles, env_clock):
        ax.clear()
        self.draw_road(ax, percept_origin = vehicles[0].x, env_clock=env_clock)
        self.draw_vehicles(ax, vehicles)

    def update_plots(self, vehicles, env_clock):
        self.draw_env(self.env_ax, vehicles, env_clock)
        plt.pause(0.00000000000000000000001)
        # plt.show()
