import matplotlib.pyplot as plt

class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(15, 2))
        self.env_ax = self.fig.add_subplot(111)
        self.focus_on_this_vehicle = None
        # self.att_ax = self.fig.add_subplot(212)
        # self.model_type = model_type
        # self.true_attention_scores = []
        # self.attention_values = None
        # self.elapsed_steps = []

    def draw_road(self, ax):
        lane_cor = self.config['lane_width']*self.config['lanes_n']
        ax.hlines(0, 0, self.config['lane_length'], colors='k', linestyles='solid')
        ax.hlines(lane_cor, 0, self.config['lane_length'],
                                                    colors='k', linestyles='solid')

        if self.config['lanes_n'] > 1:
            lane_cor = self.config['lane_width']
            for lane in range(self.config['lanes_n']-1):
                ax.hlines(lane_cor, 0, self.config['lane_length'],
                                        colors='k', linestyles='--')
                lane_cor += self.config['lane_width']

        ax.set_xlim(0, self.config['lane_length'])

        # if percept_origin < self.config['perception_range']:
        #     ax.set_xlim(0, self.config['perception_range']*2)
        # else:
        #     ax.set_xlim(percept_origin - self.config['perception_range'],
        #                         percept_origin + self.config['perception_range'])

        ax.set_yticks([])
        # ax.set_title('#Elapsed steps: '+str(self.env_clock/10)+\
        # 's  #model: '+self.model_type)

    def draw_vehicles(self, ax, vehicles):
        # vehicles = lisvehicles.values())

        # xs_idm_mobil = [veh.glob_x for veh in vehicles if veh.capability == 'IDMMOBIL']
        # ys_idm_mobil = [veh.glob_y for veh in vehicles if veh.capability == 'IDMMOBIL']
        glob_xs = [veh.glob_x for veh in vehicles]
        glob_ys = [veh.glob_y for veh in vehicles]

        annotation_mark_1 = [veh.id for veh in vehicles]
        # annotation_mark = [veh.lane_decision for veh in vehicles]
        annotation_mark_2 = [round(veh.speed, 1) for veh in vehicles]
        # annotation_mark_2 = [round(veh.lane_y, 2) for veh in vehicles]
        for i in range(len(annotation_mark_1)):
            ax.annotate(annotation_mark_1[i], (glob_xs[i], glob_ys[i]+1))
            ax.annotate(annotation_mark_2[i], (glob_xs[i], glob_ys[i]-1))



        for vehicle in vehicles:
            if vehicle.id == self.focus_on_this_vehicle:
                for key, neighbour in vehicle.neighbours.items():
                    if neighbour:
                        # if key == 'f':
                        # if vehicle.lane_decision != 'keep_lane':
                        #     ax.plot([vehicle.glob_x, neighbour.glob_x], \
                        #             [vehicle.glob_y, neighbour.glob_y], linestyle=':',
                        #                 color='red', linewidth=5, alpha=0.5)
                        # else:
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                [vehicle.glob_y, neighbour.glob_y], linestyle='-',
                                    color='grey', linewidth=5, alpha=0.3)

                        delta_x = round(neighbour.glob_x-vehicle.glob_x, 2)
                        pos_x = vehicle.glob_x + (delta_x)/2
                        delta_y = round(neighbour.glob_y-vehicle.glob_y, 2)
                        pos_y = vehicle.glob_y + (delta_y)/2
                        # ax.annotate(delta_x, (pos_x, pos_y+5))
                        ax.annotate(key, (pos_x, pos_y))
                        # print(delta_x)
                        # print('delta_x :', delta_x)
                print('lane_id: ', vehicle.lane_id)
                print('target_lane: ', vehicle.target_lane)
                print('glob_x: ', vehicle.glob_x)
                print('glob_y: ', vehicle.glob_y)
                print('lane_y: ', vehicle.lane_y)

            if 'f' in vehicle.neighbours:
                neighbour = vehicle.neighbours['f']
                if neighbour:
                    line_1 = [vehicle.glob_y, neighbour.glob_y+.6]
                    line_2 = [vehicle.glob_y, neighbour.glob_y-.6]
                    ax.fill_between([vehicle.glob_x, neighbour.glob_x+1], \
                                        line_1, line_2, alpha=0.3, color='grey')


            if vehicle.lane_decision == 'move_left':
                ax.scatter(vehicle.glob_x-7, vehicle.glob_y+.7,
                                s=50, marker="*", color='red', edgecolors='black')
            elif vehicle.lane_decision == 'move_right':
                ax.scatter(vehicle.glob_x-7, vehicle.glob_y-.7,
                                s=50, marker="*", color='red', edgecolors='black')


        color_shade = [veh.driver_params['aggressiveness'] for veh in vehicles]
        ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        c=color_shade, cmap='rainbow')

    def draw_attention_line(self, ax, vehicles):
        x1 = vehicles[0].x
        y1 = vehicles[0].y
        x2 = vehicles[0].attend_veh.glob_x
        y2 = vehicles[0].attend_veh.glob_y
        ax.plot([x1, x2],[y1, y2])
        ax.scatter([x1, x2],[y1, y2], s=10)

    def draw_highway(self, ax, vehicles):
        ax.clear()
        self.draw_road(ax)
        self.draw_vehicles(ax, vehicles)
        # self.draw_attention_line(ax, vehicles)

    def draw_att_plot(self, ax, vehicles):
        ax.clear()
        # self.true_attention_scores.append(vehicles[0].attention)
        # self.pred_attention_scores.append(vehicles[0].attention_score)
        # self.elapsed_steps.append(self.env_clock)
        #
        if self.attention_values:
            for trace in self.attention_values:
                ax.plot(trace[0:self.elapsed_time])
            ax.set_ylim(-0.1, 1.1)
        # ax.plot(self.elapsed_steps, self.pred_attention_scores)
        # ax.legend(['True attention', 'Predicted attention'])

    def update_plots(self, vehicles):
        self.draw_highway(self.env_ax, vehicles)
        # self.draw_att_plot(self.att_ax, vehicles)
        plt.pause(1e-10)
        # plt.show()
    def render(self, vehicles):
        self.update_plots(vehicles)
