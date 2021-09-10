import matplotlib.pyplot as plt
import numpy as np

class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(13, 3))
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
                print('#############  ', vehicle.id, '  ##############')
                print('My neighbours: ')
                for key, neighbour in vehicle.neighbours.items():
                    if neighbour:
                        print(key+': ', neighbour.id)
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                [vehicle.glob_y, neighbour.glob_y], linestyle='-',
                                    color='black', linewidth=1, alpha=0.3)
                    else:
                        print(key+': ', None)


                # print('target_lane: ', vehicle.target_lane)
                print('ego_decision: ', vehicle.lane_decision)
                print('ego_lane_id: ', vehicle.lane_id)
                print('lane_y: ', round(vehicle.lane_y, 2))
                print('ego_act: ', vehicle.act_long)
                print('steps_since_lc_initiation: ', vehicle.steps_since_lc_initiation)
                # print('lane_y: ', vehicle.lane_y)
                print('driver_params: ', vehicle.driver_params)

                # print('glob_x: ', vehicle.glob_x)
                # print('glob_y: ', vehicle.glob_y)
                # print('lane_y: ', vehicle.lane_y)
                print('###########################')

                # print('lane_decision: ', vehicle.lane_decision)
                # print('neighbour_f: ', vehicle.neighbours['f'].id)
                # print('true_act: ', vehicle.actions)
                # ideal_ = vehicle.idm_action(vehicle.observe(vehicle, vehicle.neighbours['f']))
                # print('ideal_act: ', ideal_)
                # print('obs: ', vehicle.observe(vehicle, vehicle.neighbours['f']))
                # print('delta_x: ', vehicle.neighbours['f'].glob_x - vehicle.glob_x)
                # print('###########################')

            if 'att' in vehicle.neighbours:
                neighbour = vehicle.neighbours['att']
                if neighbour:
                    line_1 = [vehicle.glob_y, neighbour.glob_y+.6]
                    line_2 = [vehicle.glob_y, neighbour.glob_y-.6]
                    ax.fill_between([vehicle.glob_x, neighbour.glob_x+1], \
                                        line_1, line_2, alpha=0.3, color='grey')


            if vehicle.lane_decision == 'move_left':
                ax.scatter(vehicle.glob_x-2, vehicle.glob_y+.7,
                                s=50, marker="*", color='red', edgecolors='black')
            elif vehicle.lane_decision == 'move_right':
                ax.scatter(vehicle.glob_x-2, vehicle.glob_y-.7,
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

class ViewerMC(Viewer):
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(20, 4))
        self.env_ax = self.fig.add_subplot(111)
        self.focus_on_this_vehicle = None
        self.fig = plt.figure(figsize=(5, 14))
        self.act_ax = self.fig.add_subplot(411)
        self.speed_ax = self.fig.add_subplot(412)
        self.att_ax = self.fig.add_subplot(413)
        self.desvel_ax = self.fig.add_subplot(414)

    def draw_vehicles(self, ax, vehicles, env_type):
        # vehicles = lisvehicles.values())

        # xs_idm_mobil = [veh.glob_x for veh in vehicles if veh.capability == 'IDMMOBIL']
        # ys_idm_mobil = [veh.glob_y for veh in vehicles if veh.capability == 'IDMMOBIL']
        glob_xs = [veh.glob_x for veh in vehicles]
        glob_ys = [veh.glob_y for veh in vehicles]
        annotation_mark_1 = [veh.id for veh in vehicles]
        annotation_mark_2 = [round(veh.speed, 2) for veh in vehicles]
        for i in range(len(annotation_mark_1)):
            ax.annotate(annotation_mark_1[i], (glob_xs[i], glob_ys[i]+1))
            ax.annotate(annotation_mark_2[i], (glob_xs[i], glob_ys[i]-1))


        if env_type == 'real':
            color_shade = [veh.driver_params['aggressiveness'] for veh in vehicles]
            ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        c=color_shade, cmap='rainbow')
        else:
            ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        color='grey', alpha=0.5, edgecolors='black')

        for vehicle in vehicles:
            att_veh = vehicle.neighbours['att']
            f_veh = vehicle.neighbours['f']

            if vehicle.id == self.focus_on_this_vehicle:
                print('#############  ', vehicle.id, env_type, '  ##############')
                print('My neighbours: ')
                for key, neighbour in vehicle.neighbours.items():
                    if neighbour:
                        print(key+': ', neighbour.id)
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                [vehicle.glob_y, neighbour.glob_y], linestyle='-',
                                    color='black', linewidth=1, alpha=0.3)

                    else:
                        print(key+': ', None)


                # print('target_lane: ', vehicle.target_lane)
                print('ego_decision: ', vehicle.lane_decision)
                print('ego_lane_id: ', vehicle.lane_id)
                print('lane_y: ', round(vehicle.lane_y, 2))
                print('ego_act: ', vehicle.act_long)
                print('steps_since_lc_initiation: ', vehicle.steps_since_lc_initiation)
                # print('lane_y: ', vehicle.lane_y)
                print('driver_params: ', vehicle.driver_params)

                # print('glob_x: ', vehicle.glob_x)
                # print('glob_y: ', vehicle.glob_y)
                # print('lane_y: ', vehicle.lane_y)
                print('###########################')

    def draw_highway(self, ax, vehicles, env_type):
        self.draw_road(ax)
        self.draw_vehicles(ax, vehicles, env_type)
        # self.draw_attention_line(ax, vehicles)

    def render(self, real_vehicles, ima_vehicles):
        self.env_ax.clear()

        self.draw_highway(self.env_ax, real_vehicles, 'real')
        if ima_vehicles:
            self.draw_highway(self.env_ax, ima_vehicles, 'imagined')


        plt.subplots_adjust(hspace = .001)
        plt.pause(1e-10)

    def info_plot(self, real_mc_log, ima_mc_log):
        if not self.focus_on_this_vehicle or self.focus_on_this_vehicle not in ima_mc_log:
            return
        self.act_ax.clear()
        self.speed_ax.clear()
        self.att_ax.clear()
        self.desvel_ax.clear()

        veh_id = self.focus_on_this_vehicle
        seq_len = len(real_mc_log[veh_id]['act'])
        x_range = range(seq_len)

        self.act_ax.plot(x_range, real_mc_log[veh_id]['act'], label=veh_id)
        self.speed_ax.plot(x_range, real_mc_log[veh_id]['speed'], label=veh_id)
        self.att_ax.plot(x_range, real_mc_log[veh_id]['att'])
        self.desvel_ax.plot(x_range, real_mc_log[veh_id]['desvel'])

        self.act_ax.plot(x_range, ima_mc_log[veh_id]['act'], linestyle='--')
        self.speed_ax.plot(x_range, ima_mc_log[veh_id]['speed'], linestyle='--')
        self.att_ax.plot(x_range, ima_mc_log[veh_id]['att'], linestyle='--')
        self.att_ax.plot(x_range, ima_mc_log[veh_id]['m_veh_exists'], color='purple')
        self.desvel_ax.plot(x_range, ima_mc_log[veh_id]['desvel'], linestyle='--')

        # self.act_ax.legend(['true', 'pred'])
        # self.att_ax.legend(['true', 'pred'])
        # self.desvel_ax.legend(['true', 'pred'])
        self.act_ax.set_title('action')
        self.speed_ax.set_title('speed')
        self.att_ax.set_title('attention')
        self.desvel_ax.set_title('desvel')

        major_tick = np.arange(35, len(real_mc_log[veh_id]['act']), 30)
        for axis in [self.act_ax, self.speed_ax, self.att_ax, self.desvel_ax]:
            axis.set_xticks(major_tick)
            axis.grid(axis='x')
        self.act_ax.legend('vehicle ' + str(veh_id))
