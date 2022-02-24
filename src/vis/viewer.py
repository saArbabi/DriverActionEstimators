import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(13, 2))
        self.env_ax = self.fig.add_subplot(111)
        self.focus_on_this_vehicle = None
        self.merge_box = [Rectangle((config['merge_lane_start'], 0), \
                            config['merge_lane_length'], config['lane_width'])]

    def draw_road(self, ax):
        lane_cor = self.config['lane_width']*self.config['lanes_n']
        ax.hlines(0, 0, self.config['lane_length'], colors='k', linestyles='solid')
        ax.vlines(self.config['merge_zone_end'], 0, self.config['lane_width'], \
                                                    colors='k', linestyles='solid')

        ax.vlines(self.config['ramp_exit_start'], 0, self.config['lane_width'], \
                                                    colors='k', linestyles='solid')

        ax.hlines(lane_cor, 0, self.config['lane_length'],
                                                    colors='k', linestyles='solid')

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(self.merge_box, hatch='/', alpha=0.2)
        ax.add_collection(pc)
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
                print('ego_lane_id_target: ', vehicle.target_lane)
                print('glob_y: ', vehicle.glob_y)
                print('glob_x: ', round(vehicle.glob_x, 2))
                print('ego_act: ', vehicle.act_long_c)
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
                if vehicle.neighbours['f']:
                    print('delta_x: ', vehicle.neighbours['f'].glob_x - vehicle.glob_x)
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
        self.fig = plt.figure(figsize=(13, 1))
        self.env_ax = self.fig.add_subplot(111)
        self.focus_on_this_vehicle = None
        self.merge_box = [Rectangle((config['merge_lane_start'], 0), \
                            config['merge_lane_length'], config['lane_width'])]

    def draw_vehicles(self, ax, vehicles, env_type):
        # vehicles = lisvehicles.values())

        # xs_idm_mobil = [veh.glob_x for veh in vehicles if veh.capability == 'IDMMOBIL']
        # ys_idm_mobil = [veh.glob_y for veh in vehicles if veh.capability == 'IDMMOBIL']
        glob_xs = [veh.glob_x for veh in vehicles]
        glob_ys = [veh.glob_y for veh in vehicles]

        if env_type == 'real':
            color_shade = [veh.driver_params['aggressiveness'] for veh in vehicles]
            ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        c=color_shade, cmap='rainbow')
        else:
            ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        color='grey', alpha=0.5, edgecolors='black')

        for vehicle in vehicles:
            if vehicle.vehicle_type == 'neural':
                ann_color = 'green'
            else:
                ann_color = 'black'

            ax.annotate(vehicle.id, (vehicle.glob_x, vehicle.glob_y+1), color=ann_color)
            ax.annotate(round(vehicle.speed, 2), (vehicle.glob_x, vehicle.glob_y-1), color=ann_color)

            att_veh = vehicle.neighbours['att']
            f_veh = vehicle.neighbours['f']

            if self.focus_on_this_vehicle and (vehicle.id == self.focus_on_this_vehicle or \
                                str(vehicle.id) == 'neur_'+str(self.focus_on_this_vehicle)):
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
                print('ego_act: ', vehicle.act_long_c)
                print('steps_since_lc_initiation: ', vehicle.steps_since_lc_initiation)
                # print('lane_y: ', vehicle.lane_y)
                print('driver_params: ', vehicle.driver_params)

                # print('glob_x: ', vehicle.glob_x)
                # print('glob_y: ', vehicle.glob_y)
                # print('lane_y: ', vehicle.lane_y)
                if vehicle.neighbours['f']:
                    print('delta_x: ', vehicle.neighbours['f'].glob_x - vehicle.glob_x)
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

        if not hasattr(self, 'act_ax'):
            self.fig = plt.figure(figsize=(10, 8))
            self.act_ax = self.fig.add_subplot(511)
            self.speed_ax = self.fig.add_subplot(512)
            self.att_ax = self.fig.add_subplot(513)
            self.desvel_ax = self.fig.add_subplot(514)
            self.desparam_ax = self.fig.add_subplot(515)

        self.act_ax.clear()
        self.speed_ax.clear()
        self.att_ax.clear()
        self.desvel_ax.clear()
        self.desparam_ax.clear()

        veh_id = self.focus_on_this_vehicle
        tace_len = len(real_mc_log[veh_id]['act_long_c'])
        x_range = range(tace_len)
        colors = cm.rainbow(np.linspace(0, 1, 5))

        self.act_ax.plot(x_range, real_mc_log[veh_id]['act_long_c'], label=veh_id)
        self.speed_ax.plot(x_range, real_mc_log[veh_id]['speed'], label=veh_id)
        self.att_ax.plot(x_range, real_mc_log[veh_id]['att'])
        color_i = 0
        for key in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
            color = colors[color_i]
            if key == 'desired_v':
                self.desvel_ax.plot(x_range, real_mc_log[veh_id][key], color=color)
            else:
                self.desparam_ax.plot(x_range, real_mc_log[veh_id][key], color=color)
            color_i += 1
        # Imagined vehicle
        self.act_ax.plot(x_range, ima_mc_log[veh_id]['act_long_c'], linestyle='--')
        self.speed_ax.plot(x_range, ima_mc_log[veh_id]['speed'], linestyle='--')
        speed_lim_min = min(ima_mc_log[veh_id]['speed']+real_mc_log[veh_id]['speed'])-1
        speed_lim_max = max(ima_mc_log[veh_id]['speed']+real_mc_log[veh_id]['speed'])+1
        self.speed_ax.set_ylim(speed_lim_min, speed_lim_max)

        self.att_ax.plot(x_range, ima_mc_log[veh_id]['att'], linestyle='--')
        self.att_ax.set_ylim(-0.1, 1.1)
        # self.att_ax.plot(x_range, ima_mc_log[veh_id]['m_veh_exists'], color='purple')

        color_i = 0
        for key in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
            color = colors[color_i]
            if key == 'desired_v':
                self.desvel_ax.plot(x_range, ima_mc_log[veh_id][key], linestyle='--', label=key, color=color)
            else:
                self.desparam_ax.plot(x_range, ima_mc_log[veh_id][key], linestyle='--', label=key, color=color)
            color_i += 1

        self.desparam_ax.legend(loc='upper left')

        # self.act_ax.legend(['true', 'pred'])
        # self.att_ax.legend(['true', 'pred'])
        # self.desvel_ax.legend(['true', 'pred'])
        self.act_ax.set_title('action')
        self.speed_ax.set_title('speed')
        self.att_ax.set_title('attention')
        self.desvel_ax.set_title('desvel')

        major_tick = list(range(29, tace_len, 30))+[tace_len-1]
        for axis in [self.act_ax, self.speed_ax, self.att_ax, self.desvel_ax, self.desparam_ax]:
            axis.set_xticks(major_tick)
            axis.grid(axis='x')
        self.act_ax.legend(['vehicle ' + str(veh_id)])
