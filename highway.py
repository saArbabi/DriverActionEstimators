# from factory.viewer import Viewer
import copy
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2020)


class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(20, 6))
        self.env_ax = self.fig.add_subplot(211)
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

        # if percept_origin < self.config['percept_range']:
        #     ax.set_xlim(0, self.config['percept_range']*2)
        # else:
        #     ax.set_xlim(percept_origin - self.config['percept_range'],
        #                         percept_origin + self.config['percept_range'])

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
                        print('delta_x :', delta_x)

            if 'f' in vehicle.neighbours:
                neighbour = vehicle.neighbours['f']
                if neighbour:
                    line_1 = [vehicle.glob_y, neighbour.glob_y+.5]
                    line_2 = [vehicle.glob_y, neighbour.glob_y-.5]
                    ax.fill_between([vehicle.glob_x, neighbour.glob_x+1], \
                                        line_1, line_2, alpha=0.4, color='grey')

            if vehicle.lane_decision == 'move_left':
                ax.scatter(vehicle.glob_x-7, vehicle.glob_y+self.config['lane_width']/2,
                                                    s=50, marker="*", color='red', edgecolors='black')
            elif vehicle.lane_decision == 'move_right':
                ax.scatter(vehicle.glob_x-7, vehicle.glob_y-self.config['lane_width']/2,
                                                    s=50, marker="*", color='red', edgecolors='black')


        color_shade = [veh.aggressiveness for veh in vehicles]
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

class Vehicle(object):
    STEP_SIZE = 0.1
    def __init__(self, id, lane_id, glob_x, speed):
        self.id = id
        self.lane_id = lane_id
        self.glob_x = glob_x
        # self.glob_y = glob_y
        self.speed = speed
        self.lane_y = 0
        self.lane_width = 3.7

    def observe(self):
        raise NotImplementedError

    def step(self, action):
        act_long, act_lat = action
        self.glob_x +=  self.speed * self.STEP_SIZE \
                                    + 0.5 * act_long * self.STEP_SIZE **2
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

class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        # self.capability = 'IDM'
        self.lane_id = lane_id
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.aggressiveness = aggressiveness # in range [0, 1]
        self.neighbours = {}
        self.percept_range = 200 #m
        self.lane_width = 3.8

        self.lateral_actions = {'move_left':0.7,
                                'move_right':-0.7,
                                'keep_lane':0
                                }
        self.set_idm_params(aggressiveness)


    def set_idm_params(self, aggressiveness):
        Parameter_range = {'most_aggressive': {
                                        'desired_v':30, # m/s
                                        'desired_tgap':1, # s
                                        'min_jamx':0, # m
                                        'max_act':2, # m/s^2
                                        'min_act':3, # m/s^2
                                        'politeness':0,
                                        'safe_braking':-3,
                                        'act_threshold':0
                                        },
                         'least_aggressvie': {
                                        'desired_v':19.4, # m/s
                                        'desired_tgap':2, # s
                                        'min_jamx':4, # m
                                        'max_act':0.8, # m/s^2
                                        'min_act':1, # m/s^2
                                        'politeness':1,
                                        'safe_braking':-1,
                                        'act_threshold':0.2
                                         }}

        self.driver_params = {}
        # IDM params
        self.driver_params['desired_v'] = self.get_idm_param(Parameter_range, 'desired_v')
        self.driver_params['desired_tgap'] = self.get_idm_param(Parameter_range, 'desired_tgap')
        self.driver_params['min_jamx'] = self.get_idm_param(Parameter_range, 'min_jamx')
        self.driver_params['max_act'] = self.get_idm_param(Parameter_range, 'max_act')
        self.driver_params['min_act'] = self.get_idm_param(Parameter_range, 'min_act')
        # MOBIL params
        self.driver_params['politeness'] = self.get_idm_param(Parameter_range, 'politeness')
        self.driver_params['safe_braking'] = self.get_idm_param(Parameter_range, 'safe_braking')
        self.driver_params['act_threshold'] = self.get_idm_param(Parameter_range, 'act_threshold')
        self.driver_params['attention_switch'] =  0.5*self.lane_width*self.aggressiveness

        if aggressiveness == None:
            raise ValueError('No aggressiveness specified!')

        # self.driver_params['desired_v'] += np.random.normal(0, 1)

    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act', 'safe_braking']:
            # the larger the param, the more aggressive the driver
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.aggressiveness*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness', 'act_threshold']:
            # the larger the param, the more timid the driver
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  max_value - self.aggressiveness*(max_value-min_value)


    def get_desired_gap(self, delta_v):
        gap = self.driver_params['min_jamx'] + \
                                            self.driver_params['desired_tgap']*\
                                            self.speed+(self.speed*delta_v)/ \
                                            (2*np.sqrt(self.driver_params['max_act']*\
                                            self.driver_params['min_act']))
        return max([0, gap])

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 1000]
        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        return [delta_v, delta_x]

    def idm_action(self, obs):
        delta_v, delta_x = obs
        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.driver_params['max_act']*(1-(self.speed/self.driver_params['desired_v'])**4-\
                                            (desired_gap/(delta_x+1e-5))**2)

        return act_long

    def check_reservations(self, target_lane, reservations):
        """To ensure two cars do not simultaneously move into the same lane.
        """
        if not reservations:
            return 'pass'
        else:
            for reserved in reservations.values():
                reserved_lane, max_glob_x, min_glob_x = reserved
                if target_lane != reserved_lane or self.glob_x < min_glob_x \
                                                            or self.glob_x > max_glob_x:

                    return 'pass'
                else:
                    return 'fail'

    def check_neighbours(self, neighbours):
        """To ensure neighbours keep lane while merger is performing lane change
        """
        for vehicle in neighbours.values():
            if vehicle and vehicle.lane_decision != 'keep_lane':
                return 'fail'
        return 'pass'

    def mobil_condition(self, action_gains):
        """To decide if changing lane is worthwhile/
        """
        ego_gain, new_follower_gain, old_follower_gain = action_gains
        lc_condition = ego_gain+self.driver_params['politeness']*(new_follower_gain+\
                                                                old_follower_gain )
        return lc_condition

    def check_attention(self, vehicle, delta_x, delta_xs):
        if vehicle.target_lane == self.lane_id and \
                abs(vehicle.lane_y) > self.driver_params['attention_switch'] and \
                delta_x < delta_xs[-1]:
            return True
        return False

    def find_neighbours(self, vehicles):
        """returns list of current neighbouring vehicles.
        """
        neighbours = {}
        delta_xs_f, delta_xs_fl, delta_xs_rl, delta_xs_r, \
                        delta_xs_rr, delta_xs_fr = ([self.percept_range] for i in range(6))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
                        candidate_rr, candidate_fr = (None for i in range(6))

        for vehicle in vehicles:
            if vehicle.id != self.id:
                delta_x = vehicle.glob_x-self.glob_x
                if vehicle.lane_id in [self.lane_id, self.lane_id+1, self.lane_id-1] and \
                                  abs(delta_x) < self.percept_range:

                    if self.lane_decision != 'keep_lane':
                        if self.lane_decision == 'move_left' and \
                                    vehicle.lane_id == self.target_lane and delta_x >= 0:
                            if delta_x < delta_xs_f[-1]:
                                delta_xs_f.append(delta_x)
                                candidate_f = vehicle
                                continue

                        elif self.lane_decision == 'move_right' and \
                                    vehicle.lane_id == self.target_lane and delta_x >= 0:
                            if delta_x < delta_xs_f[-1]:
                                delta_xs_f.append(delta_x)
                                candidate_f = vehicle
                                continue

                    elif self.lane_decision == 'keep_lane':
                        if delta_x >= 0:
                            if vehicle.lane_id == self.lane_id+1:
                                # right lane
                                if self.check_attention(vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fr[-1]:
                                    delta_xs_fr.append(delta_x)
                                    candidate_fr = vehicle

                            elif vehicle.lane_id == self.lane_id-1:
                                # left lane
                                if self.check_attention(vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fl[-1]:
                                    delta_xs_fl.append(delta_x)
                                    candidate_fl = vehicle

                            elif vehicle.lane_id == self.lane_id:
                                # same lane
                                if delta_x < delta_xs_f[-1]:
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                        elif delta_x < 0:
                            delta_x = abs(delta_x)
                            if vehicle.lane_id == self.lane_id+1:
                                # right lane
                                if delta_x < delta_xs_rr[-1]:
                                    delta_xs_rr.append(delta_x)
                                    candidate_rr = vehicle
                            elif vehicle.lane_id == self.lane_id-1:
                                # left lane
                                if delta_x < delta_xs_rl[-1]:
                                    delta_xs_rl.append(delta_x)
                                    candidate_rl = vehicle
                            elif vehicle.lane_id == self.lane_id:
                                # same lane
                                if delta_x < delta_xs_r[-1]:
                                    delta_xs_r.append(delta_x)
                                    candidate_r = vehicle

        neighbours['f'] = candidate_f
        neighbours['fl'] = candidate_fl
        neighbours['rl'] = candidate_rl
        neighbours['r'] = candidate_r
        neighbours['rr'] = candidate_rr
        neighbours['fr'] = candidate_fr

        return neighbours

    def act(self, vehicles, reservations):
        neighbours = self.find_neighbours(vehicles)
        self.neighbours = neighbours
        act_long = self.idm_action(self.observe(self, neighbours['f']))
        check_1 = self.check_neighbours(neighbours)

        if check_1 == 'fail' or self.glob_x < 100:
            pass

        elif self.lane_decision == 'move_left':
            if self.lane_id == self.target_lane :
                if self.lane_y >= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'move_right':
            if self.lane_id == self.target_lane :
                if self.lane_y <= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'keep_lane':
            lc_left_condition = 0
            lc_right_condition = 0

            act_rl_lc = self.idm_action(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_action(self.observe(neighbours['rr'], self))
            act_r_lc = self.idm_action(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_action(self.observe(neighbours['r'], self))
            old_follower_gain = act_r_lc-act_r_lk
            if self.id == 19:
                print('old_follower_gain: ', old_follower_gain)
                print('act_rr_lc: ', act_rr_lc)
                print('obs: ', self.observe(neighbours['rr'], self))

            if self.lane_id > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_action(self.observe(neighbours['rl'], neighbours['fl']))
                act_ego_lc_l = self.idm_action(self.observe(self, neighbours['fl']))
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])
                print('##### LEFT ######', self.id)
                print([ego_gain, new_follower_gain, old_follower_gain])
                print(['ego_gain', 'new_follower_gain', 'old_follower_gain'])
            if self.lane_id < self.lanes_n and \
                                                self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_action(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_action(self.observe(neighbours['rr'], neighbours['fr']))

                ego_gain = act_ego_lc_r-act_long
                new_follower_gain = act_rr_lc-act_rr_lk
                lc_right_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

                print('##### RIGHT ######', self.id)
                print([ego_gain, new_follower_gain, old_follower_gain])
                print(['ego_gain', 'new_follower_gain', 'old_follower_gain'])
            if max([lc_left_condition, lc_right_condition]) > self.driver_params['act_threshold']:
                if lc_left_condition > lc_right_condition:
                    target_lane = self.target_lane - 1
                    check_2 = self.check_reservations(target_lane, reservations)
                    if check_2 == 'pass':
                        self.lane_decision = 'move_left'
                        self.target_lane -= 1
                        return [act_ego_lc_l, self.lateral_actions[self.lane_decision]]

                elif lc_left_condition < lc_right_condition:
                    target_lane = self.target_lane + 1
                    check_2 = self.check_reservations(target_lane, reservations)
                    if check_2 == 'pass':
                        self.lane_decision = 'move_right'
                        self.target_lane += 1
                        return [act_ego_lc_r, self.lateral_actions[self.lane_decision]]

        return [act_long, self.lateral_actions[self.lane_decision]]


class VehicleHandler:
    def __init__(self, config=None):
        # self.entry_vehicles =
        # self.queuing_vehicles =
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.next_vehicle_id = 0
        self.lane_width = 3.7
        self.target_traffic_density = 1. # vehicle count per 100 meters
        self.traffic_density = 0
        self.reservations = {}

    def place_vehicle(self, vehicle_count, elapsed_time):
        """Creates a new IDM vehicle.
        """
        speed = 20
        new_vehicle_entries = []

        self.traffic_density = (vehicle_count/(self.lanes_n*self.lane_length))*100
        if self.traffic_density < self.target_traffic_density:
            if elapsed_time % 20 == 0:
                for lane_id in range(1, self.lanes_n+1):
                    coin_flip = np.random.random()
                    if coin_flip < 0.3:
                        id = self.next_vehicle_id
                        glob_x =  0

                        # aggressiveness = np.random.uniform(0, 1)
                        aggressiveness = np.random.choice([0, 0.5, 1])
                        new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)
                        new_vehicle.lanes_n = self.lanes_n
                        new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
                        new_vehicle_entries.append(new_vehicle)
                        self.next_vehicle_id += 1

        return new_vehicle_entries

    def get_vehicle_state(self, vehicle):
        pass

    def update_reservations(self, vehicle):
        if vehicle.id in self.reservations and vehicle.lane_decision == 'keep_lane':
            del self.reservations[vehicle.id]
        elif vehicle.lane_decision != 'keep_lane':
            max_glob_x, min_glob_x = round(vehicle.glob_x) + 100, round(vehicle.glob_x) - 100
            self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]
            print(self.reservations)

class Road:
    def __init__(self, lane_length, lane_width, lanes_n):
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.lanes_n = lanes_n

class Env:
    def __init__(self, config):
        self.config = config
        self.elapsed_time = 0
        self.handler = VehicleHandler(config)
        self.sdv = None
        self.vehicles = []
        self.initiate_environment()
        # self.vehicles = []

    def initiate_environment(self):
        self.lane_length = self.config['lane_length']

    def step(self, action=None):
        """ steps the environment forward in time.
        """
        vehicles = [] # list for vehicles with new states
        for vehicle_i in self.vehicles:
            if vehicle_i.glob_x > self.lane_length:
                # consider vehicle gone
                if vehicle_i.id in self.handler.reservations:
                    del self.handler.reservations[vehicle_i.id]
                continue

            action = vehicle_i.act(self.vehicles, self.handler.reservations)
            self.handler.update_reservations(vehicle_i)
            vehicle_ii = copy.copy(vehicle_i)
            vehicle_ii.step(action)
            vehicles.append(vehicle_ii)


        self.vehicles = vehicles
        new_vehicle_entries = self.handler.place_vehicle(len(vehicles), self.elapsed_time)
        if new_vehicle_entries:
            self.vehicles.extend(new_vehicle_entries)

        self.elapsed_time += 1

    def get_feature_vector(self, obs):
        return [item for sublist in obs.values() for item in sublist]


config = {'lanes_n':2,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
viewer = Viewer(config)
def main():
    # for i in range(100):
    step_i = 0
    while True:
        decision = input()
        if decision == 'n':
            sys.exit()
        try:
            viewer.focus_on_this_vehicle = int(decision)
        except:
            pass


        env.step()
        viewer.render(env.vehicles)
        print(step_i, ' ####### step #######')
        step_i += 1

if __name__=='__main__':
    main()

# %%
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def animation_frame(i):
        viewer.render(env.vehicles)
        env.step()
        # return line,

    animation = FuncAnimation(viewer.fig, func=animation_frame, frames=range(1800), interval=1000)


    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=25, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('sim_example.mp4', writer, dpi=500)





# get_animation()
# plt.show()
# %%
driver_params = {
                'desired_v':19.4, # m/s
                'desired_tgap':2, # s
                'min_jamx':4, # m
                'max_act':0.8, # m/s^2
                'min_act':1, # m/s^2
                }
#
speed = 19.4
def get_desired_gap(delta_v):
    gap = driver_params['min_jamx'] + driver_params['desired_tgap']*speed+(speed*delta_v)/ \
                (2*np.sqrt(driver_params['max_act']*driver_params['min_act']))
    print('gap: ', gap)
    return gap

    # return max([0, gap])

def idm_action(obs):
    delta_v, delta_x = obs
    desired_gap = get_desired_gap(delta_v)
    act_long = driver_params['max_act']*(1-(speed/driver_params['desired_v'])**4-\
                                        (desired_gap/(delta_x+1e-5))**2)
    return act_long


obs = [0.3232128791548199, 38.29334944722247]
# obs = [0.273349751276303, 55.71618653304381]
# obs = [0, 1000]
idm_action(obs)
# max([0, -10])
# plt.scatter([1,1], [1,1], marker='>', s=5000)
# plt.grid()
# old_follower_gain:  0.8343400627099891
# act_rr_lc:  -1.1572396660914637
# obs:
# 865  ####### step #######
