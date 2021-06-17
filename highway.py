# from factory.viewer import Viewer
import copy
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2020)


class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(15, 2))
        self.env_ax = self.fig.add_subplot(211)
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


        # for veh in vehicles:
        # vehicle_color = 'grey'
        # edgecolors = 'black'
            #
            # if veh.id == 'neural':
            #     vehicle_color = 'none'
            #     # print(veh.glob_x)
            #     # print(veh.glob_y.shape)
            #
            #     # if veh.control_type == 'neural':
            #     #     edgecolors = 'green'
            #     #     ax.annotate('e', (veh.glob_x, veh.glob_y+0.3))
            #
            # if veh.id == 'normal_idm':
            #     vehicle_color = 'orange'
            # if veh.id == 'timid_idm':
            #     vehicle_color = 'yellow'
            # if veh.id == 'aggressive_idm':
            #     vehicle_color = 'red'

        color_shade = [veh.aggressiveness for veh in vehicles]
        ax.scatter(glob_xs, glob_ys, s=100, marker=">", \
                                        c=color_shade, cmap='rainbow')


        # ax.scatter(xs_idm_mobil, ys_idm_mobil, s=100, marker=">", \
        #                                 facecolors='red')

        # annotation_mark = [veh.aggressiveness for veh in vehicles]
        annotation_mark = [veh.id for veh in vehicles]
        # annotation_mark = [round(veh.speed, 1) for veh in vehicles]
        for i in range(len(annotation_mark)):
            ax.annotate(annotation_mark[i], (glob_xs[i], glob_ys[i]+0.2))
            # ax.annotate(round(veh.v, 1), (veh.glob_x, veh.glob_y+0.1))

        for vehicle in vehicles:
            if vehicle.lane_decision != 'keep_lane':
                color = 'red'
                if vehicle.lane_decision == 'move_left':
                    ax.scatter(vehicle.glob_x-7, vehicle.glob_y+self.config['lane_width']/2,
                                                        s=50, marker="*", color='red', edgecolors='black')
                elif vehicle.lane_decision == 'move_right':
                    ax.scatter(vehicle.glob_x-7, vehicle.glob_y-self.config['lane_width']/2,
                                                        s=50, marker="*", color='red', edgecolors='black')
                for neighbour in vehicle.neighbours.values():
                    if neighbour:
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                            [vehicle.glob_y, neighbour.glob_y], color=color)

            else:
                for neighbour in vehicle.neighbours.values():
                    if neighbour:
                        ax.plot([vehicle.glob_x, neighbour.glob_x], \
                                            [vehicle.glob_y, neighbour.glob_y], color='grey', alpha=0.4)


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
        plt.pause(1e-3)
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

    def act(self, obs=None):
        # pass
        return 0

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


    # def act(self):
    #     raise NotImplementedError
class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, aggressiveness=None):
        super().__init__(id, lane_id, glob_x, speed)
        # self.capability = 'IDM'
        self.lane_id = lane_id
        self.target_lane = lane_id
        self.lane_decision = 'keep_lane'
        self.max_lane_id = 2
        self.aggressiveness = aggressiveness # in range [0, 1]
        self.neighbours = {}

        self.lateral_actions = {'move_left':0.7,
                                'move_right':-0.7,
                                'keep_lane':0
                                }
        self.set_idm_params(aggressiveness)


    def set_idm_params(self, aggressiveness):
        normal_idm = {
                        'desired_v':25, # m/s
                        'desired_tgap':1.5, # s
                        'min_jamx':2, # m
                        'max_act':1.4, # m/s^2
                        'min_act':2, # m/s^2
                        }

        timid_idm = {
                        'desired_v':19.4, # m/s
                        'desired_tgap':2, # s
                        'min_jamx':4, # m
                        'max_act':0.8, # m/s^2
                        'min_act':1, # m/s^2
                        }

        aggressive_idm = {
                        'desired_v':30, # m/s
                        'desired_tgap':1, # s
                        'min_jamx':0, # m
                        'max_act':2, # m/s^2
                        'min_act':3, # m/s^2
                        }

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
        if not aggressiveness:
            raise ValueError('No aggressiveness specified!')

        # if aggressiveness == 'normal_idm':
        #     self.driver_params = normal_idm
        # if aggressiveness == 'timid_idm':
        #     self.driver_params = timid_idm
        # if aggressiveness == 'aggressive_idm':
        #     self.driver_params = aggressive_idm

        # self.driver_params['desired_v'] += np.random.normal(0, 1)

    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act', 'safe_braking']:
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.aggressiveness*(max_value-min_value)
        else:
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  min_value + self.aggressiveness*(max_value-min_value)

    def get_desired_gap(self, delta_v):
        gap = self.driver_params['min_jamx'] + self.driver_params['desired_tgap']*self.speed+(self.speed*delta_v)/ \
                    (2*np.sqrt(self.driver_params['max_act']*self.driver_params['min_act']))
        return gap

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
                                            (desired_gap/delta_x)**2)
        return act_long

    def check_reservations(self, reservations):
        """To ensure two cars do not simultaneously move into the same lane.
        """
        if not reservations:
            return 'pass'
        else:
            for reserved in reservations.values():
                reserved_lane, max_glob_x, min_glob_x = reserved
                if self.target_lane != reserved_lane or self.glob_x < min_glob_x \
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

    def act(self, neighbours, reservations):
        act_long = self.idm_action(self.observe(self, neighbours['f']))
        if self.lane_decision == 'move_left':
            if self.lane_id  == self.target_lane :
                act_long = self.idm_action(self.observe(self, neighbours['f']))
                if self.lane_y >= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

            elif self.lane_id > self.target_lane :
                act_long = self.idm_action(self.observe(self, neighbours['fl']))

        elif self.lane_decision == 'move_right':
            if self.lane_id  == self.target_lane :
                act_long = self.idm_action(self.observe(self, neighbours['f']))
                if self.lane_y <= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'
            elif self.lane_id < self.target_lane :
                act_long = self.idm_action(self.observe(self, neighbours['fr']))

        elif self.lane_decision == 'keep_lane':
            check_1 = self.check_reservations(reservations)
            check_2 = self.check_neighbours(neighbours)
            if check_1 == 'fail' or check_2 == 'fail':
                self.lane_decision = 'keep_lane'
                self.target_lane = self.lane_id
                return [act_long, self.lateral_actions[self.lane_decision]]

            lc_left_condition = 0
            lc_right_condition = 0

            # action for ego vehicle if ego is to keep lane
            act_rl_lc = self.idm_action(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_action(self.observe(neighbours['rr'], self))
            act_r_lc = self.idm_action(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_action(self.observe(neighbours['r'], self))
            old_follower_gain = act_r_lc-act_r_lk

            if self.lane_id  > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_action(self.observe(neighbours['rl'], neighbours['fl']))
                act_ego_lc_l = self.idm_action(self.observe(self, neighbours['fl']))
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            elif self.lane_id  < self.max_lane_id and \
                                                self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_action(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_action(self.observe(neighbours['rr'], neighbours['fr']))

                ego_gain = act_ego_lc_r-act_long
                new_follower_gain = act_rr_lc-act_rr_lk
                lc_right_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if lc_left_condition > self.driver_params['act_threshold']:
                self.target_lane  -= 1

            elif lc_right_condition > self.driver_params['act_threshold']:
                self.target_lane  += 1

            else:
                self.lane_decision = 'keep_lane'
                self.target_lane = self.lane_id
                return [act_long, self.lateral_actions[self.lane_decision]]

            if self.target_lane < self.lane_id:
                act_long = act_ego_lc_l
                self.lane_decision = 'move_left'
                print(lc_left_condition)
            elif self.target_lane < self.lane_id:
                act_long = act_ego_lc_r
                self.lane_decision = 'move_right'
                print(lc_right_condition)


        return [act_long, self.lateral_actions[self.lane_decision]]



class VehicleHandler:
    def __init__(self, config=None):
        # self.entry_vehicles =
        # self.queuing_vehicles =
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.next_vehicle_id = 0
        self.lane_width = 3.7
        self.percept_range = 100 #m
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
            # if elapsed_time % 70 == 0:
            #     id = self.next_vehicle_id
            #     glob_x = 0
            #     lane_id = np.random.choice(range(1, self.lanes_n+1))
            #     new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)
            #     new_vehicle.capability = 'IDMMOBIL'
            #     new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
            #     new_vehicle_entries.append(new_vehicle)
            #     self.next_vehicle_id += 1

            if elapsed_time % 20 == 0:
                for lane_id in range(1, self.lanes_n+1):
                    coin_flip = np.random.random()
                    if coin_flip < 0.3:
                        id = self.next_vehicle_id
                        glob_x =  np.random.uniform(-30, 0)

                        aggressiveness = np.random.uniform(0, 1)
                        new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, aggressiveness)

                        new_vehicle.glob_y = (self.lanes_n-lane_id+1)*self.lane_width-self.lane_width/2
                        new_vehicle_entries.append(new_vehicle)
                        self.next_vehicle_id += 1

        return new_vehicle_entries

    def find_closest(self, glob_x, lane_vehicles):
        """Returns the closest vehicles.
        """
        front_vehicle, rear_vehicle = None, None # default

        if not lane_vehicles:
            return front_vehicle, rear_vehicle
        else:
            front_vehicles = [vehicle for vehicle in lane_vehicles if vehicle.glob_x>=glob_x]
            rear_vehicles = [vehicle for vehicle in lane_vehicles if vehicle.glob_x<=glob_x]
            if front_vehicles:
                glob_xs = [vehicle.glob_x for vehicle in front_vehicles]
                min_glob_x = min(glob_xs)
                if min_glob_x - glob_x <= self.percept_range:
                    indx = glob_xs.index(min_glob_x)
                    front_vehicle = front_vehicles[indx]
            if rear_vehicles:
                glob_xs = [vehicle.glob_x for vehicle in rear_vehicles]
                max_glob_x = max(glob_xs)
                if glob_x - max_glob_x  <= self.percept_range:
                    indx = glob_xs.index(max_glob_x)
                    rear_vehicle = rear_vehicles[indx]

            return front_vehicle, rear_vehicle


    def find_neighbours(self, ego, vehicles):
        """returns list of current neighbouring vehicles.
        """
        lane_id = ego.lane_id
        id = ego.id
        glob_x = ego.glob_x
        neighbours = {
                      'f':None,
                      'fl':None,
                      'rl':None,
                      'rr':None,
                      'fr':None
                        }

        same_lane = []
        left_lane = []
        right_lane = []

        for vehicle in vehicles:
            if vehicle.id != id:
                if vehicle.lane_id == lane_id:
                    same_lane.append(vehicle)
                elif vehicle.lane_id == lane_id+1:
                    right_lane.append(vehicle)
                elif vehicle.lane_id == lane_id-1:
                    left_lane.append(vehicle)

        neighbours['f'], neighbours['r'] = self.find_closest(glob_x, same_lane)
        neighbours['fr'], neighbours['rr'] = self.find_closest(glob_x, right_lane)
        neighbours['fl'], neighbours['rl'] = self.find_closest(glob_x, left_lane)
        # print(neighbours)

        return neighbours

    def get_vehicle_state(self, vehicle):

        pass

    def update_reservations(self, vehicle):
        if vehicle.id in self.reservations:
            if vehicle.lane_decision == 'keep_lane':
                del self.reservations[vehicle.id]
        else:
            if vehicle.lane_decision != 'keep_lane':
                max_glob_x, min_glob_x = vehicle.glob_x + 100, vehicle.glob_x - 100
                self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]


    # def gen_sdv(self, sdv):
    #     if not sdv:


class Road:
    def __init__(self, lane_length, lane_width, lanes_n):
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.lanes_n = lanes_n

class Env:
    def __init__(self, config):
        self.config = config
        self.elapsed_time = 0 # number of print(lane_id)s past
        self.handler = VehicleHandler(config)
        self.sdv = None
        self.vehicles = []
        self.initiate_environment()
        # self.vehicles = []

    def initiate_environment(self):
        self.lane_length = self.config['lane_length']

        # new_vehicle_entries = self.handler.place_vehicle()
        # self.vehicles = new_vehicle_entries

    def step(self, action=None):
        """ steps the environment forward in time.
        """
        vehicles = [] # list for vehicles with new states
        # print(len(self.vehicles))
        for vehicle_i in self.vehicles:
            if vehicle_i.glob_x > self.lane_length:
                # consider vehicle gone
                if vehicle_i.id in self.handler.reservations:
                    del self.handler.reservations[vehicle_i.id]
                continue

            neighbours = self.handler.find_neighbours(vehicle_i, self.vehicles)
            # obs = self.observe(vehicle_i, neighbours)
            action = vehicle_i.act(neighbours, self.handler.reservations)

            self.handler.update_reservations(vehicle_i)
            vehicle_i.neighbours = neighbours
            # print(action)
            # action = vehicle_i.act()
            vehicle_ii = copy.copy(vehicle_i)
            vehicle_ii.step(action)
            vehicles.append(vehicle_ii)


        self.vehicles = vehicles
        print(self.handler.reservations)
        new_vehicle_entries = self.handler.place_vehicle(len(vehicles), self.elapsed_time)
        if new_vehicle_entries:
            self.vehicles.extend(new_vehicle_entries)

        self.elapsed_time += 1





    def get_feature_vector(self, obs):
        return [item for sublist in obs.values() for item in sublist]

    # def get_relative_states(self, follower, leader):
    #     if not (follower or leader):
    #         return [0, 50]
    #
    #     delta_v = follower.speed-leader.speed
    #     delta_x = leader.glob_x-follower.glob_x
    #     return [delta_v, delta_x]

    #
    # def observe(self, ego, neighbours):
    #     """ returns current observation for a given vehicle.
    #     """
    #     if ego.capability == 'IDM':
    #         leader = neighbours['f']
    #         if leader:
    #             obs = [ego.speed-leader.speed, leader.glob_x-ego.glob_x}]
    #         else:
    #             obs = [0, 50]
    #         return obs
    #
    #     elif ego.capability == 'IDMMOBIL':
    #          obs = {
    #              'f': self.get_relative_states(ego,  neighbours['f']),
    #              'fl': self.get_relative_states(ego,  neighbours['fl']),
    #              'rl': self.get_relative_states(neighbours['rl']), ego),
    #              'r': self.get_relative_states(neighbours['r']), ego),
    #              'rr': self.get_relative_states(neighbours['rr']), ego),
    #              'fr': self.get_relative_states(ego,  neighbours['fr'])
    #              }
    #
    #     return obs


config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
viewer = Viewer(config)
def main():
    # for i in range(10):
    while True:
        viewer.render(env.vehicles)
        decision = input()
        if decision == 'n':
            sys.exit()

        env.step()

# %%
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def animation_frame(i):
        viewer.render(env.vehicles)
        env.step()
        # return line,

    animation = FuncAnimation(viewer.fig, func=animation_frame, frames=range(600), interval=1000)


    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=25, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('sim_example.mp4', writer, dpi=500)



if __name__=='__main__':
    main()

get_animation()
# plt.show()
# %%
