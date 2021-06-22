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
        # annotation_mark_2 = [round(veh.speed, 1) for veh in vehicles]
        # annotation_mark_2 = [round(veh.lane_y, 2) for veh in vehicles]
        for i in range(len(annotation_mark_1)):
            ax.annotate(annotation_mark_1[i], (glob_xs[i], glob_ys[i]+1))
            # ax.annotate(annotation_mark_2[i], (glob_xs[i], glob_ys[i]-1))



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
                print('glob_y: ', vehicle.glob_y)
                print('lane_y: ', vehicle.lane_y)


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

    def step(self, actions):
        act_long, act_lat = actions
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
        self.neighbours = {}
        self.perception_range = 100 #m
        self.lane_width = 3.8

        self.lateral_actions = {'move_left':0.7,
                                'move_right':-0.7,
                                'keep_lane':0
                                }
        self.set_idm_params(aggressiveness)


    def set_idm_params(self, aggressiveness):
        if aggressiveness == None:
            raise ValueError('No aggressiveness specified!')

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
        self.driver_params['aggressiveness'] = aggressiveness  # in range [0, 1]

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

        if 0 <= self.driver_params['aggressiveness'] < 0.33:
            # timid driver
            attentiveness = 0.5*self.lane_width*np.random.beta(2, 10)
        elif 0.33 <= self.driver_params['aggressiveness'] <= 0.66:
            # normal driver
            attentiveness = 0.5*self.lane_width*np.random.beta(3, 3)
        elif 0.66 < self.driver_params['aggressiveness']:
            # aggressive driver
            attentiveness = 0.5*self.lane_width*np.random.beta(10, 2)
        self.driver_params['attentiveness'] = round(attentiveness, 1)

        # self.driver_params['desired_v'] += np.random.normal(0, 1)
    def get_idm_param(self, Parameter_range, param_name):
        if param_name in ['desired_v', 'max_act', 'min_act', 'safe_braking']:
            # the larger the param, the more aggressive the driver
            min_value = Parameter_range['least_aggressvie'][param_name]
            max_value = Parameter_range['most_aggressive'][param_name]
            return  min_value + self.driver_params['aggressiveness']*(max_value-min_value)

        elif param_name in ['desired_tgap', 'min_jamx', 'politeness', 'act_threshold']:
            # the larger the param, the more timid the driver
            min_value = Parameter_range['most_aggressive'][param_name]
            max_value = Parameter_range['least_aggressvie'][param_name]
            return  max_value - self.driver_params['aggressiveness']*(max_value-min_value)


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

    def idm_actions(self, obs):
        delta_v, delta_x = obs
        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.driver_params['max_act']*(1-(self.speed/self.driver_params['desired_v'])**4-\
                                            (desired_gap/(delta_x+1e-5))**2)

        return round(act_long, 2)

    def check_reservations(self, target_lane, reservations):
        """To ensure two cars do not simultaneously move into the same lane.
        """
        if not reservations:
            return True
        else:
            for reserved in reservations.values():
                reserved_lane, max_glob_x, min_glob_x = reserved
                if target_lane == reserved_lane and min_glob_x < self.glob_x < max_glob_x:
                    return False
            return True

    def check_neighbours(self, neighbours):
        """To ensure neighbours keep lane while merger is changing lane.
        """
        for vehicle in neighbours.values():
            if vehicle and vehicle.lane_decision != 'keep_lane':
                return False
        return True

    def mobil_condition(self, actions_gains):
        """To decide if changing lane is worthwhile.
        """
        ego_gain, new_follower_gain, old_follower_gain = actions_gains
        lc_condition = ego_gain+self.driver_params['politeness']*(new_follower_gain+\
                                                                old_follower_gain )
        return lc_condition


    def act(self, neighbours, reservations):
        act_long = self.idm_actions(self.observe(self, neighbours['f']))
        if self.lane_decision == 'move_left':
            if self.lane_id == self.target_lane :
                if self.lane_y >= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'move_right':
            if self.lane_id == self.target_lane :
                if self.lane_y <= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

        elif self.lane_decision == 'keep_lane' and self.check_neighbours(neighbours):
            # if not :
            # # if not self.check_neighbours(neighbours) or self.glob_x < 100:
            #     # Keep lane if neighbours are chanigng lane or if you have just entered
            #     # highway
            #     pass

            lc_left_condition = 0
            lc_right_condition = 0

            act_rl_lc = self.idm_actions(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_actions(self.observe(neighbours['rr'], self))
            act_r_lc = self.idm_actions(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_actions(self.observe(neighbours['r'], self))
            old_follower_gain = act_r_lc-act_r_lk

            if self.lane_id > 1 and self.driver_params['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_actions(self.observe(neighbours['rl'], neighbours['fl']))
                act_ego_lc_l = self.idm_actions(self.observe(self, neighbours['fl']))
                ego_gain = act_ego_lc_l-act_long
                new_follower_gain = act_rl_lc-act_rl_lk
                lc_left_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if self.lane_id < self.lanes_n and \
                                                self.driver_params['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_actions(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_actions(self.observe(neighbours['rr'], neighbours['fr']))

                ego_gain = act_ego_lc_r-act_long
                new_follower_gain = act_rr_lc-act_rr_lk
                lc_right_condition = self.mobil_condition([ego_gain, new_follower_gain, old_follower_gain])

            if max([lc_left_condition, lc_right_condition]) > self.driver_params['act_threshold']:
                if lc_left_condition > lc_right_condition:
                    target_lane = self.target_lane - 1
                    if self.check_reservations(target_lane, reservations):
                        self.lane_decision = 'move_left'
                        self.target_lane -= 1
                        return [act_ego_lc_l, self.lateral_actions[self.lane_decision]]

                elif lc_left_condition < lc_right_condition:
                    target_lane = self.target_lane + 1
                    if self.check_reservations(target_lane, reservations):
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

    def my_neighbours(self, ego, vehicles):
        """Returns list of current neighbouring vehicles.
            Note:
            - When performing a merge, ego will follow the vehicle in the
            target lane if one exists.
        """
        neighbours = {}
        delta_xs_f, delta_xs_fl, delta_xs_rl, delta_xs_r, \
                        delta_xs_rr, delta_xs_fr = ([ego.perception_range] for i in range(6))
        candidate_f, candidate_fl, candidate_rl, candidate_r, \
                        candidate_rr, candidate_fr = (None for i in range(6))

        for vehicle in vehicles:
            if vehicle.id != ego.id:
                delta_x = vehicle.glob_x-ego.glob_x
                if vehicle.lane_id in [ego.lane_id, ego.lane_id+1, ego.lane_id-1] and \
                                  abs(delta_x) < ego.perception_range:

                    if ego.lane_decision != 'keep_lane':
                        if self.am_i_following(ego.target_lane, vehicle.lane_id, delta_x, delta_xs_f):
                            delta_xs_f.append(delta_x)
                            candidate_f = vehicle

                        elif self.will_i_lead(ego.target_lane, vehicle.lane_id, delta_x, delta_xs_r):
                            delta_xs_r.append(abs(delta_x))
                            candidate_r = vehicle
                        continue

                    elif ego.lane_decision == 'keep_lane':
                        if delta_x >= 0:
                            if vehicle.lane_id == ego.lane_id+1:
                                # right lane
                                if self.am_i_attending(ego, vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fr[-1]:
                                    delta_xs_fr.append(delta_x)
                                    candidate_fr = vehicle

                            elif vehicle.lane_id == ego.lane_id-1:
                                # left lane
                                if self.am_i_attending(ego, vehicle, delta_x, delta_xs_f):
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                                elif delta_x < delta_xs_fl[-1]:
                                    delta_xs_fl.append(delta_x)
                                    candidate_fl = vehicle

                            elif vehicle.lane_id == ego.lane_id:
                                # same lane
                                if delta_x < delta_xs_f[-1]:
                                    delta_xs_f.append(delta_x)
                                    candidate_f = vehicle

                        elif delta_x < 0:
                            delta_x = abs(delta_x)
                            if vehicle.lane_id == ego.lane_id+1:
                                # right lane
                                if delta_x < delta_xs_rr[-1]:
                                    delta_xs_rr.append(delta_x)
                                    candidate_rr = vehicle
                            elif vehicle.lane_id == ego.lane_id-1:
                                # left lane
                                if delta_x < delta_xs_rl[-1]:
                                    delta_xs_rl.append(delta_x)
                                    candidate_rl = vehicle
                            elif vehicle.lane_id == ego.lane_id:
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

    def am_i_attending(self, ego, vehicle, delta_x, delta_xs):
        """Am I attending to the merging car?
        """
        if vehicle.target_lane == ego.lane_id and \
                abs(vehicle.lane_y) > ego.driver_params['attentiveness'] \
                and delta_x < delta_xs[-1]:
            return True
        return False

    def am_i_following(self, ego_target_lane, vehicle_lane_id, delta_x, delta_xs):
        """Am I following 'vehicle' in my target lane?
        """
        if vehicle_lane_id == ego_target_lane and delta_x > 0 and delta_x < delta_xs[-1]:
            return True
        return False

    def will_i_lead(self, ego_target_lane, vehicle_lane_id, delta_x, delta_xs):
        """Will I be leading 'vehicle' at some point?
        """
        if vehicle_lane_id == ego_target_lane and delta_x < 0 \
                                            and abs(delta_x) < delta_xs[-1]:
            return True
        return False

    def update_reservations(self, vehicle):
        if vehicle.id in self.reservations and vehicle.lane_decision == 'keep_lane':
            del self.reservations[vehicle.id]
        elif vehicle.lane_decision != 'keep_lane':
            max_glob_x, min_glob_x = round(vehicle.glob_x) + 100, round(vehicle.glob_x) - 100
            self.reservations[vehicle.id] = [vehicle.target_lane, max_glob_x, min_glob_x]

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
        self.usage = None
        self.initiate_environment()
        # self.vehicles = []

    def initiate_environment(self):
        self.lane_length = self.config['lane_length']

    def recorder(self, ego, neighbours):
        """For recording vehicle trajectories. Used for:
        - model training
        - perfromance validations # TODO
        """
        if ego.glob_x < 100:
            return
        if ego.lane_decision == 'keep_lane':
            lane_decision = 0
        elif ego.lane_decision == 'move_left':
            lane_decision = 1
        elif ego.lane_decision == 'move_right':
            lane_decision = -1

        act_long, act_lat = ego.actions
        state = neighbours.copy()
        state['ego'] = [ego.speed, ego.glob_x, act_long, act_lat, ego.lane_y]
        for key, neighbour in neighbours.items():
            if neighbour:
                act_long, _ = neighbour.actions
                state[key] = [neighbour.speed, neighbour.glob_x, act_long]
            else:
                state[key] = None


        if not ego.id in self.recordings['info']:
            self.recordings['info'][ego.id] = ego.driver_params

        if not ego.id in self.recordings['states']:
            self.recordings['states'][ego.id] = []
            self.recordings['decisions'][ego.id] = []
            self.recordings['elapsed_time'][ego.id] = []

        self.recordings['states'][ego.id].append(state)
        self.recordings['decisions'][ego.id].append(lane_decision)
        self.recordings['elapsed_time'][ego.id].append(self.elapsed_time)

    def step(self, actions=None):
        """ steps the environment forward in time.
        """
        vehicles = [] # list for vehicles with new states
        for vehicle_i in self.vehicles:
            if vehicle_i.glob_x > self.lane_length:
                # vehicle has left the highway
                if vehicle_i.id in self.handler.reservations:
                    del self.handler.reservations[vehicle_i.id]
                continue
            neighbours = self.handler.my_neighbours(vehicle_i, self.vehicles)
            actions = vehicle_i.act(neighbours, self.handler.reservations)
            vehicle_i.neighbours = neighbours
            vehicle_i.actions = actions

            self.handler.update_reservations(vehicle_i)
            vehicle_ii = copy.copy(vehicle_i)
            vehicle_ii.step(actions)
            vehicles.append(vehicle_ii)

            if self.usage == 'data generation':
                self.recorder(vehicle_i, neighbours)


        self.vehicles = vehicles
        new_vehicle_entries = self.handler.place_vehicle(len(vehicles), self.elapsed_time)
        if new_vehicle_entries:
            self.vehicles.extend(new_vehicle_entries)

        self.elapsed_time += 1


class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.data_frames_n = 10000 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {'info':{}, 'elapsed_time':{}, 'decisions':{}, 'states':{}}
        self.indxs = {
                    'speeds':{'leader':0, 'follower':1, 'merger':2},
                    'actions':{'leader':3, 'follower':4, 'merger':5},
                    'relatives':{'follower_leader':[6, 7], 'follower_merger':[8, 9]},
                    'lane_y':10, 'leader_exists':11}


    def run_sim(self):
        for step in range(self.data_frames_n):
            self.env.step()

        return self.env.recordings

    def round_scalars(self, data_list):
        rounded_items = []
        for item in data_list:
            try:
                rounded_items.append(round(item, 2))
            except:
                rounded_items.append(item)
        return rounded_items

    def get_step_feature(self, merger_s, leader_s, follower_s):
        """
        Note: If a leader is missing or beyond the perception_range of the followe,
        np.nan is assigned to its feature values.
        """
        step_feature = []

        if follower_s:
            follower_speed, follower_glob_x, follower_act_long = follower_s
        else:
            return

        if leader_s:
            leader_speed, leader_glob_x, leader_act_long = leader_s
            if leader_glob_x-follower_glob_x < 100:
                leader_exists = 1
            else:
                leader_speed, leader_glob_x, leader_act_long = [np.nan]*3
                leader_exists = 0
        else:
            leader_speed, leader_glob_x, leader_act_long = [np.nan]*3
            leader_exists = 0

        merger_speed, merger_glob_x, merger_act_long, \
                                    merger_act_lat, ego_lane_y = merger_s


        step_feature = [leader_speed, follower_speed, merger_speed, \
                        leader_act_long, follower_act_long, merger_act_long]

        # as if follower following leader
        step_feature.extend([
                             follower_speed-leader_speed,
                             leader_glob_x-follower_glob_x
                            ])

        # as if follower following merger
        step_feature.extend([
                             follower_speed-merger_speed,
                             merger_glob_x-follower_glob_x
                             ])

        step_feature.extend([ego_lane_y, leader_exists])
        return self.round_scalars(step_feature)

    def get_split_indxs(self, ego_decisions):
        """
        Decision transitions can be:
        (1) 1, 1 , 1, 0 indicating manoeuvre end
        (2) -1, -1, -1, 0 indicating manoeuvre end
        (3) 0, 0, 0, 1 indicating manoeuvre start
        (4) 0, 0, 0, -1 indicating manoeuvre start
        This method returns indexes for (1) and (2).
        """
        decision_indxs = np.where(ego_decisions[:-1] != \
                                        ego_decisions[1:])[0].tolist()
        if not decision_indxs:
            return
        else:
            split_indxs = []
            for indx in decision_indxs:
                ego_end_decision = ego_decisions[indx]
                if ego_end_decision == 1 or ego_end_decision == -1:
                    split_indxs.append(indx+1)
        return [0] + split_indxs

    def extract_features(self, raw_recordings):
        """
        - remove redundancies: only keeping states for merger, leader and follower car.
        """
        feature_data = []
        episode_id = 0
        for veh_id in raw_recordings['info'].keys():
            elapsed_times = np.array(raw_recordings['elapsed_time'][veh_id])
            ego_decisions = np.array(raw_recordings['decisions'][veh_id])
            veh_states = raw_recordings['states'][veh_id]
            split_indxs = self.get_split_indxs(ego_decisions)
            if not split_indxs:
                # not a single lane change
                continue

            for i in range(len(split_indxs)-1):
                # each split forms an episode
                start_snip = split_indxs[i]
                end_snip = split_indxs[i+1]
                ego_end_decision = ego_decisions[end_snip]
                feature_data_episode = []

                for _step in range(start_snip, end_snip):
                    ego_decision = ego_decisions[_step]
                    elapsed_time = elapsed_times[_step]
                    veh_state = veh_states[_step]

                    if ego_end_decision == 1:
                        # an episode ending with a lane change left
                        if ego_decision == 0:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['fl'],
                                                                veh_state['rl'])

                        elif ego_decision == 1:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['f'],
                                                                veh_state['r'])

                    elif ego_end_decision == -1:
                        # an episode ending with a lane change right
                        if ego_decision == 0:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['fr'],
                                                                veh_state['rr'])

                        elif ego_decision == -1:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['f'],
                                                                veh_state['r'])
                    elif ego_end_decision == 0:
                        # an episode ending with lane keep
                        step_feature = self.get_step_feature(
                                                            veh_state['ego'],
                                                            veh_state['f'],
                                                            veh_state['r'])

                    if step_feature:
                        step_feature[0:0] = episode_id, veh_id, elapsed_time, ego_decision
                        feature_data_episode.append(step_feature)
                    else:
                        feature_data_episode = []
                        break

                if len(feature_data_episode) > 50:
                    # ensure enough steps are present within a given episode
                    episode_id += 1
                    feature_data.extend(feature_data_episode)

        # return feature_data
        return np.array(feature_data)

    def fill_missing_values(self, feature_data):
        """
        Fill dummy values for the missing lead vehicle.
        Note:
        Different dummy values need to be fed to the IDM action function. Here goal
        is to assign values to maintain close to gaussian data distributions. Later,
        to ensure an IDM follower is not perturbed by the leader, different dummy values
        will be assigned.
        """
        def fill_with_dummy(arr, indx, dummy_value):
            indx += 4 # first n=4 items are episode_id, veh_id, elapsed_time, ego_decision
            nan_mask = np.isnan(arr[:, indx])
            nan_indx = np.where(nan_mask)
            arr[nan_mask, indx] = dummy_value
            return arr

        indx = self.indxs['speeds']['leader']
        feature_data = fill_with_dummy(feature_data, indx, 25)
        indx = self.indxs['actions']['leader']
        feature_data = fill_with_dummy(feature_data, indx, 0)
        indx_delta_v,  indx_delta_x = self.indxs['relatives']['follower_leader']
        feature_data = fill_with_dummy(feature_data, indx_delta_v, 0)
        feature_data = fill_with_dummy(feature_data, indx_delta_x, 50)

        return feature_data

    #
    #
    # def sequence(self, feature_data):
    #     """
    #     Sequence the data into history/future sequences.
    #     """
    #     episode_ids = list(np.unique(feature_data[:, 0]))
    #     for episode_id in episode_ids:

    def prep_data(self):
        raw_recordings = self.run_sim()
        feature_data = self.extract_features(raw_recordings)
        feature_data = self.fill_missing_values(feature_data)
        return feature_data

    # def split_data(self):
    #     """Spli
    #     """
    #     train_xs = []
    #     train_ys = []


    # def preprocess(self, raw_data):
    #
    #     xs, xs_scaled, merger_a, ys, info, scaler = data_generator()
    #
    #     episode_ids = list(np.unique(xs[:, 0]))
    #     seq_scaled_s_h = []
    #     scaled_seq_xs_f = []
    #     unscaled_seq_xs_f = []
    #     seq_merger_a = []
    #     seq_ys_f = []
    #
    #     pass
    #
    # def save(self):
    #     pass

config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':1200 # m
        }

env = Env(config)
viewer = Viewer(config)
"""

data_config = {
                'future_seq_length':20,
                'history_seq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
feature_data = data_gen.prep_data()
# raw_recordings['info'][6]
# np.array(env.recordings)[:, -1]

# %%
feature_data[:, 4]
feature_data[:, 4]
#
a = np.array([False, True])
np.count_nonzero(feature_data == np.nan)

# mask = np.isnan(feature_data[:, 4])
# np.where(mask)


# %%
columns_n = feature_data.shape[-1]
for column in range(columns_n):
    plt.figure()
    _ = plt.hist(feature_data[:, column], bins=150)

# %%
        self.indxs = {
                    'speeds':{'leader':0, 'follower':1, 'merger':2},
                    'actions':{'leader':3, 'follower':4, 'merger':5},
                    'relatives':{'follower_leader':[6, 7], 'follower_merger':[8, 9]},
                    'lane_y':10, 'leader_exists':11}

a = 3 if 2 > 100 else 6
a
np.unique(feature_data[:, 0])
np.unique(feature_data[:, 0])
feature_data[:, -1].shape

feature_data[0].shape
feature_data[0]
feature_data[0]
feature_data[feature_data[:, 0]==0].shape
feature_data[feature_data[:, 0]==5][:, -1]


"""
# %%
def main():
    # for i in range(100):
    while True:
        if env.elapsed_time > 2000:
            decision = input()
            if decision == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(decision)
            except:
                pass

            viewer.render(env.vehicles)
            print(env.elapsed_time)
        env.step()


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