# from factory.viewer import Viewer
import copy
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np

class Viewer():
    def __init__(self, config):
        self.config  = config
        self.fig = plt.figure(figsize=(10, 4))
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
        xs_idm = [veh.glob_x for veh in vehicles if veh.capability == 'IDM']
        ys_idm = [veh.glob_y for veh in vehicles if veh.capability == 'IDM']

        xs_idm_mobil = [veh.glob_x for veh in vehicles if veh.capability == 'IDMMOBIL']
        ys_idm_mobil = [veh.glob_y for veh in vehicles if veh.capability == 'IDMMOBIL']
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

        ax.scatter(xs_idm, ys_idm, s=100, marker=">", \
                                        facecolors='blue', edgecolors='blue')
        ax.scatter(xs_idm_mobil, ys_idm_mobil, s=100, marker=">", \
                                        facecolors='red')
            # ax.annotate(round(veh.v, 1), (veh.glob_x, veh.glob_y+0.1))
            # ax.annotate(round(veh.v, 1), (veh.glob_x, veh.glob_y+0.1))

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
                self.lane_id['current'] += 1
                self.lane_y += self.lane_width
            elif self.lane_y >= self.lane_width/2:
                # just stepped into left lane
                self.lane_id['current'] -= 1
                self.lane_y -= self.lane_width


    # def act(self):
    #     raise NotImplementedError
class IDMMOBILVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, speed, driver_disposition=None):
        super().__init__(id, lane_id, glob_x, speed)
        self.capability = 'IDM'
        self.lane_id = {'current':lane_id, 'next':lane_id}
        self.lane_decision = 'keep_lane'

        self.lateral_actions = {'move_left':0.7,
                                'move_right':-0.7,
                                'keep_lane':0
                                }
        self.set_idm_params(driver_disposition)


    def set_idm_params(self, driver_disposition):
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
        if not driver_disposition:
            raise ValueError('No driver_disposition specified!')

        if driver_disposition == 'normal_idm':
            self.idm_param = normal_idm
        if driver_disposition == 'timid_idm':
            self.idm_param = timid_idm
        if driver_disposition == 'aggressive_idm':
            self.idm_param = aggressive_idm

        self.idm_param['desired_v'] += np.random.normal(0, 1)
        self.mobil_param = {
                    'politeness':0.5,
                    'safe_braking':-2,
                    'acceleration_threshold':0.1
            }

    def get_desired_gap(self, delta_v):
        gap = self.idm_param['min_jamx'] + self.idm_param['desired_tgap']*self.speed+(self.speed*delta_v)/ \
                    (2*np.sqrt(self.idm_param['max_act']*self.idm_param['min_act']))
        return gap

    def observe(self, follower, leader):
        if not follower or not leader:
            return [0, 100]

        delta_v = follower.speed-leader.speed
        delta_x = leader.glob_x-follower.glob_x
        return [delta_v, delta_x]

    def idm_action(self, obs):
        delta_v, delta_x = obs
        desired_gap = self.get_desired_gap(delta_v)
        act_long = self.idm_param['max_act']*(1-(self.speed/self.idm_param['desired_v'])**4-\
                                            (desired_gap/delta_x)**2)
        return act_long

    def mobil_action(self, neighbours):
        if self.lane_decision == 'move_left':
            if self.lane_id['current'] == self.lane_id['next']:
                act_long = self.idm_action(self.observe(self, neighbours['f']))
                if self.lane_y >= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'

            elif self.lane_id['current'] > self.lane_id['next']:
                act_long = self.idm_action(self.observe(self, neighbours['fl']))

        elif self.lane_decision == 'move_right':
            if self.lane_id['current'] == self.lane_id['next']:
                act_long = self.idm_action(self.observe(self, neighbours['f']))
                if self.lane_y <= 0:
                    # manoeuvre completed
                    self.lane_decision = 'keep_lane'
            elif self.lane_id['current'] < self.lane_id['next']:
                act_long = self.idm_action(self.observe(self, neighbours['fr']))

        else:
            lc_left_condition = 0
            lc_right_condition = 0

            # action for ego vehicle if ego is to keep lane
            act_rl_lc = self.idm_action(self.observe(neighbours['rl'], self))
            act_rr_lc = self.idm_action(self.observe(neighbours['rr'], self))
            act_ego_lk = self.idm_action(self.observe(self, neighbours['f']))
            act_r_lc = self.idm_action(self.observe(neighbours['r'], neighbours['f']))
            act_r_lk = self.idm_action(self.observe(neighbours['r'], self))

            if self.lane_id['current'] > 1 and self.mobil_param['safe_braking'] < act_rl_lc:
                # consider moving left
                act_rl_lk = self.idm_action(self.observe(neighbours['rl'], neighbours['fl']))

                act_ego_lc_l = self.idm_action(self.observe(self, neighbours['fl']))
                lc_left_condition = act_ego_lc_l-act_ego_lk+self.mobil_param['politeness']*\
                                                (act_rl_lc-act_rl_lk+act_r_lc-act_r_lk)

            elif self.lane_id['current'] < 4 and self.mobil_param['safe_braking'] < act_rr_lc:
                # consider moving right
                act_ego_lc_r = self.idm_action(self.observe(self, neighbours['fr']))
                act_rr_lk = self.idm_action(self.observe(neighbours['rr'], neighbours['fr']))

                lc_right_condition = act_ego_lc_r-act_ego_lk+self.mobil_param['politeness']*\
                                                (act_rr_lc-act_rr_lk+act_r_lc-act_r_lk)


            if lc_left_condition > self.mobil_param['acceleration_threshold']:
                act_long = act_ego_lc_l
                self.lane_decision = 'move_left'
                self.lane_id['next'] -= 1

            elif lc_right_condition > self.mobil_param['acceleration_threshold']:
                act_long = act_ego_lc_r
                self.lane_decision = 'move_right'
                self.lane_id['next'] += 1

            else:
                act_long = act_ego_lk


        return act_long, self.lateral_actions[self.lane_decision]


    def act(self, neighbours):
        if self.capability == 'IDMMOBIL':
            act_long, act_lat = self.mobil_action(neighbours)
            return [act_long, act_lat]

        elif self.capability == 'IDM':
            obs = self.observe(self, neighbours['f'])
            act_long = self.idm_action(obs)
            return [act_long, 0]

class VehicleHandler:
    def __init__(self, config=None):
        # self.entry_vehicles =
        # self.queuing_vehicles =
        self.lanes_n = 4
        self.next_vehicle_id = 0
        self.lane_width = 3.7

    def gen_vehicle(self):
        """Creates a new IDM vehicle.
        """
        driver_disposition = 'normal_idm'
        glob_x = 0
        speed = 20
        new_vehicle_entries = []

        for lane_id in range(1, self.lanes_n+1):
            id = self.next_vehicle_id
            coin_flip = np.random.random()
            if coin_flip < 0.3:
                new_vehicle = IDMMOBILVehicle(id, lane_id, glob_x, speed, driver_disposition)
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
                vehicles_x_globs = [vehicle.glob_x for vehicle in front_vehicles]
                indx = vehicles_x_globs.index(min(vehicles_x_globs))
                front_vehicle = front_vehicles[indx]
            if rear_vehicles:
                vehicles_x_globs = [vehicle.glob_x for vehicle in rear_vehicles]
                indx = vehicles_x_globs.index(max(vehicles_x_globs))
                rear_vehicle = rear_vehicles[indx]

            return front_vehicle, rear_vehicle


    def find_neighbours(self, ego, vehicles):
        """returns list of current neighbouring vehicles.
        """
        lane_id = ego.lane_id['current']
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
                if vehicle.lane_id['current'] == lane_id:
                    same_lane.append(vehicle)
                elif vehicle.lane_id['current'] == lane_id+1:
                    right_lane.append(vehicle)
                elif vehicle.lane_id['current'] == lane_id-1:
                    left_lane.append(vehicle)

        neighbours['f'], neighbours['r'] = self.find_closest(glob_x, same_lane)
        neighbours['fr'], neighbours['rr'] = self.find_closest(glob_x, right_lane)
        neighbours['fl'], neighbours['rl'] = self.find_closest(glob_x, left_lane)
        # print(neighbours)

        return neighbours


    def set_vehicle_capability(self, ego, neighbours):
        """Ensures only one vehicle within a given neighbourhood can move laterally.
        """
        # neighbours = list(neighbours.values())
        
        for vehicle in neighbours.values():
            if vehicle:
                if vehicle.capability == 'IDMMOBIL':
                    ego.capability == 'IDM'
                    return
        ego.capability = 'IDMMOBIL'


    def get_vehicle_state(self, vehicle):

        pass

class Road:
    def __init__(self, lane_length, lane_width, lanes_n):
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.lanes_n = lanes_n

class Env:
    def __init__(self, config=None):
        self.config = config
        self.elapsed_time = 0 # number of print(lane_id)s past
        self.handler = VehicleHandler()
        self.initiate_environment()
        # self.vehicles = []

    def initiate_environment(self):
        self.config = {'lanes_n':4,
                'lane_width':3.7, # m
                'lane_length':600 # m
                }
        self.lane_length = 600

        new_vehicle_entries = self.handler.gen_vehicle()
        self.vehicles = new_vehicle_entries

    def step(self, action=None):
        """ steps the environment forward in time.
        """
        vehicles = [] # list for vehicles with new states
        # print(len(self.vehicles))
        for vehicle_i in self.vehicles:
            if vehicle_i.glob_x > self.lane_length:
                # consider vehicle gone
                continue
            neighbours = self.handler.find_neighbours(vehicle_i, self.vehicles)
            print(neighbours)

            # print(neighbours)
            self.handler.set_vehicle_capability(vehicle_i, neighbours)
            # obs = self.observe(vehicle_i, neighbours)
            action = vehicle_i.act(neighbours)
            # print(action)
            # action = vehicle_i.act()
            vehicle_ii = copy.copy(vehicle_i)
            vehicle_ii.step(action)
            vehicles.append(vehicle_ii)


        self.vehicles = vehicles
        self.elapsed_time += 1

        if self.elapsed_time % 20 == 0:
            new_vehicle_entries = self.handler.gen_vehicle()
            self.vehicles.extend(new_vehicle_entries)

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




env = Env()
# for i in range(10):
#     xs = [vehicle.glob_x for vehicle in env.vehicles]
#     ys = [vehicle.glob_y for vehicle in env.vehicles]
#     plt.figure()
#     plt.scatter(xs, ys)
#     env.step()
#
config = {'lanes_n':4,
        'lane_width':3.7, # m
        'lane_length':600 # m
        }

viewer = Viewer(config)
def run_sim():
    # for i in range(10):
    while True:
        viewer.render(env.vehicles)
        input()

        env.step()
        # cap = [vehicle.capability for vehicle in env.vehicles]
        # cap = [1 if x == 'IDM' else 0 for x in cap]
        # print(np.mean(cap))

run_sim()
# %%
