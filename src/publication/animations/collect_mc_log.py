"""
Use this for creating animations
"""
import sys
sys.path.insert(0, './src')
sys.path.insert(0, './src/envs')
from envs.merge_mc import EnvMergeMC
import pickle
import json
import tensorflow as tf


class RecordEpisode(EnvMergeMC):
    def __init__(self, config):
        super().__init__(config)
        self.rollout_len = 200
        self.history_len = 30
        self.traces_n = 50
        self.trans_time = 31
        self.collections = {}

        self.model_vehicle_map = {
            "neural_idm_367": "NeuralIDMVehicle",
            "neural_045": "NeuralVehicle",
            "latent_mlp_22": "LatentMLPVehicle",
            "mlp_05": "MLPVehicle",
            "lstm_05": "LSTMVehicle"}

    def veh_ima_action(self, veh_real, veh_ima):
        # self.set_ima_veh_neighbours(veh_real, veh_ima)
        veh_ima.neighbours = veh_ima.my_neighbours(self.ima_vehicles+[self.dummy_stationary_car])
        act_long = 0
        if veh_ima.vehicle_type == 'idmmobil_merge':
            self.set_ima_veh_decision(veh_real, veh_ima)

        if veh_ima.vehicle_type == 'neural':
            obs = veh_ima.neur_observe()
            # if self.check_collision(veh_ima):
                # self.collision_detected = True
            # if veh_real.id == 3:
            #     print(veh_ima.obs_history)
            veh_ima.update_obs_history(obs[0])

            if veh_ima.control_type != 'neural':
                if not (veh_ima.obs_history[0,0,:] == 0).all() and \
                                    self.time_step >= self.trans_time:

                    # controller change
                    veh_ima.control_type = 'neural'

            if veh_ima.control_type == 'neural':
                # _act_long = veh_ima.act(obs)
                act_long = veh_ima.act(obs)
                # if veh_ima.id == 'neur_4':
                    # act_long = 5
                veh_ima.act_long_c = act_long

        if not act_long:
            act_long = veh_real.act_long_c

        self.mc_log_info(veh_real, veh_ima)
        return act_long

    def fetch_state(self, vehicle):
        st = [
             self.time_step,
             vehicle.act_long_c,
             vehicle.speed,
             vehicle.glob_x,
             vehicle.glob_y]
        return st

    def mc_log_info(self, veh_real, veh_ima):
        """
        Informatin to be logged:
        - ego (real and imagined) global_x for rwse and collision detection
        - ego (real and imagined) speed for rwse
        - ego (real and imagined) action for comparing action distributions
        """
        veh_id =  veh_real.id
        if veh_id not in self.real_mc_log:
            self.real_mc_log[veh_id] = {}
            self.ima_mc_log[veh_id] = {}

            self.real_mc_log[veh_id] = []
            self.ima_mc_log[veh_id] = []

        self.real_mc_log[veh_id].append(self.fetch_state(veh_real))
        self.ima_mc_log[veh_id].append(self.fetch_state(veh_ima))

    def run_trace(self, trace, episode_id):
        self.initialize_env(episode_id)
        tf.random.set_seed(trace) # each trace has a unique tf seed

        for i in range(self.trans_time + self.rollout_len):
            self.step()

    def assign_neural_vehicle(self, model_name):
        if self.model_vehicle_map[model_name] == 'NeuralVehicle':
            epoch_count = '20'
            from vehicles.neural.neural_vehicle import NeuralVehicle
            self.neural_vehicle = NeuralVehicle()
        elif self.model_vehicle_map[model_name] == 'NeuralIDMVehicle':
            epoch_count = '20'
            from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
            self.neural_vehicle = NeuralIDMVehicle()
        elif self.model_vehicle_map[model_name] == 'LatentMLPVehicle':
            epoch_count = '20'
            from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
            self.neural_vehicle = LatentMLPVehicle()
        elif self.model_vehicle_map[model_name] == 'MLPVehicle':
            epoch_count = '20'
            from vehicles.neural.mlp_vehicle import MLPVehicle
            self.neural_vehicle = MLPVehicle()
        elif self.model_vehicle_map[model_name] == 'LSTMVehicle':
            epoch_count = '20'
            from vehicles.neural.lstm_vehicle import LSTMVehicle
            self.neural_vehicle = LSTMVehicle()

        self.neural_vehicle.initialize_agent(
                        model_name,
                        epoch_count,
                        '049')


    def save_logs(self):
        _dir = './src/publication/animations'
        with open(_dir+'/collections.pickle', 'wb') as handle:
            pickle.dump(self.collections, handle)

    def run_episode(self, model_name, episode_id):
        self.real_collection = {}
        self.ima_collection = {}

        self.assign_neural_vehicle(model_name)
        for trace in range(self.traces_n):
            self.run_trace(trace, episode_id)
            self.ima_collection[trace] = self.ima_mc_log
            if trace == 0:
                self.real_collection[trace] = self.real_mc_log
        self.collections[model_name] = [self.real_collection, self.ima_collection]


with open('./src/envs/config.json', 'rb') as handle:
    config = json.load(handle)



episode_id = 363
rec_obj = RecordEpisode(config)
for model_name in rec_obj.model_vehicle_map.keys():
    rec_obj.run_episode(model_name, episode_id)
rec_obj.save_logs()
