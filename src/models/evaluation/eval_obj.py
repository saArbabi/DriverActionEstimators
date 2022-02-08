import tensorflow as tf
from envs.merge_mc import EnvMergeMC
import time
from datetime import datetime
import os
import numpy as np
import pickle
import json

class MCEVAL():
    eval_config_dir = './src/models/evaluation/config.json'
    def __init__(self, mc_run_name):
        with open('./src/envs/config.json', 'rb') as handle:
            config = json.load(handle)
        self.env = EnvMergeMC(config)
        self.mc_run_name = mc_run_name # folder name in which val logs are dumped
        self.config = self.read_eval_config()
        self.env.metric_collection_mode = True
        self.rollout_len = self.config['mc_config']['rollout_len']
        self.history_len = self.config['mc_config']['history_len']

    def create_empty(self):
        """
        Create empty files in which monte carlo logs can be dumped.
        """
        self.ima_collection = {}
        self.real_collection = {}
        self.collision_log = []
        self.runtime = [] # compute time per trace

    def update_eval_config(self, model_name):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.config['mc_config']
        progress_logging = self.config['progress_logging'][model_name]
        progress_logging['last_update'] = dt_string
        progress_logging['current_episode_count'] = \
                                    f'{self.current_episode_count}/{mc_config["episodes_n"]}'

        progress_logging['episode_in_prog'] = self.episode_id

        if self.current_episode_count == mc_config['episodes_n']:
            self.config['status'] = 'COMPLETE'
        else:
            self.config['status'] = 'IN PROGRESS ...'


        self.config['progress_logging'][model_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def read_eval_config(self):
        with open(self.eval_config_dir, 'rb') as handle:
            config = json.load(handle)
        return config

    def assign_neural_vehicle(self, model_name):
        if self.model_vehicle_map[model_name] == 'NeuralVehicle':
            epoch_count = '10'
            from vehicles.neural.neural_vehicle import NeuralVehicle
            self.env.neural_vehicle = NeuralVehicle()
        elif self.model_vehicle_map[model_name] == 'NeuralIDMVehicle':
            epoch_count = '10'
            from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
            self.env.neural_vehicle = NeuralIDMVehicle()
        elif self.model_vehicle_map[model_name] == 'LatentMLPVehicle':
            epoch_count = '20'
            from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
            self.env.neural_vehicle = LatentMLPVehicle()
        elif self.model_vehicle_map[model_name] == 'MLPVehicle':
            epoch_count = '20'
            from vehicles.neural.mlp_vehicle import MLPVehicle
            self.env.neural_vehicle = MLPVehicle()
        elif self.model_vehicle_map[model_name] == 'LSTMVehicle':
            epoch_count = '20'
            from vehicles.neural.lstm_vehicle import LSTMVehicle
            self.env.neural_vehicle = LSTMVehicle()

        self.env.neural_vehicle.initialize_agent(
                        model_name,
                        epoch_count,
                        self.config['mc_config']['data_id'])

    def dump_mc_logs(self, model_name):
        exp_dir = './src/models/experiments/'+ self.mc_run_name + '/' + model_name
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        with open(exp_dir+'/real_collection.pickle', 'wb') as handle:
            pickle.dump(self.real_collection, handle)

        with open(exp_dir+'/ima_collection.pickle', 'wb') as handle:
            pickle.dump(self.ima_collection, handle)

        with open(exp_dir+'/runtime.pickle', 'wb') as handle:
            pickle.dump(self.runtime, handle)

        if self.collision_log:
            with open(exp_dir+'/collision_log.pickle', 'wb') as handle:
                pickle.dump(self.collision_log, handle)

    def run_trace(self, trace):
        self.env.initialize_env(self.episode_id)
        tf.random.set_seed(trace) # each trace has a unique tf seed

        time_start = time.time()
        for i in range(0, self.env.trans_time+self.rollout_len):
            self.env.step()
            if self.env.collision_detected:
                collision_id = f'{self.episode_id}_{trace}_'+self.env.collision_vehs
                if collision_id not in self.collision_log:
                    self.collision_log.append(collision_id)

        time_end = time.time()
        runtime = (time_end - time_start)/len(self.env.real_vehicles)

        for veh_id, data_log in self.env.ima_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [self.episode_id, veh_id, trace]
            if not self.episode_id in self.ima_collection:
                self.ima_collection[self.episode_id] = {}
            if not veh_id in self.ima_collection[self.episode_id]:
                self.ima_collection[self.episode_id][veh_id] = [data_log]
            else:
                # append additional traces
                self.ima_collection[self.episode_id][veh_id].append(data_log)
        self.runtime.append([self.episode_id, trace, runtime])

    def run_episode(self):
        self.episode_id += 1
        self.current_episode_count += 1
        np.random.seed(self.episode_id)
        self.env.trans_time = np.random.randint(\
                            self.history_len, self.history_len*2) # controller ==> 'neural'

        for trace in range(self.config['mc_config']['trace_n']):
            self.run_trace(trace)
        for veh_id, data_log in self.env.real_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [self.episode_id, veh_id, trace]
            if not self.episode_id in self.real_collection:
                self.real_collection[self.episode_id] = {}
            if not veh_id in self.real_collection[self.episode_id]:
                self.real_collection[self.episode_id][veh_id] = [data_log]
            else:
                # in case there are multiple traces per episode
                self.real_collection[self.episode_id][veh_id].append(data_log)

    def initiate_eval(self, model_name):
        self.current_episode_count = 0
        self.create_empty()
        progress_logging = {}
        self.episode_id = 500
        self.target_episode = self.config['mc_config']['episodes_n'] + \
                                                            self.episode_id

        progress_logging['episode_in_prog'] = self.episode_id
        progress_logging['current_episode_count'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.config['progress_logging'][model_name] = progress_logging

    def load_collections(self, model_name):
        exp_dir = './src/models/experiments/'+ self.mc_run_name + '/' + model_name
        with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
            self.real_collection = pickle.load(handle)

        with open(exp_dir+'/ima_collection.pickle', 'rb') as handle:
            self.ima_collection = pickle.load(handle)

        with open(exp_dir+'/runtime.pickle', 'rb') as handle:
            self.runtime = pickle.load(handle)

        try:
            with open(exp_dir+'/collision_log.pickle', 'rb') as handle:
                self.collision_log = pickle.load(handle)
        except:
            self.collision_log = []

    def is_eval_complete(self, model_name):
        """Check if this model has been fully evaluated.
        """
        if not model_name in self.config['progress_logging']:
            self.initiate_eval(model_name)
            return False

        progress_logging = self.config['progress_logging'][model_name]
        mc_config = self.config['mc_config']
        epis_n_left = 0 # remaining episodes ot compelte

        current_episode_count = progress_logging['current_episode_count']
        current_episode_count = current_episode_count.split('/')
        self.current_episode_count = int(current_episode_count[0])
        epis_n_left = mc_config['episodes_n'] - self.current_episode_count
        if epis_n_left == 0:
            return True
        else:
            self.load_collections(model_name)
            self.episode_id = progress_logging['episode_in_prog']
            progress_logging['current_episode_count'] = \
                        f'{self.current_episode_count}/{mc_config["episodes_n"]}'
            self.target_episode =  self.episode_id + epis_n_left
            self.update_eval_config(model_name)
            return False

    def run(self):
        self.model_vehicle_map = self.config['model_vehicle_map']
        model_names = self.model_vehicle_map.keys()

        for model_name in model_names:
            if self.is_eval_complete(model_name):
                continue
            self.assign_neural_vehicle(model_name)
            print('Model being evaluated: ', model_name)
            while self.episode_id < self.target_episode:
                self.run_episode()
                self.dump_mc_logs(model_name)
                self.update_eval_config(model_name)
