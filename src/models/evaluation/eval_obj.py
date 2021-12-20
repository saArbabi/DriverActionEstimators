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
    def __init__(self):
        self.env = EnvMergeMC(
                {'lanes_n':2,
                'lane_width':3.75,
                'lane_length':300})

        self.config = self.read_eval_config()
        self.env.metric_collection_mode = True
        self.model_in_prog = 0
        self.episode_in_prog = 0
        self.trace_in_prog = 0
        self.rollout_len = self.config['mc_config']['rollout_len']
        self.history_len = self.config['mc_config']['history_len']
        self.create_empty()

    def create_empty(self):
        """
        Create empty files in which monte carlo logs can be dumped.
        """
        self.ima_collection = {}
        self.real_collection = {}
        self.collision_log = []
        self.avg_compute_time = [] # compute time per trace

    def update_eval_config(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        progress_logging = self.config['progress_logging']
        mc_config = self.config['mc_config']
        progress_logging['last_update'] = dt_string
        progress_logging['model_in_prog'] = \
            f'{self.model_in_prog}/{len(self.model_vehicle_map)}'
        progress_logging['episode_in_prog'] = \
                                    f'{self.episode_in_prog}/{mc_config["episodes_n"]}'
        progress_logging['trace_in_prog'] = \
                                    f'{self.trace_in_prog}/{mc_config["trace_n"]}'
        self.config['progress_logging'] = progress_logging

        if self.config['status'] == 'NOT STARTED':
            self.config['status'] = 'IN PROGRESS ...'

        if self.episode_in_prog == mc_config['episodes_n'] and \
                    self.trace_in_prog == mc_config['trace_n']:
            self.config['status'] = 'COMPLETE'

        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)


    def read_eval_config(self):
        with open(self.eval_config_dir, 'rb') as handle:
            config = json.load(handle)
        return config

    def assign_neural_vehicle(self, model_name):
        if self.model_vehicle_map[model_name] == 'NeuralVehicle':
            epoch_count = '20'
            from vehicles.neural.neural_vehicle import NeuralVehicle
            self.env.neural_vehicle = NeuralVehicle()
        elif self.model_vehicle_map[model_name] == 'NeuralIDMVehicle':
            epoch_count = '20'
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
            epoch_count = '10'
            from vehicles.neural.lstm_vehicle import LSTMVehicle
            self.env.neural_vehicle = LSTMVehicle()

        self.env.neural_vehicle.initialize_agent(
                        model_name,
                        epoch_count,
                        self.config['mc_config']['data_id'])

    def dump_mc_logs(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/eval'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        with open(exp_dir+'/real_collection.pickle', 'wb') as handle:
            pickle.dump(self.real_collection, handle)

        with open(exp_dir+'/ima_collection.pickle', 'wb') as handle:
            pickle.dump(self.ima_collection, handle)

        if self.collision_log:
            with open(exp_dir+'/collision_log.pickle', 'wb') as handle:
                pickle.dump(self.collision_log, handle)

    def run_trace(self, trace, episode_id):
        self.trace_in_prog += 1
        self.env.initialize_env(episode_id)
        tf.random.set_seed(trace) # each trace has a unique tf seed

        time_start = time.time()
        for i in range(0, self.env.trans_time+self.rollout_len):
            self.env.step()
            if self.env.collision_detected:
                collision_id = f'{episode_id}_{trace}'
                if collision_id not in self.collision_log:
                    self.collision_log.append(collision_id)
                    print('collision_detected')

        time_end = time.time()
        runtime = time_end - time_start

        for veh_id, data_log in self.env.ima_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [episode_id, veh_id, trace, runtime]
            if not episode_id in self.ima_collection:
                self.ima_collection[episode_id] = {}
            if not veh_id in self.ima_collection[episode_id]:
                self.ima_collection[episode_id][veh_id] = [data_log]
            else:
                # append additional traces
                self.ima_collection[episode_id][veh_id].append(data_log)

        self.update_eval_config()

    def run_episode(self, episode_id):
        self.episode_in_prog += 1
        np.random.seed(episode_id)
        self.env.trans_time = np.random.randint(\
                            self.history_len, self.history_len+50) # controller ==> 'neural'

        for trace in range(self.config['mc_config']['trace_n']):
            self.run_trace(trace, episode_id)
        for veh_id, data_log in self.env.real_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [episode_id, veh_id, trace]
            if not episode_id in self.real_collection:
                self.real_collection[episode_id] = {}
            if not veh_id in self.real_collection[episode_id]:
                self.real_collection[episode_id][veh_id] = [data_log]
            else:
                # in case there are multiple traces per episode
                self.real_collection[episode_id][veh_id].append(data_log)

    def run(self):
        self.model_vehicle_map = self.config['model_vehicle_map']
        model_names = self.model_vehicle_map.keys()
        for model_name in model_names:
            self.model_in_prog += 1
            self.config['progress_logging']['model_name'] = model_name
            self.assign_neural_vehicle(model_name)
            for episode_id in range(501, 501+self.config['mc_config']['episodes_n']):
                self.run_episode(episode_id)
                self.dump_mc_logs(model_name)
