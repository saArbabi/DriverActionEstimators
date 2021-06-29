import numpy as np
from collections import deque
from sklearn import preprocessing
np.random.seed(2020)
import time

class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.env_steps_n = 500 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {}
        self.env.veh_log = ['lane_decision', 'lane_id',
                             'target_lane', 'glob_x', 'lane_y', 'speed']
        self.indxs = {}
        feature_names = ['episode_id', 'follower_id', 'time_step', \
                 'follower_speed', 'leader_speed', 'merger_speed', \
                 'follower_action', 'leader_action', 'merger_action', \
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
                 'lane_y', 'merger_exists', 'aggressiveness']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1

    def run_sim(self):
        for step in range(self.env_steps_n):
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

    def get_step_feature(self, follower, leader, merger):
        """
        Note: Follower and leader always exist. If a merger is missing, np.nan
        is assigned to its feature values.
        """
        step_feature  = []

        if not merger:
            merger = {key: np.nan for key in leader.keys()}
            merger_exists = 0
        else:
            merger_exists = 1
            follower, leader, merger

        step_feature = [follower['speed'], leader['speed'], merger['speed'], \
                        follower['act_long'], leader['act_long'], merger['act_long']]

        # as if follower following leader
        step_feature.extend([
                             follower['speed']-leader['speed'],
                             leader['glob_x']-follower['glob_x']])

        # as if follower following merger
        step_feature.extend([
                             follower['speed']-merger['speed'],
                             merger['glob_x']-follower['glob_x']])

        step_feature.extend([merger['lane_y'], merger_exists, \
                                                    follower['aggressiveness']])
        # return self.round_scalars(step_feature)
        return step_feature


    def extract_features(self, raw_recordings):
        """
        Extrtacts features from follower's perspective.
        """
        def trace_back(time_step):
            """
            When a merger is found, trace back its state while it is being observed
            by follower.
            """
            pointer = -1
            while True:
                try:
                    follower, leader, merger = follower_ts[time_step], \
                                            leader_ts[time_step], merger_ts[time_step]

                    if merger['glob_x'] >= follower['glob_x'] and \
                                            merger['glob_x'] <= leader['glob_x']:
                            step_feature = self.get_step_feature(\
                                                            follower, leader, merger)
                            step_feature[0:0] = [episode_id, follower_id, time_step]
                            epis_features[pointer] = step_feature
                    else:
                        break
                except:
                    break
                time_step -= 1
                pointer -= 1

        def is_epis_end():
            """
            To tell if an episode should end. Reasons for ending an episode:
            - follower changes lane.
            - leader leaves lane.
            - merger completes its manoeuvre.
            """
            if leader['lane_decision'] != 'keep_lane' or \
                                follower['lane_decision'] != 'keep_lane':
                return True

            elif merger:
                if merger['lane_decision'] == 'keep_lane':
                    return True
            return False

        def reset_episode():
            nonlocal epis_features, leader_id, merger_id, episode_id
            if len(epis_features) > 10:
                features.extend(epis_features)
                episode_id += 1
            epis_features = []
            leader_id = None
            merger_id = None

        features = [] # episode_id ...
        episode_id = 0
        epis_features = []
        leader_id = None
        merger_id = None

        for follower_id in raw_recordings.keys():
            # if follower_id != 14:
            #     continue
            follower_ts = raw_recordings[follower_id]
            reset_episode()

            for time_step, follower in follower_ts.items():
                if follower['att_veh_id']:
                    att_veh = raw_recordings[follower['att_veh_id']][time_step]
                else:
                    # no point if follower is not attending to anyone
                    reset_episode()
                    continue

                if not leader_id:
                    if att_veh['lane_decision'] == 'keep_lane' and \
                                    att_veh['lane_id'] == follower['lane_id']:
                        # confirm this is the leader
                        leader_id = follower['att_veh_id']
                        leader_ts = raw_recordings[leader_id]
                        leader = leader_ts[time_step]
                    else:
                        reset_episode()
                        continue
                else:
                    leader = leader_ts[time_step]

                if not merger_id:
                    if att_veh['lane_decision'] != 'keep_lane' and \
                                    att_veh['target_lane'] == follower['lane_id']:
                        # confirm this is the leader
                        merger_id = follower['att_veh_id']
                        merger_ts = raw_recordings[merger_id]
                        trace_back(time_step-1)
                        merger = merger_ts[time_step]
                    else:
                        merger = None
                else:
                    merger = merger_ts[time_step]

                if is_epis_end():
                    reset_episode()
                    continue
                step_feature = self.get_step_feature(follower, leader, merger)
                step_feature[0:0] = [episode_id, follower_id, time_step]
                epis_features.append(step_feature)

        return np.array(features)

    def fill_missing_values(self, features):
        """
        Fill dummy values for the missing lead vehicle.
        Note:
        Different dummy values need to be fed to the IDM action function. Here goal
        is to assign values to maintain close to gaussian data distributions. Later,
        to ensure an IDM follower is not perturbed by the leader, different dummy values
        will be assigned.
        """
        def fill_with_dummy(indx):
            nonlocal features
            dummy_value = non_nan_values[:, indx].mean()
            nan_mask = np.isnan(features[:, indx])
            nan_indx = np.where(nan_mask)
            features[nan_indx, indx] = dummy_value

        non_nan_values = features[~np.isnan(features).any(axis=1)]
        fill_with_dummy(self.indxs['merger_speed'])
        fill_with_dummy(self.indxs['merger_action'])
        fill_with_dummy(self.indxs['lane_y'])
        fill_with_dummy(self.indxs['fm_delta_v'])
        fill_with_dummy(self.indxs['fm_delta_x'])

        return features

    def sequence(self, features, history_length, future_length):
        """
        Sequence the data into history/future sequences.
        """
        episode_ids = list(np.unique(features[:, 0]))
        history_seqs, future_seqs = [], []
        for episode_id in episode_ids:
            epis_data = features[features[:, 0] == episode_id]
            history_seq = deque(maxlen=history_length)
            for step in range(len(epis_data)):
                history_seq.append(epis_data[step])
                if len(history_seq) == history_length:
                    future_indx = step + future_length
                    if future_indx + 1 > len(epis_data):
                        break

                    history_seqs.append(list(history_seq))
                    future_seqs.append(epis_data[step+1:future_indx+1])
        return [np.array(history_seqs), np.array(future_seqs)]

    def names_to_index(self, col_names):
        return [self.indxs[item] for item in col_names]

    def split_data(self, history_future_seqs_seqs, history_future_seqs_scaled):
        history_seqs, future_seqs = history_future_seqs_seqs
        history_seqs_scaled, future_seqs_scaled = history_future_seqs_scaled
        # future and histroy states - fed to LSTMs
        col_names = ['episode_id',
                'follower_speed', 'leader_speed', 'merger_speed', \
                'follower_action', 'leader_action', 'merger_action', \
                'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
                'lane_y', 'merger_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history+future info for debugging/ visualisation
        col_names = ['episode_id', 'time_step',
                'follower_action', 'leader_action', 'merger_action', \
                'lane_y', 'aggressiveness']

        history_usc = history_seqs[:, :, self.names_to_index(col_names)]
        future_usc = future_seqs[:, :, self.names_to_index(col_names)]
        history_future_usc = np.append(history_usc, future_usc, axis=1)

        # future states - fed to idm_layer
        col_names = ['episode_id', 'follower_speed',
                        'fl_delta_v', 'fl_delta_x',
                        'fm_delta_v', 'fm_delta_x']
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]

        # future action of merger - fed to LSTMs
        col_names = ['episode_id', 'merger_action', 'lane_y']
        future_merger_a = future_seqs[:, :, self.names_to_index(col_names)]

        # future action of follower - used as target
        col_names = ['episode_id', 'follower_action']
        future_follower_a = future_seqs[:, :, self.names_to_index(col_names)]

        data_arrays = [history_future_usc, history_sca, future_sca, future_idm_s, \
                        future_merger_a, future_follower_a]

        return data_arrays

    def scale_data(self, features):
        col_names = ['leader_speed', 'follower_speed', 'merger_speed', \
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']

        scalar_indexs = self.names_to_index(col_names)
        scaler = preprocessing.StandardScaler().fit(features[:, scalar_indexs])
        features_scaled = features.copy()
        features_scaled[:, scalar_indexs] = scaler.transform(features[:, scalar_indexs])
        return features_scaled

    def mask_steps(self, history_future_seqss):
        """
        This is needed for cases where the merger passes
        several cars (i.e., a sequence has states from more than one vehicle
        """
        history_seqs, future_seqs = history_future_seqss
        follower_id_index = self.indxs['follower_id']

        axis_0, axis_1 = np.where(history_seqs[:, 1:, follower_id_index] != \
                                    history_seqs[:, :-1, follower_id_index])
        for sample, step in zip(axis_0, axis_1):
            history_seqs[sample, :step+1, 1:] = 0

        axis_0, axis_1 = np.where(future_seqs[:, 1:, follower_id_index] != \
                                    future_seqs[:, :-1, follower_id_index])
        for sample, step in zip(axis_0, axis_1):
            future_seqs[sample, step+1:, 1:] = 0
        return [history_seqs, future_seqs]

    def prep_data(self):
        raw_recordings = self.run_sim()
        features = self.extract_features(raw_recordings)

        features = self.fill_missing_values(features)
        features_scaled = self.scale_data(features)
        history_future_seqs_seqs = self.sequence(features, 20, 20)
        # history_future_seqs_seqs = self.mask_steps(history_future_seqs_seqs)
        history_future_seqs_scaled = self.sequence(features_scaled, 20, 20)
        # history_future_seqs_scaled = self.mask_steps(history_future_seqs_scaled)
        data_arrays = self.split_data(history_future_seqs_seqs, history_future_seqs_scaled)
        return data_arrays
        # return data_arrays, raw_recordings['info']

    # def save(self):
    #     pass
