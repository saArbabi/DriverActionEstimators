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
                             'target_lane', 'glob_x', 'glob_y', 'speed', 'act_long']
        self.indxs = {}

        feature_names = [
                 'episode_id', 'time_step',
                 'ego_id', 'leader_id', 'merger_id',
                 'ego_decision', 'leader_exists', 'merger_exists',
                 'aggressiveness', 'ego_att',
                 'ego_glob_x', 'leader_glob_x', 'merger_glob_x',
                 'ego_speed', 'leader_speed', 'merger_speed',
                 'ego_action', 'leader_action', 'merger_action',
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', 'fm_delta_y']

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

    def get_step_feature(self, ego, leader, merger, ego_att):
        """
        Note: ego and leader always exist. If a merger is missing, np.nan
        is assigned to its feature values.
        """
        if not merger:
            merger = {key: np.nan for key in self.env.veh_log}
            merger_exists = 0
        else:
            merger_exists = 1

        if not leader:
            leader = {key: np.nan for key in self.env.veh_log}
            leader_exists = 0
        else:
            leader_exists = 1

        ego_decision = 1 if ego['lane_decision'] != 'keep_lane' else 0
        ego_aggressiveness = ego['aggressiveness']
        step_feature = [ego_decision, leader_exists, merger_exists,
                        ego_aggressiveness, ego_att,
                        ego['glob_x'], leader['glob_x'], merger['glob_x'],
                        ego['speed'], leader['speed'], merger['speed'],
                        ego['act_long'], leader['act_long'], merger['act_long']]

        step_feature.extend([
                             ego['speed']-leader['speed'],
                             leader['glob_x']-ego['glob_x']])

        step_feature.extend([
                             ego['speed']-merger['speed'],
                             merger['glob_x']-ego['glob_x'],
                             abs(merger['glob_y']-ego['glob_y'])])

        # print(leader['glob_x']-ego['glob_x'])
        # print()
        return step_feature

    def extract_features(self, raw_recordings):
        """
        Extrtacts features from ego's perspective.
        Note: ego is the vehicle to be modelled. It can be found in one of the
        following scenarios:
        (1) Ego following a leader in its lane.
        (2) Ego performing a lane change.
        (3) Ego yielding to a merger.
        (4) Ego following a leader who is changing lane.
        """
        # print(raw_recordings[47])
        def trace_back(merger_id, time_step):
            """
            When a merger is found, trace back while:
            - merger exists
            - follower keeps lane
            - follower leader unchanged
            """
            nonlocal epis_features

            while True:
                try:
                    leader_id = ego_ts[time_step]['att_veh_id']
                    ego = ego_ts[time_step]
                    merger = merger_ts[time_step]
                    merger_decision_tt = merger['lane_decision']
                    merger_decision_t = merger_ts[time_step-1]['lane_decision']
                    leader = raw_recordings[leader_id][time_step]
                    ml_delta_x = leader['glob_x'] - merger['glob_x']
                except:
                    break

                if merger_decision_tt == 'keep_lane' and \
                                        merger_decision_t != 'keep_lane':
                    # merger performing an other lane change
                    break

                fm_delta_x = merger['glob_x'] - ego['glob_x']
                if fm_delta_x > 1 and ml_delta_x > 1 and \
                            ego['lane_decision'] == 'keep_lane' and merger_id != leader_id:
                    step_feature = self.get_step_feature(ego, leader, merger, ego_att=0)
                    step_feature[0:0] = add_info(leader_id, merger_id, time_step)
                    epis_features.append(step_feature)
                    time_step -= 1
                else:
                    break
            epis_features = list(reversed(epis_features))
            # print(len(epis_features))

        def end_episode():
            """
            End episode when an episode is complete.
            """
            nonlocal epis_features, episode_id, merger_id, merger, ego_att
            merger_id = None
            merger = None
            ego_att = 0

            if len(epis_features) > 40:
                features.extend(epis_features)
                episode_id += 1
            epis_features = []

        def reset_episode():
            nonlocal leader_id, merger_id, \
                        leader, merger, ego_att, leader_ts, merger_ts

            end_episode()
            leader_id = None
            merger_id = None
            leader = None
            leader_ts = None
            ego_att = 0
            merger = None
            merger_ts = None

        def add_info(leader_id, merger_id, time_step):
            """Useful for debugging
            """
            leader_id = leader_id if leader_id else -1
            merger_id = merger_id if merger_id else -1
            return [episode_id, time_step, ego_id, leader_id, merger_id]

        def is_merger(att_veh, ego):
            if att_veh['lane_decision'] != 'keep_lane' and \
                                ego['lane_decision'] == 'keep_lane':
                return True
            return False

        features = [] # episode_id ..
        epis_features = []
        leader_id = None
        merger_id = None
        ego_att = 0
        leader = None
        merger = None
        leader_ts = None
        merger_ts = None
        episode_id = 0
        vehicle_ids = list(raw_recordings.keys())
        print(len(vehicle_ids))
        np.random.shuffle(vehicle_ids)
        for ego_id in vehicle_ids:
            # if ego_id != 155:
            #     continue
            reset_episode()
            ego_ts = raw_recordings[ego_id]
            for time_step, ego in ego_ts.items():
                att_veh_id = ego['att_veh_id']
                if not att_veh_id:
                    # no point if ego is not attending to anyone
                    reset_episode()
                    continue
                else:
                    att_veh = raw_recordings[att_veh_id][time_step]
                    if is_merger(att_veh, ego):
                        if merger_id != att_veh_id:
                            # there is a new merger
                            end_episode()
                            merger_ts = raw_recordings[att_veh_id]
                            ego_att = 1
                            trace_back(att_veh_id, time_step-1)
                            merger_id = att_veh_id
                            if merger_id == leader_id:
                                leader = None
                                leader_id = None

                        merger = merger_ts[time_step]
                        if leader_id:
                            try:
                                leader = leader_ts[time_step]
                            except:
                                leader = None
                                leader_id = None
                    else:
                        if merger_id:
                            end_episode()

                        if leader_id == att_veh_id:
                            leader = leader_ts[time_step]

                        else:
                            # new leader
                            leader_id = att_veh_id
                            leader_ts = raw_recordings[att_veh_id]
                            leader = leader_ts[time_step]



                    # if leader_id and leader_id != merger_id:
                    #     try:
                    #         leader = leader_ts[time_step]
                    #     except:
                    #         leader = None
                    #         leader_id = None
                    #
                    # elif leader_id == merger_id:
                    #     # happens when eg0 (follower) and merger are both changing lane
                    #     leader = None
                    #     leader_id = None
                # print('time_step')
                # print(time_step)
                # # print(epis_features)
                # print(str(leader_id), leader)
                # print(str(merger_id), merger)
                # print(merger)


                step_feature = self.get_step_feature(ego, leader, merger, ego_att)
                step_feature[0:0] = add_info(leader_id, merger_id, time_step)
                epis_features.append(step_feature)

        return np.array(features)

    def fill_missing_values(self, features):
        """
        Fill dummy values for the missing lead vehicle.
        Note:
        Different dummy values need to be fed to the IDM action function. Here goal
        is to assign values to maintain close to gaussian data distributions. Later,
        to ensure an IDM ego is not perturbed by the leader, different dummy values
        will be assigned.
        """
        def fill_with_dummy(indx):
            nonlocal features
            col_vals = features[:, indx]
            dummy_value = col_vals[~np.isnan(col_vals)].mean()
            nan_mask = np.isnan(features[:, indx])
            nan_indx = np.where(nan_mask)
            features[nan_indx, indx] = dummy_value

        cols_with_nans = ['leader_speed', 'merger_speed',
                          'leader_action','merger_action',
                          'fl_delta_v', 'fl_delta_x',
                          'fm_delta_v', 'fm_delta_x', 'fm_delta_y',
                          'leader_glob_x', 'merger_glob_x']
        for col in cols_with_nans:
            fill_with_dummy(self.indxs[col])

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

    def split_data(self, history_future_seqs, history_future_seqs_scaled):
        history_seqs, future_seqs = history_future_seqs
        history_seqs_scaled, future_seqs_scaled = history_future_seqs_scaled
        # future and histroy states - fed to LSTMs
        col_names = ['episode_id', 'time_step',
                'ego_speed', 'leader_speed', 'merger_speed',
                'ego_action', 'leader_action', 'merger_action',
                'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
                'fm_delta_y', 'leader_exists', 'merger_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history+future info for debugging/ visualisation
        col_names = ['episode_id', 'time_step', 'ego_id',
                'ego_speed', 'leader_speed', 'merger_speed',
                'ego_action', 'leader_action', 'merger_action',
                'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
                'fm_delta_y', 'ego_att', 'leader_exists', 'merger_exists',
                'ego_decision', 'aggressiveness']

        history_usc = history_seqs[:, :, self.names_to_index(col_names)]
        future_usc = future_seqs[:, :, self.names_to_index(col_names)]
        history_future_usc = np.append(history_usc, future_usc, axis=1)

        # future states - fed to idm_layer
        col_names = ['episode_id', 'time_step',
                        'ego_speed', 'leader_speed', 'merger_speed',
                        'ego_glob_x', 'leader_glob_x', 'merger_glob_x', 'ego_att']
        history_idm_s = history_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = np.append(history_idm_s, future_idm_s, axis=1)

        # future action of merger - fed to LSTMs
        col_names = ['episode_id', 'time_step', 'fm_delta_y', 'leader_exists', 'merger_exists']
        future_merger_a = future_seqs[:, :, self.names_to_index(col_names)]

        # future action of ego - used as target
        col_names = ['episode_id', 'time_step', 'ego_action']
        history_ego_a = history_seqs[:, :, self.names_to_index(col_names)]
        future_ego_a = future_seqs[:, :, self.names_to_index(col_names)]
        future_ego_a = np.append(history_ego_a, future_ego_a, axis=1)

        data_arrays = [history_future_usc, history_sca, future_sca, future_idm_s, \
                        future_merger_a, future_ego_a]

        return data_arrays

    def scale_data(self, features):
        col_names = ['leader_speed', 'ego_speed', 'merger_speed',
                        'ego_action', 'leader_action', 'merger_action',
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
        ego_id_index = self.indxs['ego_id']

        axis_0, axis_1 = np.where(history_seqs[:, 1:, ego_id_index] != \
                                    history_seqs[:, :-1, ego_id_index])
        for sample, step in zip(axis_0, axis_1):
            history_seqs[sample, :step+1, 1:] = 0

        axis_0, axis_1 = np.where(future_seqs[:, 1:, ego_id_index] != \
                                    future_seqs[:, :-1, ego_id_index])
        for sample, step in zip(axis_0, axis_1):
            future_seqs[sample, step+1:, 1:] = 0
        return [history_seqs, future_seqs]

    def prep_data(self):
        raw_recordings = self.run_sim()
        features = self.extract_features(raw_recordings)
        # features = self.fill_missing_values(features)
        # features = features[features[:, 7]==0] # aggrss
        # features = features[features[:, 6]==0] # merger exists
        # features_scaled = self.scale_data(features)
        # history_future_seqs = self.sequence(features, 20, 20)
        # history_future_seqs_scaled = self.sequence(features_scaled, 20, 20)
        # data_arrays = self.split_data(history_future_seqs, history_future_seqs_scaled)
        # return data_arrays
        return features

    # def save(self):
    #     pass
