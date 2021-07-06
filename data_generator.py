import numpy as np
from collections import deque
from sklearn import preprocessing
np.random.seed(2020)
import time

class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.env_steps_n = 2000 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {}
        self.env.veh_log = ['lane_decision', 'lane_id',
                             'target_lane', 'glob_x', 'lane_y', 'speed']
        self.indxs = {}

        feature_names = [
                 'episode_id', 'time_step',
                 'ego_id', 'leader_id', 'merger_id',
                 'ego_decision', 'leader_exists', 'merger_exists',
                 'aggressiveness', 'lane_y', 'ego_att',
                 'ego_speed', 'leader_speed', 'merger_speed',
                 'ego_action', 'leader_action', 'merger_action',
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']

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
            # merger = {'glob_x':ego['glob_x']+50, 'lane_y':0, 'speed':25, 'act_long':0}
            merger = {key: np.nan for key in leader.keys()}
            merger['lane_y'] = np.nan
            merger_exists = 0
        else:
            merger_exists = 1

        if not leader:
            # leader = {'glob_x':ego['glob_x']+50, 'lane_y':0, 'speed':25, 'act_long':0}
            leader = {key: np.nan for key in merger.keys()}
            leader_exists = 0
        else:
            leader_exists = 1

        ego_decision = 1 if ego['lane_decision'] != 'keep_lane' else 0
        ego_aggressiveness = ego['aggressiveness']
        step_feature = [ego_decision, leader_exists, merger_exists,
                        ego_aggressiveness, merger['lane_y'], ego_att,
                        ego['speed'], leader['speed'], merger['speed'],
                        ego['act_long'], leader['act_long'], merger['act_long']]

        # as if ego following leader
        # if leader_exists:
        step_feature.extend([
                             ego['speed']-leader['speed'],
                             leader['glob_x']-ego['glob_x']])
        # else:
        #     step_feature.extend([
        #                          np.nan,
        #                          np.nan])
        # as if ego following merger
        # if merger_exists:
        step_feature.extend([
                             ego['speed']-merger['speed'],
                             merger['glob_x']-ego['glob_x']])
        # else:
        #     step_feature.extend([
        #                          np.nan,
        #                          np.nan])
        # return self.round_scalars(step_feature)
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
        def trace_back(leader_id, merger_id, time_step):
            """
            When a merger is found, trace back while merger exists.
            """

            nonlocal epis_features
            init_lane_id = merger_ts[time_step+1]['lane_id']
            pointer = -1

            while True:
                try:
                    ego = ego_ts[time_step]
                    merger = merger_ts[time_step]
                    past_lane_id = merger_ts[time_step-1]['lane_id']
                except:
                    break
                att_veh_id = ego['att_veh_id']
                if att_veh_id:
                    # if att_veh exists, consider it to be a leader, regardless
                        # what its doing ()
                    leader = raw_recordings[att_veh_id][time_step]
                    leader_id = att_veh_id
                else:
                    # no car is being attended to
                    break

                delta_x = merger['glob_x'] - ego['glob_x']
                if abs(pointer) != len(epis_features)+1 and delta_x > 0 and \
                                        past_lane_id == init_lane_id and \
                                        ego['lane_decision'] == 'keep_lane':
                    step_feature = self.get_step_feature(ego, leader, merger, ego_att=0)
                    step_feature[0:0] = add_info(leader_id, merger_id, time_step)
                    epis_features[pointer] = step_feature
                    time_step -= 1
                    pointer -= 1
                else:
                    break

        def reset_episode():
            nonlocal epis_features, leader_id, merger_id, episode_id, \
                        leader, merger, ego_att, leader_ts, merger_ts
            if len(epis_features) > 50:
                features.extend(epis_features)
                episode_id += 1
            epis_features = []
            leader_id = None
            merger_id = None
            ego_att = 0
            leader = None
            merger = None
            leader_ts = None
            merger_ts = None

        def add_info(leader_id, merger_id, time_step):
            """Useful for debugging
            """
            leader_id = leader_id if leader_id else -1
            merger_id = merger_id if merger_id else -1
            return [episode_id, time_step, ego_id, leader_id, merger_id]

        def is_leader():
            """
            Leader can be doing one of the following:
            - exiting ego's lane
            - keeping lane while ego entering leader's lane
            - keeping lane while ego also keeps lane
            """
            if (ego['lane_decision'] == 'keep_lane' and \
                        att_veh['target_lane'] != ego['target_lane']) or \
                        (ego['lane_decision'] != 'keep_lane' and \
                                att_veh['lane_decision'] == 'keep_lane') or \
                ego['lane_decision'] == att_veh['lane_decision'] == 'keep_lane':
                return True
            return False

        def is_merger(att_veh, ego):
            if att_veh['lane_decision'] != 'keep_lane' and \
                                att_veh['target_lane'] == ego['lane_id'] and \
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
            if ego_id != 287:
                continue
            reset_episode()
            ego_ts = raw_recordings[ego_id]
            for time_step, ego in ego_ts.items():

                att_veh_id = ego['att_veh_id']
                if att_veh_id:
                    att_veh = raw_recordings[att_veh_id][time_step]
                else:
                    # no point if ego is not attending to anyone
                    reset_episode()
                    continue

                if not is_merger(att_veh, ego):
                    leader_id = att_veh_id
                    leader_ts = raw_recordings[att_veh_id]
                    leader = att_veh
                elif leader_id:
                    try:
                        leader = leader_ts[time_step]
                    except:
                        # leader existed at some point, but no more.
                        leader = None
                        leader_id = None

                if is_merger(att_veh, ego):
                    #Episode ends
                    if merger_id and merger_id != att_veh_id:
                        reset_episode()
                        continue
                elif merger_id:
                    reset_episode()
                    continue

                if is_merger(att_veh, ego):
                    if not merger_id:
                        # reset_episode()
                        merger_ts = raw_recordings[att_veh_id]
                        if epis_features:
                            trace_back(leader_id, att_veh_id, time_step-1)
                    merger_id = att_veh_id
                    merger = att_veh
                    ego_att = 1

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

        cols_with_nans = ['lane_y', 'leader_speed', 'merger_speed', 'leader_action', \
                'merger_action', 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']
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
                'lane_y', 'leader_exists', 'merger_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history+future info for debugging/ visualisation
        col_names = ['episode_id', 'time_step', 'ego_id',
                'ego_speed', 'leader_speed', 'merger_speed',
                'ego_action', 'leader_action', 'merger_action',
                'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
                'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
                'ego_decision', 'aggressiveness']

        history_usc = history_seqs[:, :, self.names_to_index(col_names)]
        future_usc = future_seqs[:, :, self.names_to_index(col_names)]
        history_future_usc = np.append(history_usc, future_usc, axis=1)

        # future states - fed to idm_layer
        col_names = ['episode_id', 'time_step', 'ego_speed',
                        'fl_delta_v', 'fl_delta_x',
                        'fm_delta_v', 'fm_delta_x',
                        'leader_exists', 'merger_exists']
        history_idm_s = history_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = np.append(history_idm_s, future_idm_s, axis=1)

        # future action of merger - fed to LSTMs
        col_names = ['episode_id', 'time_step', 'merger_exists', 'merger_action', 'lane_y']
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
