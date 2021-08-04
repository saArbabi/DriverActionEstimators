import numpy as np
from collections import deque
from sklearn import preprocessing
np.random.seed(2020)
import time

class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.env_steps_n = 300 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {}
        self.env.veh_log = ['lane_decision', 'glob_x', 'glob_y', 'speed', 'act_long']
        self.indxs = {}

        feature_names = [
                 'episode_id', 'time_step',
                 'e_veh_id', 'f_veh_id', 'm_veh_id',
                 'e_veh_decision', 'f_veh_exists', 'm_veh_exists',
                 'aggressiveness', 'e_veh_att',
                 'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                 'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                 'e_veh_action', 'f_veh_action', 'm_veh_action',
                 'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x', 'em_delta_y']

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

    def get_step_feature(self, e_veh, f_veh, m_veh, e_veh_att):
        """
        Note: e_veh and f_veh always exist. If a m_veh is missing, np.nan
        is assigned to its feature values.
        """
        if not m_veh:
            m_veh = {key: np.nan for key in self.env.veh_log}
            m_veh_exists = 0
        else:
            m_veh_exists = 1

        if not f_veh:
            f_veh = {key: np.nan for key in self.env.veh_log}
            f_veh_exists = 0
        else:
            f_veh_exists = 1

        e_veh_decision = 1 if e_veh['lane_decision'] != 'keep_lane' else 0
        e_veh_aggressiveness = e_veh['aggressiveness']
        step_feature = [e_veh_decision, f_veh_exists, m_veh_exists,
                        e_veh_aggressiveness, e_veh_att,
                        e_veh['glob_x'], f_veh['glob_x'], m_veh['glob_x'],
                        e_veh['speed'], f_veh['speed'], m_veh['speed'],
                        e_veh['act_long'], f_veh['act_long'], m_veh['act_long']]

        step_feature.extend([
                             e_veh['speed']-f_veh['speed'],
                             f_veh['glob_x']-e_veh['glob_x']])

        step_feature.extend([
                             e_veh['speed']-m_veh['speed'],
                             m_veh['glob_x']-e_veh['glob_x'],
                             abs(m_veh['glob_y']-e_veh['glob_y'])])

        # print(f_veh['glob_x']-e_veh['glob_x'])
        # print()
        return step_feature

    def extract_features(self, raw_recordings):
        """
        Extrtacts features from e_veh's perspective.
        Note: e_veh is the vehicle to be modelled. It can be found in one of the
        following scenarios:
        (1) e_veh following a f_veh in its lane.
        (2) e_veh performing a lane change.
        (3) e_veh yielding to a m_veh.
        (4) e_veh following a f_veh who is changing lane.
        """

        def add_info(f_veh_id, m_veh_id, time_step):
            """Useful for debugging
            """
            f_veh_id = f_veh_id if f_veh_id else -1
            m_veh_id = m_veh_id if m_veh_id else -1
            return [episode_id, time_step, e_veh_id, f_veh_id, m_veh_id]

        def end_episode():
            """
            End episode when an episode is complete.
            """
            nonlocal epis_features, episode_id
            if len(epis_features) > 40:
                features.extend(epis_features)
                episode_id += 1
            epis_features = []

        features = []
        epis_features = []
        episode_id = 0
        vehicle_ids = list(raw_recordings.keys())
        print(len(vehicle_ids))
        np.random.shuffle(vehicle_ids)
        for e_veh_id in vehicle_ids:
            # if e_veh_id != 155:
            #     continue
            end_episode()
            e_veh_ts = raw_recordings[e_veh_id]
            for time_step, e_veh in e_veh_ts.items():
                att_veh_id = e_veh['att_veh_id']
                f_veh_id = e_veh['f_veh_id']
                m_veh_id = e_veh['m_veh_id']

                if not att_veh_id or not f_veh_id:
                    if epis_features:
                        end_episode()
                    continue

                att_veh = raw_recordings[att_veh_id][time_step]
                f_veh_id = e_veh['f_veh_id']
                m_veh_id = e_veh['m_veh_id']

                if f_veh_id:
                    f_veh = raw_recordings[f_veh_id][time_step]
                else:
                    f_veh = None

                if m_veh_id:
                    m_veh = raw_recordings[m_veh_id][time_step]
                    if m_veh_id == att_veh_id:
                        e_veh_att = 1
                    else:
                        e_veh_att = 0
                else:
                    m_veh = None
                    e_veh_att = 0

                step_feature = self.get_step_feature(e_veh, f_veh, m_veh, e_veh_att)
                step_feature[0:0] = add_info(f_veh_id, m_veh_id, time_step)
                epis_features.append(step_feature)
        return np.array(features)

    def fill_missing_values(self, features):
        """
        Fill dummy values for the missing lead vehicle.
        Note:
        Different dummy values need to be fed to the IDM action function. Here goal
        is to assign values to maintain close to gaussian data distributions. Later,
        to ensure an IDM e_veh is not perturbed by the f_veh, different dummy values
        will be assigned.
        """
        def fill_with_dummy(indx):
            nonlocal features
            col_vals = features[:, indx]
            dummy_value = col_vals[~np.isnan(col_vals)].mean()
            nan_mask = np.isnan(features[:, indx])
            nan_indx = np.where(nan_mask)
            features[nan_indx, indx] = dummy_value

        cols_with_nans = ['f_veh_speed', 'm_veh_speed',
                          'f_veh_action','m_veh_action',
                          'el_delta_v', 'el_delta_x',
                          'em_delta_v', 'em_delta_x', 'em_delta_y',
                          'f_veh_glob_x', 'm_veh_glob_x']
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
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'e_veh_action', 'f_veh_action', 'm_veh_action',
                'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                'em_delta_y', 'f_veh_exists', 'm_veh_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history+future info for debugging/ visualisation
        col_names = ['episode_id', 'time_step', 'e_veh_id',
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'e_veh_action', 'f_veh_action', 'm_veh_action',
                'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                'em_delta_y', 'e_veh_att', 'f_veh_exists', 'm_veh_exists',
                'e_veh_decision', 'aggressiveness']

        history_usc = history_seqs[:, :, self.names_to_index(col_names)]
        future_usc = future_seqs[:, :, self.names_to_index(col_names)]
        history_future_usc = np.append(history_usc, future_usc, axis=1)

        # future states - fed to idm_layer
        col_names = ['episode_id', 'time_step',
                        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                        'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                        'e_veh_att','f_veh_exists', 'm_veh_exists']
        history_idm_s = history_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = np.append(history_idm_s, future_idm_s, axis=1)

        # future action of m_veh - fed to LSTMs
        col_names = ['episode_id', 'time_step', 'em_delta_y', 'm_veh_action',\
                                                    'f_veh_exists', 'm_veh_exists']
        history_m_veh_a = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_m_veh_a = future_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_m_veh_a = np.append(history_m_veh_a, future_m_veh_a, axis=1)

        # future action of e_veh - used as target
        col_names = ['episode_id', 'time_step', 'e_veh_action']
        history_e_veh_a = history_seqs[:, :, self.names_to_index(col_names)]
        future_e_veh_a = future_seqs[:, :, self.names_to_index(col_names)]
        future_e_veh_a = np.append(history_e_veh_a, future_e_veh_a, axis=1)

        data_arrays = [history_future_usc, history_sca, future_sca, future_idm_s, \
                        future_m_veh_a, future_e_veh_a]

        return data_arrays

    def scale_data(self, features):
        col_names = ['f_veh_speed', 'e_veh_speed', 'm_veh_speed',
                        'e_veh_action', 'f_veh_action', 'm_veh_action',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']

        scalar_indexs = self.names_to_index(col_names)
        scaler = preprocessing.StandardScaler().fit(features[:, scalar_indexs])
        features_scaled = features.copy()
        features_scaled[:, scalar_indexs] = scaler.transform(features[:, scalar_indexs])
        return features_scaled

    def mask_steps(self, history_future_seqss):
        """
        This is needed for cases where the m_veh passes
        several cars (i.e., a sequence has states from more than one vehicle
        """
        history_seqs, future_seqs = history_future_seqss
        e_veh_id_index = self.indxs['e_veh_id']

        axis_0, axis_1 = np.where(history_seqs[:, 1:, e_veh_id_index] != \
                                    history_seqs[:, :-1, e_veh_id_index])
        for sample, step in zip(axis_0, axis_1):
            history_seqs[sample, :step+1, 1:] = 0

        axis_0, axis_1 = np.where(future_seqs[:, 1:, e_veh_id_index] != \
                                    future_seqs[:, :-1, e_veh_id_index])
        for sample, step in zip(axis_0, axis_1):
            future_seqs[sample, step+1:, 1:] = 0
        return [history_seqs, future_seqs]

    def prep_data(self):
        raw_recordings = self.run_sim()
        features = self.extract_features(raw_recordings)
        # features = self.fill_missing_values(features)
        # features = features[features[:, 7]==0] # aggrss
        # features = features[features[:, 6]==0] # m_veh exists
        # features_scaled = self.scale_data(features)
        # history_future_seqs = self.sequence(features, 20, 20)
        # history_future_seqs_scaled = self.sequence(features_scaled, 20, 20)
        # data_arrays = self.split_data(history_future_seqs, history_future_seqs_scaled)
        # return data_arrays
        return features

    # def save(self):
    #     pass
