import numpy as np
from collections import deque
from sklearn import preprocessing
np.random.seed(2020)
import time

class DataGenecrator():
    def __init__(self, env=None, config=None):
        if config:
            self.env_steps_n = config['env_steps_n']
        if env:
            self.env = env
            self.env.usage = 'data generation'
            self.env.recordings = {}
            self.env.veh_log = ['lane_decision', 'glob_x', 'glob_y', 'speed', 'act_long']
        self.initiate()

    def initiate(self):
        self.indxs = {}
        self.ego_internal_state = ['aggressiveness', 'desired_v', \
                'desired_tgap', 'min_jamx', 'max_act', 'min_act']
        feature_names = [
                 'episode_id', 'time_step',
                 'e_veh_id', 'f_veh_id', 'm_veh_id',
                 'e_veh_decision', 'f_veh_exists', 'm_veh_exists', 'e_veh_att',
                 'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                 'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                 'e_veh_action', 'f_veh_action', 'm_veh_action',
                 'aggressiveness', 'desired_v',
                 'desired_tgap', 'min_jamx', 'max_act', 'min_act',
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
        step_feature = [e_veh_decision, f_veh_exists, m_veh_exists,e_veh_att,
                        e_veh['glob_x'], f_veh['glob_x'], m_veh['glob_x'],
                        e_veh['speed'], f_veh['speed'], m_veh['speed'],
                        e_veh['act_long'], f_veh['act_long'], m_veh['act_long']]

        ego_internal_state = [e_veh.get(key) for key in self.ego_internal_state]
        step_feature.extend(ego_internal_state)

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
            if len(epis_features) > 60:
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
            end_episode()
            e_veh_ts = raw_recordings[e_veh_id]
            for time_step, e_veh in e_veh_ts.items():
                att_veh_id = e_veh['att_veh_id']
                f_veh_id = e_veh['f_veh_id']
                m_veh_id = e_veh['m_veh_id']

                if m_veh_id:
                    # try:
                    #     m_veh_id_next = e_veh_ts[time_step+1]['m_veh_id']
                    # except:
                    #     if epis_features:
                    #         end_episode()
                    #     continue

                    m_veh = raw_recordings[m_veh_id][time_step]
                    if m_veh_id == att_veh_id:
                        e_veh_att = 1
                    else:
                        e_veh_att = 0
                else:
                    m_veh = None
                    e_veh_att = 0

                if not att_veh_id:
                    if epis_features:
                        end_episode()
                    continue

                if f_veh_id:
                    try:
                        f_veh = raw_recordings[f_veh_id][time_step]
                    except:
                        if epis_features:
                            end_episode()
                        continue
                else:
                    f_veh = None

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
            return dummy_value

        cols_with_nans = ['f_veh_speed', 'm_veh_speed',
                          'f_veh_action','m_veh_action',
                          'el_delta_v', 'el_delta_x',
                          'em_delta_v', 'em_delta_x', 'em_delta_y',
                          'f_veh_glob_x', 'm_veh_glob_x']
        dummy_value_set = {}
        for col in cols_with_nans:
            dummy_value = fill_with_dummy(self.indxs[col])
            dummy_value_set[col] = dummy_value

        return features, dummy_value_set

    def sequence(self, features, history_length, future_length):
        """
        Sequence the data into history/future sequences.
        """
        if future_length == 1:
            episode_ids = list(np.unique(features[:, 0]))
            history_seqs, future_seqs = [], []
            for episode_id in episode_ids:
                epis_data = features[features[:, 0] == episode_id]
                history_seq = deque(maxlen=history_length)
                for step in range(len(epis_data)):
                    history_seq.append(epis_data[step])
                    if len(history_seq) == history_length:
                        history_seqs.append(list(history_seq))
                        future_seqs.append(epis_data[step:step+1])
            return [np.array(history_seqs), np.array(future_seqs)]
        else:
            episode_ids = list(np.unique(features[:, 0]))
            history_seqs, future_seqs = [], []
            for episode_id in episode_ids:
                epis_data = features[features[:, 0] == episode_id]
                history_seq = deque(maxlen=history_length)
                for step in range(len(epis_data)):
                    history_seq.append(epis_data[step])
                    if len(history_seq) == history_length:
                        future_indx = step + future_length
                        if future_indx > len(epis_data):
                            break

                        history_seqs.append(list(history_seq))
                        future_seqs.append(epis_data[step:future_indx])
            return [np.array(history_seqs), np.array(future_seqs)]

    def names_to_index(self, col_names):
        if type(col_names) == list:
            return [self.indxs[item] for item in col_names]
        else:
            return self.indxs[col_names]

    def split_data(self, history_future_seqs, history_future_seqs_scaled):
        history_seqs, future_seqs = history_future_seqs
        history_seqs_scaled, future_seqs_scaled = history_future_seqs_scaled
        # future and histroy states - fed to LSTMs
        col_names = ['episode_id', 'time_step',
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                'em_delta_y','f_veh_exists', 'm_veh_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history and future info for debugging/ visualisation
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
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                        'e_veh_att', 'f_veh_exists', 'm_veh_exists']
        # history_idm_s = history_seqs[:, :, self.names_to_index(col_names)]
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]
        # future_idm_s = np.append(history_idm_s, future_idm_s, axis=1)

        # future action of m_veh - fed to LSTMs
        col_names = ['episode_id', 'time_step', 'em_delta_y', 'm_veh_action',\
                                                    'f_veh_exists', 'm_veh_exists']
        # history_m_veh_a = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_m_veh_a = future_seqs_scaled[:, :, self.names_to_index(col_names)]
        # future_m_veh_a = np.append(history_m_veh_a, future_m_veh_a, axis=1)

        # future action of e_veh - used as target
        col_names = ['episode_id', 'time_step', 'e_veh_action']
        # history_e_veh_a = history_seqs[:, :, self.names_to_index(col_names)]
        future_e_veh_a = future_seqs[:, :, self.names_to_index(col_names)]
        # future_e_veh_a = np.append(history_e_veh_a, future_e_veh_a, axis=1)

        data_arrays = [history_future_usc, history_sca, future_sca, future_idm_s, \
                        future_m_veh_a, future_e_veh_a]

        return data_arrays

    def scale_data(self, features):
        col_names = ['e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']

        scalar_indexs = self.names_to_index(col_names)
        scaler = preprocessing.StandardScaler().fit(features[:, scalar_indexs])
        features_scaled = features.copy()
        features_scaled[:, scalar_indexs] = scaler.transform(features[:, scalar_indexs])
        return features_scaled, scaler

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

class DataGeneratorMerge(DataGenecrator):
    def __init__(self, env=None, config=None):
        if config:
            self.episodes_n = config['episodes_n']
        if env:
            self.env = env
            self.env.usage = 'data generation'
            self.env.recordings = {}
            self.env.veh_log = ['lane_decision', 'glob_x', 'glob_y', 'speed', 'act_long']
        self.initiate()

    def is_episode_complete(self):
        """Episode is considered complete when all cars have exited the road.
        """
        if not self.env.vehicles:
            return True
        return False

    def run_sim(self):
        for episode_id in range(1, self.episodes_n+1):
            self.env.episode_id = episode_id
            self.env.initialize_env(episode_id)
            while not self.is_episode_complete():
                self.env.step()
        return self.env.recordings

    def extract_features(self, raw_recordings):
        """
        Extrtacts features from e_veh's perspective.
        Note: e_veh is the vehicle to be modelled.
        """
        def add_info(f_veh_id, m_veh_id, time_step, episode_id):
            """Useful for debugging
            """
            f_veh_id = f_veh_id if f_veh_id else -1
            m_veh_id = m_veh_id if m_veh_id else -1
            return [episode_id, time_step, e_veh_id, f_veh_id, m_veh_id]

        def end_vehicle_tracing():
            """
            End episode when an episode is complete.
            """
            nonlocal trace_features
            features.extend(trace_features)
            trace_features = []

        episode_ids = list(raw_recordings.keys())
        features = []
        for episode_id in episode_ids:
            trace_features = []
            epis_data = raw_recordings[episode_id]
            vehicle_ids = list(epis_data.keys())
            for e_veh_id in vehicle_ids:
                end_vehicle_tracing()
                e_veh_ts = epis_data[e_veh_id]
                for time_step, e_veh in e_veh_ts.items():
                    att_veh_id = e_veh['att_veh_id']
                    f_veh_id = e_veh['f_veh_id']
                    m_veh_id = e_veh['m_veh_id']

                    if f_veh_id and f_veh_id != 'dummy':
                        f_veh = epis_data[f_veh_id][time_step]
                    else:
                        f_veh = None
                        f_veh_id = None

                    if m_veh_id:
                        m_veh = epis_data[m_veh_id][time_step]
                        if m_veh_id and m_veh_id == att_veh_id:
                            e_veh_att = 1
                        else:
                            e_veh_att = 0
                    else:
                        m_veh = None
                        e_veh_att = 0

                    step_feature = self.get_step_feature(e_veh, f_veh, m_veh, e_veh_att)
                    step_feature[0:0] = add_info(f_veh_id, m_veh_id, time_step, episode_id)
                    trace_features.append(step_feature)
        return np.array(features)

    def clean_sequences(self, history_seqs, future_seqs):
        """
        Remove unwanted samples
        """
        cond_hist = ((history_seqs[:,:, self.names_to_index('f_veh_id')] != -1) &\
                (history_seqs[:,:, self.names_to_index('e_veh_decision')] == 0)).all(axis=1)

        cond_fut = ((future_seqs[:,:, self.names_to_index('f_veh_id')] != -1) &\
                (future_seqs[:,:, self.names_to_index('e_veh_decision')] == 0)).all(axis=1)

        cond = np.all([cond_hist, cond_fut], axis=0)
        return history_seqs[cond], future_seqs[cond]

    def sequence(self, features, history_length, future_length):
        """
        Sequence the data into history/future sequences.
        """
        episode_ids = list(np.unique(features[:, 0]))
        history_seqs, future_seqs = [], []
        for episode_id in episode_ids:
            epis_data = features[features[:, 0] == episode_id]
            vehicle_ids = list(np.unique(epis_data[:, 2]))

            for e_veh_id in vehicle_ids:
                trace_data = epis_data[epis_data[:, 2] == e_veh_id]
                history_seq = deque(maxlen=history_length)
                for step in range(len(trace_data)):
                    history_seq.append(trace_data[step])
                    if len(history_seq) == history_length:
                        future_indx = step + future_length
                        if future_indx > len(trace_data):
                            break
                        history_seqs.append(list(history_seq))
                        future_seqs.append(trace_data[step:future_indx])

        return self.clean_sequences(np.array(history_seqs), np.array(future_seqs))
