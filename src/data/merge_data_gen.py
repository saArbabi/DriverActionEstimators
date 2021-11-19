import numpy as np
from collections import deque
from sklearn import preprocessing
from data.data_gen import DataGen

class DataGenMerge(DataGen):
    def __init__(self, env, episodes_n):
        self.episodes_n = episodes_n
        if env:
            self.env = env
            self.env.usage = 'data generation'
            self.env.recordings = {}
            self.env.veh_log = ['lane_decision', 'lane_id', 'glob_x', \
                                     'glob_y', 'speed', 'act_long']
        self.initiate()

    def initiate(self):
        self.indxs = {}
        self.ego_internal_state = ['aggressiveness', 'desired_v', \
                'desired_tgap', 'min_jamx', 'max_act', 'min_act']
        feature_names = [
                 'episode_id', 'time_step',
                 'e_veh_id', 'f_veh_id', 'm_veh_id',
                 'e_veh_decision', 'e_veh_lane',
                 'f_veh_exists', 'm_veh_exists', 'e_veh_att',
                 'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                 'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                 'e_veh_action', 'f_veh_action', 'm_veh_action',
                 'aggressiveness', 'desired_v',
                 'desired_tgap', 'min_jamx', 'max_act', 'min_act',
                 'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                 'em_delta_y', 'delta_x_to_merge']

        index = 0
        for item_name in feature_names:
            self.indxs[item_name] = index
            index += 1

    def is_episode_complete(self):
        """Episode is considered complete if all
            cars exit the road.
        """
        car_exists = True
        for vehicle in self.env.vehicles:
            if vehicle.glob_x < self.env.lane_length:
                return False
        else:
            return True

    def run_sim(self):
        for episode_id in range(1, self.episodes_n+1):
            self.env.episode_id = episode_id
            self.env.initialize_env(episode_id)
            while not self.is_episode_complete():
                self.env.step()
        return self.env.recordings

    def get_step_feature(self, vehicles):
        e_veh, f_veh, m_veh, e_veh_att = vehicles
        if not f_veh:
            f_veh = {key: np.nan for key in self.env.veh_log}
            f_veh_exists = 0
        else:
            f_veh_exists = 1

        if not m_veh:
            m_veh = {key: np.nan for key in self.env.veh_log}
            m_veh_exists = 0
        else:
            m_veh_exists = 1

        e_veh_decision = 1 if e_veh['lane_decision'] != 'keep_lane' else 0
        step_feature = [e_veh_decision, e_veh['lane_id'],
                        f_veh_exists, m_veh_exists, e_veh_att,
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
                             abs(m_veh['glob_y']-e_veh['glob_y']),
                             200-m_veh['glob_x']])
        return step_feature

    def extract_features(self, raw_recordings):
        """
        Extrtacts features from e_veh's perspective.
        Note: e_veh is the vehicle to be modelled.
        """
        def add_info(vehicle_ids):
            """Useful for debugging
            """
            f_veh_id, m_veh_id = vehicle_ids
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
                if e_veh_ts[0]['lane_id'] == 2:
                    # merging car's behavior is not predicted currently
                    continue
                for time_step, e_veh in e_veh_ts.items():
                    att_veh_id = e_veh['att_veh_id']
                    f_veh_id = e_veh['f_veh_id']
                    m_veh_id = e_veh['m_veh_id']

                    if f_veh_id:
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

                    vehicle_ids = [f_veh_id, m_veh_id]
                    vehicles = [e_veh, f_veh, m_veh, e_veh_att]
                    step_feature = self.get_step_feature(vehicles)
                    step_feature[0:0] = add_info(vehicle_ids)
                    trace_features.append(step_feature)
        return np.array(features)

    def clean_sequences(self, history_seqs, future_seqs):
        """
        Remove unwanted samples
        """
        cond_hist = ((history_seqs[:,:, self.names_to_index('f_veh_id')] != -1) &\
                (history_seqs[:,:, self.names_to_index('e_veh_glob_x')] < 250)).all(axis=1)

        cond_fut = ((future_seqs[:,:, self.names_to_index('f_veh_id')] != -1) &\
                (future_seqs[:,:, self.names_to_index('e_veh_glob_x')] < 250)).all(axis=1)

        cond = np.all([cond_hist, cond_fut], axis=0)
        return history_seqs[cond], future_seqs[cond]

    def sequence(self, features, history_len, rollout_len):
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
                history_seq = deque(maxlen=history_len)
                for step in range(len(trace_data)):
                    history_seq.append(trace_data[step])
                    if len(history_seq) == history_len:
                        future_indx = step + rollout_len
                        if future_indx > len(trace_data):
                            break
                        history_seqs.append(list(history_seq))
                        future_seqs.append(trace_data[step:future_indx])

        return self.clean_sequences(np.array(history_seqs), np.array(future_seqs))

    def scale_data(self, features):
        features_scaled = features.copy()
        # env state
        col_names = ['e_veh_speed', 'f_veh_speed',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x']

        scalar_indexs = self.names_to_index(col_names)
        env_scaler = preprocessing.StandardScaler().fit(features[:, scalar_indexs])
        features_scaled[:, scalar_indexs] = env_scaler.transform(features[:, scalar_indexs])
        # merger context
        col_names = ['m_veh_speed','em_delta_y', 'delta_x_to_merge']
        scalar_indexs = self.names_to_index(col_names)
        m_scaler = preprocessing.StandardScaler().fit(features[:, scalar_indexs])
        features_scaled[:, scalar_indexs] = m_scaler.transform(features[:, scalar_indexs])

        return features_scaled, env_scaler, m_scaler

    def split_data(self, history_future_seqs, history_future_seqs_scaled):
        history_seqs, future_seqs = history_future_seqs
        history_seqs_scaled, future_seqs_scaled = history_future_seqs_scaled
        # future and histroy states - fed to LSTMs
        col_names = ['episode_id', 'time_step',
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'el_delta_v', 'el_delta_x',
                'em_delta_v', 'em_delta_x',
                'em_delta_y', 'delta_x_to_merge', 'm_veh_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history and future info for debugging/ visualisation
        col_names = [
                 'episode_id', 'time_step',
                 'e_veh_id', 'f_veh_id', 'm_veh_id',
                 'm_veh_exists', 'e_veh_att',
                 'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                 'e_veh_action', 'f_veh_action', 'm_veh_action',
                 'aggressiveness',
                 'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                 'em_delta_y', 'delta_x_to_merge']

        history_usc = history_seqs[:, :, self.names_to_index(col_names)]
        future_usc = future_seqs[:, :, self.names_to_index(col_names)]
        history_future_usc = np.append(history_usc, future_usc, axis=1)

        # future states - fed to idm_layer
        col_names = ['episode_id', 'time_step',
                        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                        'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                        'e_veh_att', 'f_veh_exists', 'm_veh_exists']
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]

        # future context of m_veh - fed to LSTMs
        # this holds all the context needed to infer merger type
        col_names = ['episode_id', 'time_step', 'm_veh_speed', 'em_delta_y', 'delta_x_to_merge']
        future_m_veh_c = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        # future action of e_veh - used as target
        col_names = ['episode_id', 'time_step', 'e_veh_action']
        future_e_veh_a = future_seqs[:, :, self.names_to_index(col_names)]

        data_arrays = [history_future_usc, history_sca, future_sca, future_idm_s, \
                        future_m_veh_c, future_e_veh_a]

        return data_arrays

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
                          'em_delta_v', 'em_delta_x',
                          'delta_x_to_merge',
                          'em_delta_y',
                          'f_veh_glob_x', 'm_veh_glob_x']
        dummy_value_set = {}
        for col in cols_with_nans:
            dummy_value = fill_with_dummy(self.indxs[col])
            dummy_value_set[col] = dummy_value

        return features, dummy_value_set
