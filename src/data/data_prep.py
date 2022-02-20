import numpy as np
from collections import deque
from sklearn import preprocessing

class DataPrep():
    def __init__(self):
        self.indxs = {}
        index_names = [
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
        for index_name in index_names:
            self.indxs[index_name] = index
            index += 1

    def names_to_index(self, col_names):
        if type(col_names) == list:
            return [self.indxs[item] for item in col_names]
        else:
            return self.indxs[col_names]

    def clean_sequences(self, history_seqs, future_seqs):
        """
        Remove unwanted samples
        """
        cond_hist = ((history_seqs[:,:, self.names_to_index('f_veh_id')] != -1)).all(axis=1)
        cond_fut = ((future_seqs[:,:, self.names_to_index('f_veh_id')] != -1)).all(axis=1)

        action_cond_hist = ((np.abs(history_seqs[:,:, self.names_to_index('e_veh_action')]) > 0.1)).any(axis=1)
        action_cond_fut = ((np.abs(future_seqs[:,:, self.names_to_index('e_veh_action')]) > 0.1)).any(axis=1)

        cond = np.all([cond_hist, cond_fut, action_cond_hist, action_cond_fut], axis=0)
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
                    'e_veh_speed', 'f_veh_speed',
                    'el_delta_v', 'el_delta_x',
                    'em_delta_v', 'em_delta_x',
                    'm_veh_speed','em_delta_y',
                    'delta_x_to_merge','m_veh_exists']
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
                 'desired_v','desired_tgap', 'min_jamx', 'max_act', 'min_act',
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
        # m_veh_exists is a boolean (not scaled) and is appended later
        col_names = ['episode_id', 'time_step',
            'm_veh_speed', 'em_delta_y', 'delta_x_to_merge', 'm_veh_exists']
        future_m_veh_c = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        # future action of e_veh - used as target
        col_names = ['episode_id', 'time_step', 'e_veh_action', 'e_veh_speed']
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
