from data.data_prep import DataPrep
import numpy as np

class DataPrepLatentMLP(DataPrep):
    def __init__(self):
        super().__init__()

    def split_data(self, history_future_seqs, history_future_seqs_scaled):
        """
        Return:
        (1) history_future_usc
        (2) history_sca: fed to lstm encoder
        (3) future_idm_s: used at test time
        (4) future_m_veh_c: used at test time
        (5) history_e_veh_a: used at train time
        """
        history_seqs, future_seqs = history_future_seqs
        history_seqs_scaled, future_seqs_scaled = history_future_seqs_scaled
        col_names = ['episode_id', 'time_step',
                    'e_veh_speed', 'f_veh_speed',
                    'el_delta_v', 'el_delta_x',
                    'em_delta_v', 'em_delta_x',
                    'm_veh_speed','em_delta_y',
                    'delta_x_to_merge','m_veh_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]

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

        col_names = ['episode_id', 'time_step',
                        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                        'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
                        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
                        'e_veh_att', 'f_veh_exists', 'm_veh_exists']
        future_idm_s = future_seqs[:, :, self.names_to_index(col_names)]

        col_names = ['episode_id', 'time_step',
                     'm_veh_speed', 'em_delta_y', 'delta_x_to_merge', 'm_veh_exists']
        future_m_veh_c = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        col_names = ['episode_id', 'time_step', 'e_veh_action']
        history_e_veh_a = history_seqs[:, :, self.names_to_index(col_names)]

        data_arrays = [history_future_usc, history_sca, \
                        future_idm_s, future_m_veh_c, history_e_veh_a]

        return data_arrays
