import numpy as np
from collections import deque
from sklearn import preprocessing
np.random.seed(2020)

class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.data_frames_n = 1000 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {'info':{}, 'elapsed_time':{}, 'decisions':{}, 'states':{}}
        self.indxs = {}
        all_col_names = ['episode_id', 'veh_id', 'elapsed_time', 'ego_decision', \
                 'leader_speed', 'follower_speed', 'merger_speed', \
                 'leader_action', 'follower_action', 'merger_action', \
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
                 'lane_y', 'leader_exists', 'follower_aggress', \
                 'follower_atten', 'follower_id']

        index = 0
        for item_name in all_col_names:
            self.indxs[item_name] = index
            index += 1

    def run_sim(self):
        for step in range(self.data_frames_n):
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

    def get_step_feature(self, merger_s, leader_s, follower_s):
        """
        Note: If a leader is missing or beyond the perception_range of the followe,
        np.nan is assigned to its feature values.
        """
        step_feature = []

        if follower_s:
            follower_speed, follower_glob_x, follower_act_long, \
                    follower_aggress, follower_atten, follower_id = follower_s
        else:
            return

        if leader_s:
            leader_speed, leader_glob_x, leader_act_long, _, _, _ = leader_s
            if leader_glob_x-follower_glob_x < 100:
                leader_exists = 1
            else:
                leader_speed, leader_glob_x, leader_act_long = [np.nan]*3
                leader_exists = 0
        else:
            leader_speed, leader_glob_x, leader_act_long = [np.nan]*3
            leader_exists = 0

        merger_speed, merger_glob_x, merger_act_long, \
                                merger_act_lat, ego_lane_y = merger_s


        step_feature = [leader_speed, follower_speed, merger_speed, \
                        leader_act_long, follower_act_long, merger_act_long]

        # as if follower following leader
        step_feature.extend([
                             follower_speed-leader_speed,
                             leader_glob_x-follower_glob_x
                            ])

        # as if follower following merger
        step_feature.extend([
                             follower_speed-merger_speed,
                             merger_glob_x-follower_glob_x
                             ])

        step_feature.extend([ego_lane_y, leader_exists, follower_aggress, \
                                                    follower_atten, follower_id])
        # return self.round_scalars(step_feature)
        return step_feature

    def get_split_indxs(self, ego_decisions):
        """
        Decision transitions can be:
        (1) 1, 1 , 1, 0 indicating manoeuvre end
        (2) -1, -1, -1, 0 indicating manoeuvre end
        (3) 0, 0, 0, 1 indicating manoeuvre start
        (4) 0, 0, 0, -1 indicating manoeuvre start
        This method returns indexes for (1) and (2).
        """
        decision_indxs = np.where(ego_decisions[:-1] != \
                                        ego_decisions[1:])[0].tolist()
        if not decision_indxs:
            return
        else:
            split_indxs = []
            for indx in decision_indxs:
                ego_end_decision = ego_decisions[indx]
                if ego_end_decision == 1 or ego_end_decision == -1:
                    if split_indxs:
                        start_snip = end_snip+1
                    else:
                        start_snip = 0
                    end_snip = indx
                    split_indxs.append([start_snip, end_snip])
        return split_indxs

    def extract_features(self, raw_recordings):
        """
        - remove redundancies: only keeping states for merger, leader and follower car.
        """
        feature_data = []
        episode_id = 0
        for veh_id in raw_recordings['info'].keys():
            elapsed_times = np.array(raw_recordings['elapsed_time'][veh_id])
            ego_decisions = np.array(raw_recordings['decisions'][veh_id])
            veh_states = raw_recordings['states'][veh_id]
            split_indxs = self.get_split_indxs(ego_decisions)
            if not split_indxs:
                # not a single lane change
                continue

            for split_indx in split_indxs:
                # each split forms an episode
                start_snip, end_snip = split_indx
                ego_end_decision = ego_decisions[end_snip]
                feature_data_episode = []

                for _step in range(start_snip, end_snip+1):
                    ego_decision = ego_decisions[_step]
                    elapsed_time = elapsed_times[_step]
                    veh_state = veh_states[_step]

                    if ego_end_decision == 1:
                        # an episode ending with a lane change left
                        if ego_decision == 0:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['fl'],
                                                                veh_state['rl'])

                        elif ego_decision == 1:
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['f'],
                                                                veh_state['r'])

                    elif ego_end_decision == -1:
                        # an episode ending with a lane change right
                        if ego_decision == 0:
                            # print(ego_decision)
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['fr'],
                                                                veh_state['rr'])

                        elif ego_decision == -1:
                            # print(ego_decision)
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['f'],
                                                                veh_state['r'])
                    elif ego_end_decision == 0:
                        # an episode ending with lane keep
                        step_feature = self.get_step_feature(
                                                            veh_state['ego'],
                                                            veh_state['f'],
                                                            veh_state['r'])

                    if step_feature:
                        step_feature[0:0] = episode_id, veh_id, elapsed_time, ego_decision
                        feature_data_episode.append(step_feature)
                    else:
                        break
                else:
                    # runs if no breaks in loop
                    if len(feature_data_episode) > 50:
                        # ensure enough steps are present within a given episode
                        episode_id += 1
                        feature_data.extend(feature_data_episode)

        # return feature_data
        return np.array(feature_data)

    def fill_missing_values(self, feature_data):
        """
        Fill dummy values for the missing lead vehicle.
        Note:
        Different dummy values need to be fed to the IDM action function. Here goal
        is to assign values to maintain close to gaussian data distributions. Later,
        to ensure an IDM follower is not perturbed by the leader, different dummy values
        will be assigned.
        """
        def fill_with_dummy(arr, indx):
            dummy_value = arr[~np.isnan(arr).any(axis=1)][:, indx].mean()
            nan_mask = np.isnan(arr[:, indx])
            nan_indx = np.where(nan_mask)
            arr[nan_indx, indx] = dummy_value
            return arr

        feature_data = fill_with_dummy(feature_data, self.indxs['leader_speed'])
        feature_data = fill_with_dummy(feature_data, self.indxs['leader_action'])
        feature_data = fill_with_dummy(feature_data, self.indxs['fl_delta_v'])
        feature_data = fill_with_dummy(feature_data, self.indxs['fl_delta_x'])

        return feature_data

    def sequence(self, feature_data, history_length, future_length):
        """
        Sequence the data into history/future sequences.
        """
        episode_ids = list(np.unique(feature_data[:, 0]))
        history_seqs, future_seqs = [], []
        for episode_id in episode_ids:
            epis_data = feature_data[feature_data[:, 0] == episode_id]
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
        col_names = ['episode_id', 'leader_speed', 'follower_speed', 'merger_speed', \
                 'leader_action', 'follower_action', 'merger_action', \
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x', \
                 'lane_y', 'leader_exists']
        history_sca = history_seqs_scaled[:, :, self.names_to_index(col_names)]
        future_sca = future_seqs_scaled[:, :, self.names_to_index(col_names)]

        #  history+future info for debugging/ visualisation
        col_names = ['episode_id', 'elapsed_time', 'ego_decision', \
                'leader_action', 'follower_action', 'merger_action', \
                'lane_y', 'follower_aggress', \
                'follower_atten', 'veh_id', 'follower_id']

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

    def scale_data(self, feature_data):
        col_names = ['leader_speed', 'follower_speed', 'merger_speed', \
                 'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']

        scalar_indexs = self.names_to_index(col_names)
        scaler = preprocessing.StandardScaler().fit(feature_data[:, scalar_indexs])
        feature_data_scaled = feature_data.copy()
        feature_data_scaled[:, scalar_indexs] = scaler.transform(feature_data[:, scalar_indexs])
        return feature_data_scaled

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
        feature_data = self.extract_features(raw_recordings)
        feature_data = self.fill_missing_values(feature_data)
        feature_data_scaled = self.scale_data(feature_data)
        history_future_seqs_seqs = self.sequence(feature_data, 20, 20)
        # history_future_seqs_seqs = self.mask_steps(history_future_seqs_seqs)
        history_future_seqs_scaled = self.sequence(feature_data_scaled, 20, 20)
        # history_future_seqs_scaled = self.mask_steps(history_future_seqs_scaled)
        data_arrays = self.split_data(history_future_seqs_seqs, history_future_seqs_scaled)
        return data_arrays, raw_recordings['info']

    # def save(self):
    #     pass
