import numpy as np
np.random.seed(2020)

class DataGenerator:
    def __init__(self, env, config):
        self.config = config
        self.data_frames_n = 500 # number of data samples. Not all of it is useful.
        self.env = env
        self.initiate()

    def initiate(self):
        self.env.usage = 'data generation'
        self.env.recordings = {'info':{}, 'elapsed_time':{}, 'decisions':{}, 'states':{}}
        self.indxs = {
                    'speeds':{'leader':0, 'follower':1, 'merger':2},
                    'actions':{'leader':3, 'follower':4, 'merger':5},
                    'relatives':{'follower_leader':[6, 7], 'follower_merger':[8, 9]},
                    'lane_y':10, 'leader_exists':11}


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
            follower_speed, follower_glob_x, follower_act_long = follower_s
        else:
            return

        if leader_s:
            leader_speed, leader_glob_x, leader_act_long = leader_s
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

        step_feature.extend([ego_lane_y, leader_exists])
        return self.round_scalars(step_feature)

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
                    split_indxs.append(indx+1)
        return [0] + split_indxs

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

            for i in range(len(split_indxs)-1):
                # each split forms an episode
                start_snip = split_indxs[i]
                end_snip = split_indxs[i+1]
                ego_end_decision = ego_decisions[end_snip]
                feature_data_episode = []

                for _step in range(start_snip, end_snip):
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
                            step_feature = self.get_step_feature(
                                                                veh_state['ego'],
                                                                veh_state['fr'],
                                                                veh_state['rr'])

                        elif ego_decision == -1:
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
            indx += 4 # first n=4 items are episode_id, veh_id, elapsed_time, ego_decision
            dummy_value = arr[~np.isnan(arr).any(axis=1)][:, indx].mean()
            nan_mask = np.isnan(arr[:, indx])
            nan_indx = np.where(nan_mask)
            arr[nan_indx, indx] = dummy_value
            return arr

        indx = self.indxs['speeds']['leader']
        feature_data = fill_with_dummy(feature_data, indx)
        indx = self.indxs['actions']['leader']
        feature_data = fill_with_dummy(feature_data, indx)
        indx_delta_v,  indx_delta_x = self.indxs['relatives']['follower_leader']
        feature_data = fill_with_dummy(feature_data, indx_delta_v)
        feature_data = fill_with_dummy(feature_data, indx_delta_x)

        return feature_data

    #
    #
    # def sequence(self, feature_data):
    #     """
    #     Sequence the data into history/future sequences.
    #     """
    #     episode_ids = list(np.unique(feature_data[:, 0]))
    #     for episode_id in episode_ids:

    def prep_data(self):
        raw_recordings = self.run_sim()
        feature_data = self.extract_features(raw_recordings)
        feature_data = self.fill_missing_values(feature_data)
        return feature_data

    # def split_data(self):
    #     """Spli
    #     """
    #     train_xs = []
    #     train_ys = []


    # def preprocess(self, raw_data):
    #
    #     xs, xs_scaled, merger_a, ys, info, scaler = data_generator()
    #
    #     episode_ids = list(np.unique(xs[:, 0]))
    #     seq_scaled_s_h = []
    #     scaled_seq_xs_f = []
    #     unscaled_seq_xs_f = []
    #     seq_merger_a = []
    #     seq_ys_f = []
    #
    #     pass
    #
    # def save(self):
    #     pass
