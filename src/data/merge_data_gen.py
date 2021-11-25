import numpy as np

class DataGenMerge():
    def __init__(self, env, episodes_n):
        self.episodes_n = episodes_n
        self.env = env
        self.env.usage = 'data generation'
        self.env.recordings = {}
        self.env.veh_log = ['lane_decision', 'lane_id', 'glob_x', \
                                 'glob_y', 'speed', 'act_long']
        self.ego_internal_state = ['aggressiveness', 'desired_v', \
                'desired_tgap', 'min_jamx', 'max_act', 'min_act']

    def is_episode_complete(self):
        """Episode is considered complete if all
            cars exit the road.
        """
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
        return np.float32(features)

    def prep_data(self):
        raw_recordings = self.run_sim()
        features = self.extract_features(raw_recordings)
        return features
