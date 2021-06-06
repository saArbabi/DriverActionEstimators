from collections import deque
from sklearn import preprocessing
import numpy as np
np.random.seed(2020)
normal_idm = {
                'desired_v':25, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_act':1.4, # m/s^2
                'min_act':2, # m/s^2
                }

timid_idm = {
                'desired_v':19.4, # m/s
                'desired_tgap':2, # s
                'min_jamx':4, # m
                'max_act':0.8, # m/s^2
                'min_act':1, # m/s^2
                }

aggressive_idm = {
                'desired_v':30, # m/s
                'desired_tgap':1, # s
                'min_jamx':0, # m
                'max_act':2, # m/s^2
                'min_act':3, # m/s^2
                }

"""
Synthetic data generation
"""

def get_idm_params(driver_type):
    vel_noise = np.random.normal(0, 1)

    if driver_type == 'normal':
        idm_param = normal_idm
    if driver_type == 'timid':
        idm_param = timid_idm
    if driver_type == 'aggressive':
        idm_param = aggressive_idm

    desired_v = idm_param['desired_v'] + vel_noise
    desired_tgap = idm_param['desired_tgap']
    min_jamx = idm_param['min_jamx']
    max_act = idm_param['max_act']
    min_act = idm_param['min_act']

    return [desired_v, desired_tgap, min_jamx, max_act, min_act]

def idm_act(_v, _dv, _dx, idm_params):
    desired_v, desired_tgap, min_jamx, max_act, min_act = idm_params
    desired_gap = min_jamx + desired_tgap*_v+(_v*_dv)/ \
                                    (2*np.sqrt(max_act*min_act))

    act = max_act*(1-(_v/desired_v)**4-\
                                        (desired_gap/_dx)**2)
    return act

def get_relative_states(front_x, front_v, rear_x, rear_v):
    """
    front vehicle and rear vehicle
    """
    dx = front_x - rear_x
    dv = rear_v - front_v
    head_way = dx/rear_v
    if head_way < .5:
        return dv, dx, 'unsafe'
    return dv, dx, 'safe'

def get_random_vals(mean_vel):
    init_v = 20 + np.random.choice(range(-3, 3))
    action_magnitute = np.random.uniform(0, 3)
    action_freq = np.random.uniform(0.02, 0.06)
    return init_v, action_magnitute, action_freq

def data_generator():
    xs = []
    ys = []
    merger_a = []
    info = {}
    episode_steps_n = 100
    # drivers = ['normal', 'timid', 'aggressive']
    drivers = ['normal']
    # drivers = ['aggressive']
    episode_id = 0
    episode_n = 100 * 2
    # episode_n = 100 * 4
    step_size = 0.1 #s
    lane_width = 1.85
    attentiveness = {'timid': [4, 30], 'normal': [45, 45], 'aggressive': [30, 4]} # attention probabilities

    while episode_id < episode_n:
        for driver in drivers:
            idm_params = get_idm_params(driver)
            mean_vel = 20
            try_lane_change_step = np.random.choice(range(0, episode_steps_n))
            a = attentiveness[driver][0]
            b = attentiveness[driver][1]
            being_noticed_my = lane_width*np.random.beta(a, b)
            # sim initializations
            # follower

            f_x = np.random.choice(range(0, 30))
            f_v = mean_vel + np.random.choice(range(-3, 3))
            f_att = 'leader'
            # leader
            l_x = np.random.choice(range(70, 100))
            l_v, l_act_mag, l_sin_freq = get_random_vals(mean_vel)
            # merger
            m_x = np.random.choice(range(40, 70))
            m_y = 0 # lane relative
            m_v, m_act_mag, m_sin_freq = get_random_vals(mean_vel)
            m_vlat = 0
            lane_id = 1

            for time_step in range(episode_steps_n):
                # leader
                fl_dv, lf_dx, safety_situation = get_relative_states(l_x, l_v, f_x, f_v)
                if safety_situation == 'unsafe':
                    break
                fl_act = idm_act(f_v, fl_dv, lf_dx, idm_params)
                l_v = l_v + l_act_mag*np.sin(l_x*l_sin_freq) * step_size
                l_x = l_x + l_v * step_size
                leader_feature = [l_v, fl_dv, lf_dx]

                # merger
                fm_dv, mf_dx, safety_situation= get_relative_states(m_x, m_v, f_x, f_v)
                if mf_dx < 1:
                    break
                fm_act = idm_act(f_v, fm_dv, mf_dx, idm_params)
                if time_step > try_lane_change_step and f_att == 'leader' \
                                                and abs(fm_act) < 3 or m_vlat !=0:

                    m_vlat = -0.7
                    if abs(m_y) >= being_noticed_my or lane_id == 0:
                        f_att = 'merger'
                    # f_att = get_att_vehicle(attentiveness[driver], m_y, lane_width)
                # if f_att == 'merger':
                if lane_id == 1 and m_y <= -lane_width:
                    lane_id = 0
                    m_y = lane_width
                elif lane_id == 0 and m_y <= 0:
                    m_y = 0
                    m_vlat = 0

                m_v = m_v + m_act_mag*np.sin(m_x*m_sin_freq) * step_size
                m_x = m_x + m_v * step_size
                m_y = m_y + m_vlat * step_size

                merger_feature = [m_v, fm_dv, mf_dx, m_y]

                if f_att == 'leader':
                    act = fl_act
                else:
                    act = fm_act

                # act = fl_act
                # f_att = 1

                f_v = f_v + act * step_size
                f_x = f_x + f_v * step_size + 0.5 * act * step_size **2

                feature = [episode_id, f_v]
                feature.extend(leader_feature)
                feature.extend(merger_feature)
                feature.extend([act, 0 if f_att == 'merger' else 1])

                xs.append(feature)
                merger_a.append([episode_id, m_y, m_vlat])
                ys.append([episode_id, act])

            info[episode_id] = driver
            episode_id += 1
    xs = np.array(xs)
    # scale_data = False
    scale_data = True

    if scale_data:
        bool_indx = 1 # these are values not to be scaled
        scaler = preprocessing.StandardScaler().fit(xs[:, bool_indx:-3])
        xs_scaled = xs.copy()
        xs_scaled[:, bool_indx:-3] = scaler.transform(xs[:, bool_indx:-3]).tolist()

        return xs, xs_scaled, np.array(merger_a), np.array(ys), info, scaler

    else:
        return xs, xs, np.array(merger_a), np.array(ys), info, None

def seqseq_sequence(training_states, h_len, f_len):
    scaled_s, unscaled_s, merger_a, actions = training_states
    scaled_s_h = [] # history, scaled
    scaled_s_f = [] # future, scaled
    unscaled_s_hf = [] # history and future, not scaled
    merger_a_f = []
    ys_hf = [] # future, not scaled
    episode_steps_n = len(scaled_s)
    scaled_s_h_seq = deque(maxlen=h_len)
    unscaled_s_h_seq = deque(maxlen=h_len)
    ys_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        scaled_s_h_seq.append(scaled_s[i])
        unscaled_s_h_seq.append(unscaled_s[i])
        ys_h_seq.append(actions[i])
        if len(scaled_s_h_seq) == h_len:
            indx = i + f_len
            if indx + 1 > episode_steps_n:
                break

            scaled_s_h.append(list(scaled_s_h_seq))
            # scaled_s_h.append(np.array(scaled_s_h_seq))
            scaled_s_f.append(scaled_s[i+1:indx+1])
            unscaled_s_hf.append(list(unscaled_s_h_seq)+unscaled_s[i+1:indx+1])
            merger_a_f.append(merger_a[i+1:indx+1])
            ys_hf.append(list(ys_h_seq)+actions[i+1:indx+1])

    return scaled_s_h, scaled_s_f, unscaled_s_hf, merger_a_f, ys_hf

def seq_sequence(training_states, h_len):
    states_h, states_c, actions = training_states
    scaled_s_h = []
    xs_c = []
    ys_c = []
    episode_steps_n = len(states_h)
    scaled_s_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        scaled_s_h_seq.append(states_h[i])
        if len(scaled_s_h_seq) == h_len:
            scaled_s_h.append(list(scaled_s_h_seq))
            xs_c.append(states_c[i])
            ys_c.append(actions[i])

    return scaled_s_h, xs_c, ys_c

def dnn_prep(training_samples_n):
    _, xs_scaled, ys, _, scalar = data_generator()
    return [xs_scaled[:training_samples_n,:], ys[:training_samples_n,:]]

def seq_prep(h_len, training_samples_n):
    xs, xs_scaled, ys, _, scaler = data_generator()

    episode_ids = list(np.unique(xs[:, 0]))
    seq_scaled_s_h = []
    seq_xs_c = []
    seq_ys_c = []
    for episode_id in episode_ids:
        if len(seq_scaled_s_h) >= training_samples_n:
            break
        xs_id = xs[xs[:,0]==episode_id].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id].tolist()
        ys_id = ys[ys[:,0]==episode_id].tolist()

        scaled_s_h, xs_c, ys_c = seq_sequence([xs_scaled_id, xs_id, ys_id], h_len)
        seq_scaled_s_h.extend(scaled_s_h)
        seq_xs_c.extend(xs_c)
        seq_ys_c.extend(ys_c)

    return [np.array(seq_scaled_s_h), np.array(seq_xs_c), np.array(seq_ys_c)]

def seqseq_prep(h_len, f_len, training_samples_n):
    xs, xs_scaled, merger_a, ys, info, scaler = data_generator()
    episode_ids = list(np.unique(xs[:, 0]))
    seq_scaled_s_h = []
    scaled_seq_xs_f = []
    unscaled_seq_xs_f = []
    seq_merger_a = []
    seq_ys_f = []

    for episode_id in episode_ids:
        if len(seq_scaled_s_h) >= training_samples_n:
            break
        xs_id = xs[xs[:,0]==episode_id].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id].tolist()
        merger_a_id = merger_a[merger_a[:,0]==episode_id].tolist()
        ys_id = ys[ys[:,0]==episode_id].tolist()
        scaled_s_h, scaled_s_f, unscaled_s_f, merger_a_f, ys_f = seqseq_sequence([xs_scaled_id, xs_id, merger_a_id, ys_id], h_len, f_len)
        seq_scaled_s_h.extend(scaled_s_h)
        scaled_seq_xs_f.extend(scaled_s_f)
        unscaled_seq_xs_f.extend(unscaled_s_f)
        seq_merger_a.extend(merger_a_f)
        seq_ys_f.extend(ys_f)

    return [np.array(seq_scaled_s_h), np.array(scaled_seq_xs_f), np.array(unscaled_seq_xs_f), \
                                np.array(seq_merger_a), np.array(seq_ys_f)], info, scaler
