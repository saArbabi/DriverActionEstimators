from collections import deque
from sklearn import preprocessing
import numpy as np

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

config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 256,
    },
    "exp_id": "NA",
    "Note": ""
}

"""
Synthetic data generation
"""
def get_idm_params(driver_type):
    if driver_type == 'normal':
        idm_param = normal_idm
    if driver_type == 'timid':
        idm_param = timid_idm
    if driver_type == 'aggressive':
        idm_param = aggressive_idm

    desired_v = idm_param['desired_v']
    desired_tgap = idm_param['desired_tgap']
    min_jamx = idm_param['min_jamx']
    max_act = idm_param['max_act']
    min_act = idm_param['min_act']

    return [desired_v, desired_tgap, min_jamx, max_act, min_act]

def get_alpha(dy):
    if dy < -1.85:
        return 1
    else:
        mean = abs(dy)/1.85
        # alpha = np.random.normal(mean, 0.1, 1)
        return mean
        # return np.clip(alpha, 0, 1).tolist()[0]

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
    if dx < 0.5:
        return 0, 0.5
    return dv, dx

def get_random_vals(mean_vel):
    init_v = 20 + np.random.choice(range(-3, 3))
    action_magnitute = np.random.uniform(0, 3)
    action_freq = np.random.uniform(0.02, 0.06)
    return init_v, action_magnitute, action_freq

def data_generator():
    xs = []
    ys = []
    info = {}
    episode_steps_n = 100
    drivers = ['normal', 'timid', 'aggressive']
    # drivers = ['normal']
    # drivers = ['aggressive']
    episode_id = 0
    episode_n = 100 * 3


    while episode_id < episode_n:
        for driver in drivers:
            idm_params = get_idm_params(driver)
            mean_vel = 20
            att_switch_step = np.random.choice(range(0, episode_steps_n))
            # sim initializations
            # follower
            f_x = np.random.choice(range(30, 50))
            f_v = mean_vel + np.random.choice(range(-3, 3))
            # leader
            l_x = 100
            l_v, l_act_mag, l_sin_freq = get_random_vals(mean_vel)
            # merger
            m_x = np.random.choice(range(40, l_x))
            m_v, m_act_mag, m_sin_freq = get_random_vals(mean_vel)

            # dy = 0
            m_vlat = 0

            for time_step in range(episode_steps_n):
                # leader
                fl_dv, lf_dx = get_relative_states(l_x, l_v, f_x, f_v)
                if lf_dx != 0.5:
                    fl_act = idm_act(f_v, fl_dv, lf_dx, idm_params)
                    l_v = l_v + l_act_mag*np.sin(l_x*l_sin_freq) * 0.1
                    l_x = l_x + l_v * 0.1
                    leader_feature = [l_v, fl_dv, lf_dx]

                else:
                    print("Collosion with lead vehicle")
                    break

                # merger
                fm_dv, mf_dx = get_relative_states(m_x, m_v, f_x, f_v)
                if mf_dx != 0.5 and m_x < l_x:
                    fm_act = idm_act(f_v, fm_dv, mf_dx, idm_params)
                    m_v = m_v + m_act_mag*np.sin(m_x*m_sin_freq) * 0.1
                    m_x = m_x + m_v * 0.1
                    m_exists = 1
                    merger_feature = [m_v, fm_dv, mf_dx]

                else:
                    m_exists = 0
                    merger_feature = [20, 0, 30]


                # merger_feature = [19, 1, 55]
                # if time_step > att_switch_step and dy > -3.7:
                #     m_vlat = -1
                # elif dy < -3.7:
                #     m_vlat = w
                # dy += m_vlat*0.1

                # follower
                # alpha = 0
                # alpha = get_alpha(dy)
                # act = (1-alpha)*fl_act + (alpha)*fm_act
                if m_exists == 1 and abs(fm_act) < 3:
                    act = fm_act
                    attention = 0

                else:
                    act = fl_act
                    attention = 1
                # act = fl_act
                # attention = 1
                if abs(act) > 3.5:
                    print("Bad state - action_value: ", act)
                    break

                f_v = f_v + act * 0.1
                f_x = f_x + f_v * 0.1 + 0.5 * act * 0.1 **2

                feature = [episode_id, m_exists, attention, f_v]
                feature.extend(leader_feature)
                feature.extend(merger_feature)
                xs.append(feature)
                ys.append([episode_id, act])

                info[episode_id] = episode_id
            episode_id += 1
    xs = np.array(xs)
    # scale_data = False
    scale_data = True

    if scale_data:
        bool_indx = 3 # these are boolians to be ignored by scaler
        scaler = preprocessing.StandardScaler().fit(xs[:, 3:])
        xs_scaled = xs.copy()
        xs_scaled[:, bool_indx:] = scaler.transform(xs[:, bool_indx:]).tolist()

        return xs, xs_scaled, np.array(ys), info, scaler

    else:
        return xs, xs, np.array(ys), info, None

def seqseq_sequence(training_states, h_len, f_len):
    scaled_ss, unscaled_ss, actions = training_states
    xs_h = [] # history, scaled
    scaled_xs_f = [] # future, scaled
    unscaled_xs_f = [] # future, not scaled
    ys_f = [] # future, not scaled
    episode_steps_n = len(scaled_ss)
    xs_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        xs_h_seq.append(scaled_ss[i])
        if len(xs_h_seq) == h_len:
            indx = i + f_len
            if indx > episode_steps_n:
                break

            xs_h.append(list(xs_h_seq))
            # xs_h.append(np.array(xs_h_seq))
            scaled_xs_f.append(scaled_ss[i:indx])
            unscaled_xs_f.append(unscaled_ss[i:indx])
            ys_f.append(actions[i:indx])

    return xs_h, scaled_xs_f, unscaled_xs_f, ys_f

def seq_sequence(training_states, h_len):
    states_h, states_c, actions = training_states
    xs_h = []
    xs_c = []
    ys_c = []
    episode_steps_n = len(states_h)
    xs_h_seq = deque(maxlen=h_len)

    for i in range(episode_steps_n):
        xs_h_seq.append(states_h[i])
        if len(xs_h_seq) == h_len:
            xs_h.append(list(xs_h_seq))
            xs_c.append(states_c[i])
            ys_c.append(actions[i])

    return xs_h, xs_c, ys_c

def dnn_prep(training_samples_n):
    _, xs_scaled, ys, _, scalar = data_generator()
    return [xs_scaled[:training_samples_n,:], ys[:training_samples_n,:]]

def seq_prep(h_len, training_samples_n):
    xs, xs_scaled, ys, _, scaler = data_generator()

    episode_ids = list(np.unique(xs[:, 0]))
    seq_xs_h = []
    seq_xs_c = []
    seq_ys_c = []
    for episode_id in episode_ids:
        if len(seq_xs_h) >= training_samples_n:
            break
        xs_id = xs[xs[:,0]==episode_id].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id].tolist()
        ys_id = ys[ys[:,0]==episode_id].tolist()

        xs_h, xs_c, ys_c = seq_sequence([xs_scaled_id, xs_id, ys_id], h_len)
        seq_xs_h.extend(xs_h)
        seq_xs_c.extend(xs_c)
        seq_ys_c.extend(ys_c)

    return [np.array(seq_xs_h), np.array(seq_xs_c), np.array(seq_ys_c)]

def seqseq_prep(h_len, f_len, training_samples_n):
    xs, xs_scaled, ys, info, scaler = data_generator()
    episode_ids = list(np.unique(xs[:, 0]))
    seq_xs_h = []
    scaled_seq_xs_f = []
    unscaled_seq_xs_f = []
    seq_ys_f = []

    for episode_id in episode_ids:
        if len(seq_xs_h) >= training_samples_n:
            break
        xs_id = xs[xs[:,0]==episode_id].tolist()
        xs_scaled_id = xs_scaled[xs_scaled[:,0]==episode_id].tolist()
        ys_id = ys[ys[:,0]==episode_id].tolist()
        xs_h, scaled_xs_f, unscaled_xs_f, ys_f = seqseq_sequence([xs_scaled_id, xs_id, ys_id], h_len, f_len)
        seq_xs_h.extend(xs_h)
        scaled_seq_xs_f.extend(scaled_xs_f)
        unscaled_seq_xs_f.extend(unscaled_xs_f)
        seq_ys_f.extend(ys_f)

    return [np.array(seq_xs_h), np.array(scaled_seq_xs_f), np.array(unscaled_seq_xs_f), \
                                                np.array(seq_ys_f)], info, scaler
