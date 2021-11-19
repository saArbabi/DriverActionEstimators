from importlib import reload
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import pickle
from envs import merge
reload(merge)
from envs.merge import EnvMerge
import os
import time
import pandas as pd
config = {'lanes_n':2,
        'lane_width':3.75, # m
        'lane_length':300 # m
        }
env = EnvMerge(config)
data_arr_indexes = {}
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
    data_arr_indexes[item_name] = index
    index += 1
data_arr_indexes['e_veh_att']
data_arr_indexes['desired_v']
# %%
"""
Generate data
"""
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge(env=env, episodes_n=300)
data_arr = data_gen.prep_data()
data_arr.shape
# %%
"""
Pickle generated data.
"""
data_id = '_024'
file_name = 'sim_data'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(data_arr, handle)
else:
    print('This data id exists')
# %%
"""
retrieve scenarios
"""
episode = 6
time_step = 111
vehicle_id = 4
array = data_arr[(data_arr[:, data_arr_indexes['episode_id']] == episode) &
                        (data_arr[:, data_arr_indexes['time_step']] == time_step) &
                        (data_arr[:, data_arr_indexes['e_veh_id']] == vehicle_id)]

pd.DataFrame(array, columns=feature_names).iloc[0]
# %%
data_arr[data_arr[:, data_arr_indexes['time_step']] > 500]
data_arr[data_arr[:, data_arr_indexes['m_veh_action']] < -3]
data_arr[data_arr[:, data_arr_indexes['e_veh_action']] < -4]
# data_arr[data_arr[:, data_arr_indexes['m_veh_id']] > 8]

# %%
veh_id = 33
for param_name in [ 'aggressiveness', 'desired_v',
                            'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    print(param_name, ' ', data_arr[data_arr[:, 2] == veh_id][0, indxs[param_name]])

# %%
"""
Data prep
"""
history_len = 20 # steps
rollout_len = 20
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge()
with open('./models/experiments/sim_data_024.pickle', 'rb') as handle:
    features = pickle.load(handle)
features, dummy_value_set = data_gen.fill_missing_values(features)
features_scaled, env_scaler, m_scaler = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, history_len, rollout_len)
history_future_seqs_scaled = data_gen.sequence(features_scaled, history_len, rollout_len)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
future_e_veh_a[:, :, -1].std()
future_e_veh_a[:, :, -1].mean()
# %%
# features[(features[:, data_arr_indexes['e_veh_att']] == 1) & (features[:, data_arr_indexes['aggressiveness']] > 0.8)][:, 0]
# _ = plt.hist(features[(features[:, data_arr_indexes['e_veh_att']] == 1)][:, data_arr_indexes['aggressiveness']], bins=150)
# %%
"""
Who are the yielding vehicles?
"""
cond = (history_future_usc[:, :, -5] == 1).any(axis=1)
att_on_vehicles = history_future_usc[cond]
att_on_vehicles.shape
_aggressiveness = att_on_vehicles[:, 0, -1]
_ = plt.hist(_aggressiveness, bins=50)

# %%
"""
For debugging - all samples
To check if actions and states match up. Note that given a traffic state, and driver attention,
actions are determinstic.
"""
np.random.seed(0)
for i in np.random.randint(0, future_idm_s.shape[0], 500):
    aggressiveness = history_future_usc[i, 0, -1]
    veh_id = history_future_usc[i, 0, 2]
    episode_id = history_future_usc[i, 0, 0]
    veh_arr = features[(features[:, 0] == episode_id) & \
                            (features[:, 2] == veh_id)]
    desired_v = veh_arr[0, data_arr_indexes['desired_v']]
    desired_tgap = veh_arr[0, data_arr_indexes['desired_tgap']]
    min_jamx = veh_arr[0, data_arr_indexes['min_jamx']]
    max_act = veh_arr[0, data_arr_indexes['max_act']]
    min_act = veh_arr[0, data_arr_indexes['min_act']]

    vel = future_idm_s[i, :, 2]
    f_veh_v = future_idm_s[i, :, 3]
    m_veh_v = future_idm_s[i, :, 4]
    e_veh_glob_x = future_idm_s[i, :, 5]
    f_veh_glob_x = future_idm_s[i, :, 6]
    m_veh_glob_x = future_idm_s[i, :, 7]
    f_veh_exists = future_idm_s[i, :, -2]
    m_veh_exists = future_idm_s[i, :, -1]

    dv = (vel - f_veh_v)
    dx = (f_veh_glob_x - e_veh_glob_x)

    desired_gap = min_jamx + \
    np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

    ef_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)

    dv = (vel - m_veh_v)
    dx = (m_veh_glob_x - e_veh_glob_x)
    desired_gap = min_jamx + \
    np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

    em_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
    att_scores = future_idm_s[i, :, -3]
    act = (1-att_scores)*ef_act + att_scores*em_act
    # features = features[features[:, 6]==0] # merger exists
    loss = abs(act-future_e_veh_a[i, :, -1])
    if not loss.max() < 0.00001:
        print('sample-i: :  ', i)
        print(loss.max())
        plt.figure()
        plt.plot(act)
        plt.plot(future_e_veh_a[i, :, -1])
        plt.legend(['here_act', 'sim_gen_act'])
        break

# plt.plot(act)
# plt.plot(future_e_veh_a[3567, :, -1])
# plt.plot(att_scores)
# plt.plot(m_veh_exists)
#
# %%
"""
To get a sense of what action profiles are present in the dataset.
Once vehicles reach a stable point, they follow a constant velocity (this is nature of IDM)
"""
set_size = future_idm_s.shape[0]
for i in np.random.randint(0, future_idm_s.shape[0], 50):
    plt.figure()
    random_indx = np.random.randint(0, set_size)
    plt.plot(future_e_veh_a[random_indx, :, -1])
    plt.ylim([-3, 3])
    title = 'epis: '+str(history_future_usc[random_indx, 0, 0])+'   ' + \
                'time: ' + str(history_future_usc[random_indx, 0, 1])+'   ' + \
                'veh_id: ' + str(history_future_usc[random_indx, 0, 2])+'   ' + \
                'agg: ' + str(history_future_usc[random_indx, 0, -1].round(2))
    plt.title(title)
    plt.grid()
# %%

"""
EPISODE EVALUATION
Used this to plot vehicle states - for debugging
"""
# %%
np.unique(features[features[:, 2] == 19][:, 0])
# features[features[:, 2] == 34]
veh_arr[:, -1]
veh_arr[:, data_arr_indexes['time_step']]
plt.scatter(veh_arr[:, data_arr_indexes['time_step']], veh_arr[:, data_arr_indexes['time_step']])
np.where(veh_arr[:, data_arr_indexes['e_veh_att']] == 1)
veh_arr[:, data_arr_indexes['f_veh_id']]
veh_arr[:, data_arr_indexes['em_delta_y']][13]
veh_arr[:, data_arr_indexes['em_delta_y']][85+39]
future_m_veh_c[37964, :, -1]
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_list
history_future_usc[37964, :, -6]

veh_arr[:, data_arr_indexes['e_veh_decision']]
veh_arr[:, data_arr_indexes['m_veh_id']]
veh_arr[:, data_arr_indexes['f_veh_id']]
veh_arr[:, data_arr_indexes['e_veh_id']]
veh_arr[:, data_arr_indexes['m_veh_action']]
# veh_arr[:, data_arr_indexes['e_veh_att']][25]
# %%
veh_arr = features[features[:, 0] == 164]
time_snap_start = veh_arr[0, 1]
time_snap_1 = 1840
time_snap_2 = time_snap_1+60
for i in range(veh_arr.shape[-1]):
    plt.figure(figsize=(3, 3))
    plt.plot(veh_arr[:, 1], veh_arr[:, i])
    plt.plot([time_snap_1, time_snap_1],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.plot([time_snap_2, time_snap_2],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.plot([time_snap_start, time_snap_start],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.title(feature_names[i])
    plt.grid()

# %%
col_names =[
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'm_veh_exists', 'e_veh_att',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'e_veh_action', 'f_veh_action', 'm_veh_action',
         'aggressiveness',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
         'em_delta_y', 'delta_x_to_merge']
# np.count_nonzero(history_future_usc[:, :, 6] == 0)
_sample_indxs = np.random.randint(0, history_future_usc.shape[0], 5000)
for i in range(history_future_usc.shape[-1]):
    plt.figure(figsize=(5, 3))
    to_plot = history_future_usc[_sample_indxs, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%
col_names = ['episode_id', 'time_step',
        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
        'el_delta_v', 'el_delta_x',
        'em_delta_v', 'em_delta_x',
        'delta_x_to_merge',
        'em_delta_y', 'm_veh_exists']

for i in range(future_sca.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_sca[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
