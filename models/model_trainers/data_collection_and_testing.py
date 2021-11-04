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
# viewer = Viewer(config)
data_config = {
                # 'future_scaeq_length':40,
                'history_scaeq_length':20,
                'episodes_n':150,
                'model_type':'belief_net'
                }

indxs = {}
feature_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id', 'mf_veh_id',
         'e_veh_decision', 'e_veh_lane',
         'f_veh_exists', 'm_veh_exists', 'mf_veh_exists', 'e_veh_att',
         'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed', 'mf_veh_speed',
         'e_veh_action', 'f_veh_action', 'm_veh_action',
         'aggressiveness', 'desired_v',
         'desired_tgap', 'min_jamx', 'max_act', 'min_act',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x', 'em_delta_y',
         'mmf_delta_v', 'mmf_delta_x']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1
indxs['e_veh_att']
indxs['desired_v']
# features_origin[features_origin[:, 0] == 102][0, indxs['desired_v']]
# features_origin[features_origin[:, 0] == 102][0, indxs['desired_tgap']]
# %%

# %%

"""
Generate data
"""
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge(env, data_config)
# np.random.seed(0)
features_origin = data_gen.prep_data()
# features_origin.shape
# features_origin = features_origin[features_origin[:, indxs['m_veh_exists']] == 1]
features_origin[0, :]
# df = pd.DataFrame(features_origin[100:101, :], columns=feature_names)
# df.iloc[0]
features_origin.shape
features_origin.shape

# %%
"""
retrieve scenarios
"""
episode = 6
time_step = 111
vehicle_id = 4
array = features_origin[(features_origin[:, indxs['episode_id']] == episode) &
                        (features_origin[:, indxs['time_step']] == time_step) &
                        (features_origin[:, indxs['e_veh_id']] == vehicle_id)]

pd.DataFrame(array, columns=feature_names).iloc[0]
# %%

array = features_origin[(features_origin[:, indxs['episode_id']] == episode) &
                (features_origin[:, indxs['e_veh_id']] == vehicle_id)][:, indxs['m_veh_id']]

plt.plot(array[24:])
# %%

# %%
_ = plt.hist(features_origin[:, indxs['e_veh_speed']], bins=150)
plt.figure()
_ = plt.hist(features_origin[:, indxs['e_veh_action']], bins=150)

# %%
features_origin.shape
features_origin.shape
features_origin[features_origin[:, indxs['time_step']] > 500]
features_origin[features_origin[:, indxs['m_veh_action']] < -10]
features_origin[features_origin[:, indxs['e_veh_speed']] < -0]
features_origin[-1, :]
# features_origin[:, indxs['m_veh_action']].mean()


# %%
veh_id = 33
for param_name in [ 'aggressiveness', 'desired_v',
                            'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    print(param_name, ' ', features_origin[features_origin[:, 2] == veh_id][0, indxs[param_name]])
# %%
data_id = '_023'
file_name = 'sim_data'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(features_origin, handle)
else:
    print('This data id exists')

# %%

epis = 95
veh_id = 5
features_origin[(features_origin[:, 0] == epis) & \
                            (features_origin[:, 2] == veh_id)][0, indxs['desired_v']]

# %%

features_origin[features_origin[:, 0] == epis][0, indxs['desired_tgap']]
features_origin[features_origin[:, 0] == epis][0, indxs['max_act']]
features_origin[features_origin[:, 0] == epis][0, indxs['aggressiveness']]
features_origin[features_origin[:, 0] == epis][0, indxs['min_jamx']]

# %%
for veh_id in range(2, 8):
    plt.figure()
    veh_arr = features_origin[(features_origin[:, 0] == 2) & (features_origin[:, 2] == veh_id)]
    plt.plot(veh_arr[:, indxs['e_veh_action']])
    til = str(veh_id)+' agg: '+str(veh_arr[0, indxs['aggressiveness']])
    plt.title(til)


# %%
"""
Driver model - neural idm
"""
features = features_origin.copy()
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge(env, data_config)
# features = features[features[:, indxs['aggressiveness']] == 0.5]
# features[features[:, indxs['m_veh_exists']] == 1].shape
features, dummy_value_set = data_gen.fill_missing_values(features)
features_scaled, env_scaler, m_scaler = data_gen.scale_data(features)

history_future_seqs = data_gen.sequence(features, 20, 20)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 20)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
future_m_veh_c[future_m_veh_c[:, :, 2] == 1]

# data_arrays = [np.nan_to_num(data_array, 0) for data_array in data_arrays]
future_m_veh_c.shape
future_m_veh_c.shape
# %%

history_future_usc[:,0,:]
history_future_usc[:,0,0].min()
features[:, indxs['e_veh_action']].var()
features[:, indxs['em_delta_x']].min()
features[features[:, indxs['e_veh_att']] == 1][:, indxs['em_delta_x']].min()
features[features[:, indxs['m_veh_id']] == ][:, indxs['em_delta_x']].min()
features[(features[:, indxs['m_veh_id']] == -1) & (features[:, indxs['mf_veh_id']] == 1)]


features[features[:, indxs['em_delta_x']] < 14]


[:, indxs['em_delta_x']].min()

[:, indxs['em_delta_x']].min()
future_idm_s[0, 0, :]
future_idm_s[1, 0, :]
future_sca.shape
future_e_veh_a[:, :, -1].std()
future_e_veh_a[:, :, -1].mean()
features[features[:, indxs['em_delta_x']] < 0.1]

# %%

"""
Driver model - lstm
"""
history_future_seqs = data_gen.sequence(features, 20, 1)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 1)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays

future_e_veh_a.shape
# %%
"""
Driver model - mlp
"""
history_future_seqs = data_gen.sequence(features, 1, 1)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 1, 1)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays

future_e_veh_a.shape

history_sca.flatten().shape
future_e_veh_a[0]
history_future_usc[0]


#
# # %%
# np.count_nonzero((history_future_usc[:, :, -3] == 1).any(axis=1))/history_future_usc.shape[0]
#
# a = np.zeros([100, 20, 3])
# a[34:36, 3:5, 1] = 1
# (a[:, :, 1] == 1).any(axis=1).shape
# %%
"""
Number of episodes with a merging vehicle
"""
all_epis = np.unique(features[:, 0])
lc_epis = []
for _epis in all_epis:
    veh_arr = features[features[:, 0] == _epis]
    if veh_arr[:, indxs['e_veh_att']].mean() > 0:
        lc_epis.append(_epis)
len(lc_epis)/len(all_epis)
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
"""
# with open('./models/experiments/sim_data.pickle', 'rb') as handle:
#     features = pickle.load(handle)
# for i in range(future_idm_s.shape[0]):
# history_future_usc[i, :, 1]
# history_future_usc[i, :, 1]
# plt.plot(future_idm_s[5317, :, 1])
# plt.plot(history_future_usc[5317, :, 8])
# plt.plot(history_future_usc[5317, :, 8])
# for i in [4522]:

for i in np.random.randint(0, future_idm_s.shape[0], 5000):

        aggressiveness = history_future_usc[i, 0, -1]
        veh_id = history_future_usc[i, 0, 2]
        episode_id = history_future_usc[i, 0, 0]
        veh_arr = features[(features[:, 0] == episode_id) & \
                                (features[:, 2] == veh_id)]
        desired_v = veh_arr[0, indxs['desired_v']]
        desired_tgap = veh_arr[0, indxs['desired_tgap']]
        min_jamx = veh_arr[0, indxs['min_jamx']]
        max_act = veh_arr[0, indxs['max_act']]
        min_act = veh_arr[0, indxs['min_act']]

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

# plt.plot(act)
# plt.plot(future_e_veh_a[3567, :, -1])
# plt.plot(att_scores)
# plt.plot(m_veh_exists)
#
# %%
plt.plot()
history_future_usc[3567, 0, :]

# %%
"""
To get a sense of what action profiles are present in the dataset.
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

# %%

"""
EPISODE EVALUATION
"""
# %%
np.unique(features[features[:, 2] == 19][:, 0])
# features[features[:, 2] == 34]
veh_arr[:, -1]
veh_arr[:, indxs['time_step']]
plt.scatter(veh_arr[:, indxs['time_step']], veh_arr[:, indxs['time_step']])
np.where(veh_arr[:, indxs['e_veh_att']] == 1)
veh_arr[:, indxs['f_veh_id']]
veh_arr[:, indxs['em_delta_y']][13]
veh_arr[:, indxs['em_delta_y']][85+39]
future_m_veh_c[37964, :, -1]
history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
history_future_usc[37964, :, -6]

veh_arr[:, indxs['e_veh_decision']]
veh_arr[:, indxs['m_veh_id']]
veh_arr[:, indxs['f_veh_id']]
veh_arr[:, indxs['e_veh_id']]
veh_arr[:, indxs['m_veh_action']]
# veh_arr[:, indxs['e_veh_att']][25]
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
a = []
def add_to_list(this_list):
    this_list.append(5)
add_to_list(a)
a
# %%
round(0.111, 1)
"""
RAW FEATURE ANAL
"""
for i in range(features.shape[-1]):
    plt.figure(figsize=(3, 3))
    _ = plt.hist(features[:, i], bins=150)
    plt.title(feature_names[i])
    plt.grid()

# %%


col_names = ['episode_id', 'time_step',
        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
        'ef_delta_v', 'ef_delta_x', 'em_delta_v', 'em_delta_x',
        'em_delta_y', 'f_veh_exists', 'm_veh_exists']

for i in range(future_sca.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_sca[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%
col_names = ['episode_id', 'time_step', 'em_delta_y', 'm_veh_action',\
                                            'f_veh_exists', 'm_veh_exists']
for i in range(future_m_veh_c.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = future_m_veh_c[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
# %%

col_names = ['episode_id', 'time_step', 'e_veh_id', 'm_veh_id',
        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
        'e_veh_action', 'f_veh_action', 'm_veh_action',
        'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
        'em_delta_y', 'e_veh_att', 'f_veh_exists', 'm_veh_exists', 'mf_veh_exists',
        'e_veh_decision', 'aggressiveness']
# np.count_nonzero(history_future_usc[:, :, 6] == 0)

for i in range(history_future_usc.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_future_usc[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%
col_names = ['episode_id', 'time_step',
                'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
                'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x']

for i in range(future_idm_s.shape[-1]):
    plt.figure(figsize=(3, 3))
    _ = plt.hist(future_idm_s[:, -1, i], bins=150)
    plt.title(col_names[i])
    plt.grid()
# %%

# %%
