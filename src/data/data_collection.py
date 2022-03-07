import sys
sys.path.insert(0, './src')
# import os
# os.getcwd()
# os.chdir('../')
from importlib import reload
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from envs import merge
reload(merge)
from envs.merge import EnvMerge
import os
import time
import json
import matplotlib.pyplot as plt

with open('./src/envs/config.json', 'rb') as handle:
    config = json.load(handle)

env = EnvMerge(config)

def pickle_this(item, data_files_dir, item_name):
    data_files_dir += '/'+item_name+'.pickle'
    if not os.path.exists(data_files_dir):
        with open(data_files_dir, 'wb') as handle:
            pickle.dump(item, handle)
    else:
        print('This data id exists')

def prep_data(training_data):
    all_epis = np.unique(training_data[0][:, 0, 0])
    np.random.seed(2021)
    np.random.shuffle(all_epis)

    train_epis = all_epis[:int(len(all_epis)*0.7)]
    val_epis = np.setdiff1d(all_epis, train_epis)
    train_samples = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
    val_samples = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

    _, history_sca, future_sca, future_idm_s,\
                future_m_veh_c, future_e_veh_a = training_data
    train_input = [history_sca[train_samples, :, 2:],
                future_sca[train_samples, :, 2:],
                future_idm_s[train_samples, :, 2:],
                future_m_veh_c[train_samples, :, 2:],
                future_e_veh_a[train_samples, :, 2:]]

    test_input = [history_sca[val_samples, :, 2:],
                future_sca[val_samples, :, 2:],
                future_idm_s[val_samples, :, 2:],
                future_m_veh_c[val_samples, :, 2:],
                future_e_veh_a[val_samples, :, 2:]]

    return train_input, test_input
# %%
"""
Generate datan
"""
from data import merge_data_gen
reload(merge_data_gen)
from data.merge_data_gen import DataGenMerge
data_gen = DataGenMerge(env=env, episodes_n=500)
sim_data = data_gen.prep_data()
sim_data.shape

# %%
"""
Pickle generated data.
"""
data_id = '049'
dataset_name = 'sim_data_'+data_id
data_files_dir = './src/datasets/'+dataset_name
if not os.path.exists(data_files_dir):
    os.makedirs(data_files_dir)

if not os.path.exists(data_files_dir+'/sim_data.pickle'):
    with open(data_files_dir+'/sim_data.pickle', 'wb') as handle:
        pickle.dump(sim_data, handle)
else:
    print('This data id exists')

# %%
"""
Load generated data.
"""
data_id = '049'
dataset_name = 'sim_data_'+data_id
data_files_dir = './src/datasets/'+dataset_name
with open(data_files_dir+'/sim_data.pickle', 'rb') as handle:
    sim_data = pickle.load(handle)
sim_data.shape
sim_data[20, 1]
sim_data[:, 2]
sim_data[:, 2]
# %%
"""
count cars
"""
count = 0
np.unique(sim_data[sim_data[:, 0] == 1][:, 2]).shape[0]
for i in range(500):
    count += np.unique(sim_data[sim_data[:, 0] == i][:, 2]).shape[0]-2 #
count
2000 * 200 * 0.1/60**2
# %%



lens = []
for i in range(100):
    lens.append(np.unique(sim_data[sim_data[:, 0] == i][:, 1]).shape[0])
lens
plt.hist(lens, bins=50)
# %%
"""
Prepare data
"""
history_len = 30 # steps
rollout_len = 50
from data import data_prep
reload(data_prep)
from data.data_prep import DataPrep
data_prep = DataPrep()
sim_data, dummy_value_set = data_prep.fill_missing_values(sim_data)
sim_data_scaled, env_scaler, m_scaler = data_prep.scale_data(sim_data)
history_future_seqs = data_prep.sequence(sim_data, history_len, rollout_len)
history_future_seqs_scaled = data_prep.sequence(sim_data_scaled, history_len, rollout_len)
data_arrays = data_prep.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays

history_future_usc.shape
history_future_usc[:, 2]
# %%
"""
count cars
"""
count = 0
np.unique(sim_data[sim_data[:, 0] == 1][:, 2]).shape[0]
history_future_usc = history_future_usc[:, 0, 0:3]
for i in range(500):
    count += np.unique(history_future_usc[history_future_usc[:, 0] == i][:, 2]).shape[0]
count
2000 * 50 * 0.1/(60**2)
np.unique(history_future_usc[history_future_usc[:, 0] == 9][:, 2]).shape[0]
# %%


# %%
train_input, test_input = prep_data(data_arrays)
train_input[-1].shape
# %%

"""
Make displacement a target
"""
def get_target_vals(arr):
    dxs = np.zeros([arr.shape[0], rollout_len+1, 1])

    for step in range(1, rollout_len+1):
        dxs[:, step, :] = dxs[:, step-1, :] + arr[:, step-1, 1:]*0.1 \
                                                    + 0.5*arr[:, step-1, 0:1]*0.1**2
    return dxs[:, 1:, :]
dxs = get_target_vals(train_input[-1])
train_input[-1] = np.append(train_input[-1], dxs, axis=-1)

dxs = get_target_vals(test_input[-1])
test_input[-1] = np.append(test_input[-1], dxs, axis=-1)

plt.plot(train_input[-1][0, :, -1])
plt.plot(train_input[-1][1000, :, -1])
# %%
# """
# BALANCE DATA
# """
# hf_usc_indexs = {}
# col_names = [
#          'episode_id', 'time_step',
#          'e_veh_id', 'f_veh_id', 'm_veh_id',
#          'm_veh_exists', 'e_veh_att',
#          'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
#          'e_veh_action', 'f_veh_action', 'm_veh_action',
#          'aggressiveness',
#          'desired_v','desired_tgap', 'min_jamx', 'max_act', 'min_act',
#          'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x',
#          'em_delta_y', 'delta_x_to_merge']
#
# for i, item_name in enumerate(col_names):
#     hf_usc_indexs[item_name] = i
# # %%
# history_future_usc, history_sca, future_sca, future_idm_s, future_m_veh_c, future_e_veh_a = data_arrays
# balance_value = np.count_nonzero((\
#                         history_future_usc[:, :, hf_usc_indexs['e_veh_att']] == 1).any(axis=1))/\
#                         history_future_usc.shape[0]
# print(balance_value)
# cond = (history_future_usc[:, :, hf_usc_indexs['e_veh_att']] == 1).any(axis=1)
# data_arrays = [np.append(sim_dataay, sim_dataay[cond], axis=0) for sim_dataay in data_arrays]
# %%
"""
Anywhere an array is fed to a model, the data is first scaled.
Scalers are pickled here.
"""
item_name = 'env_scaler'
pickle_this(env_scaler, data_files_dir, item_name)

item_name = 'm_scaler'
pickle_this(m_scaler, data_files_dir, item_name)

item_name = 'dummy_value_set'
pickle_this(dummy_value_set, data_files_dir, item_name)


item_name = 'train_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(train_input, data_files_dir, item_name)

item_name = 'test_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(test_input, data_files_dir, item_name)

item_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(data_arrays, data_files_dir, item_name)
