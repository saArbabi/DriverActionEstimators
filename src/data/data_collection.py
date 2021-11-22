import sys
sys.path.insert(0, './src')
# import os
# os.getcwd()
# os.chdir('../')
from envs import merge
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

def pickle_this(item, data_files_dir, item_name):
    data_files_dir += item_name+'.pickle'
    if not os.path.exists(data_files_dir):
        with open(data_files_dir, 'wb') as handle:
            pickle.dump(item, handle)
    else:
        print('This data id exists')

# %%
"""
Generate data
"""
from data import merge_data_gen
reload(merge_data_gen)
from data.merge_data_gen import DataGenMerge
data_gen = DataGenMerge(env=env, episodes_n=300)
data_arr = data_gen.prep_data()
data_arr.shape

# %%
"""
Pickle generated data.
"""
data_id = '_025'
file_name = 'sim_data'+data_id
file_address = './src/models/experiments/data_files/'+file_name
if not os.path.exists(file_address):
    os.makedirs(file_address)

if not os.path.exists(file_address+'/sim_data.pickle'):
    with open(file_address+'/sim_data.pickle', 'wb') as handle:
        pickle.dump(data_arr, handle)
else:
    print('This data id exists')

# %%
"""
Data prep
"""
history_len = 20 # steps
rollout_len = 20
from data import data_prep
reload(data_prep)
from data.data_prep import DataPrep
data_prep = DataPrep()
data_id = '_025'
file_name = 'sim_data'+data_id
data_files_dir = './src/models/experiments/data_files/'+file_name 
with open(data_files_dir+'/sim_data.pickle', 'rb') as handle:
    sim_data = pickle.load(handle)
sim_data, dummy_value_set = data_prep.fill_missing_values(sim_data)
sim_data_scaled, env_scaler, m_scaler = data_prep.scale_data(sim_data)
history_future_seqs = data_prep.sequence(sim_data, history_len, rollout_len)
history_future_seqs_scaled = data_prep.sequence(sim_data_scaled, history_len, rollout_len)
data_arrays = data_prep.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_arrays
future_e_veh_a[:, :, -1].std()
future_e_veh_a[:, :, -1].mean()
# %%
# """
# BALANCE DATA
# """
# history_future_usc, history_sca, future_sca, future_idm_s, future_m_veh_c, future_e_veh_a = data_arrays
# balance_value = np.count_nonzero((history_future_usc[:, :, -14] == 1).all(axis=1))/\
#                         history_future_usc.shape[0]
# print(balance_value)
# cond = (history_future_usc[:, :, -14] == 1).all(axis=1)
# data_arrays = [np.append(data_array, data_array[cond], axis=0) for data_array in data_arrays]
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

item_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                            history_len=20, rollout_len=20)

pickle_this(data_arrays, data_files_dir, item_name)
