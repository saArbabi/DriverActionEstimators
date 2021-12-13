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
    train_examples = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
    val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

    _, history_sca, future_sca, future_idm_s,\
                future_m_veh_c, future_e_veh_a = training_data
    train_input = [history_sca[train_examples, :, 2:],
                future_sca[train_examples, :, 2:],
                future_idm_s[train_examples, :, 2:],
                future_m_veh_c[train_examples, :, 2:],
                future_e_veh_a[train_examples, :, 2:]]

    val_input = [history_sca[val_indxs, :, 2:],
                future_sca[val_indxs, :, 2:],
                future_idm_s[val_indxs, :, 2:],
                future_m_veh_c[val_indxs, :, 2:],
                future_e_veh_a[val_indxs, :, 2:]]

    return train_input, val_input
# %%
"""
Generate data
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
data_id = '028'
dataset_name = 'sim_data_'+data_id
data_files_dir = './src/models/experiments/data_files/'+dataset_name
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
data_id = '028'
dataset_name = 'sim_data_'+data_id
data_files_dir = './src/models/experiments/data_files/'+dataset_name
with open(data_files_dir+'/sim_data.pickle', 'rb') as handle:
    sim_data = pickle.load(handle)
sim_data.shape
# %%
"""
Prepare data
"""
history_len = 50 # steps
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
future_e_veh_a[:, :, -1].std()
future_e_veh_a[:, :, -1].mean()
future_e_veh_a.shape
# %%
train_input, val_input = prep_data(data_arrays)

# %%

# """
# BALANCE DATA
# """
# history_future_usc, history_sca, future_sca, future_idm_s, future_m_veh_c, future_e_veh_a = data_arrays
# balance_value = np.count_nonzero((history_future_usc[:, :, -14] == 1).all(axis=1))/\
#                         history_future_usc.shape[0]
# print(balance_value)
# cond = (history_future_usc[:, :, -14] == 1).all(axis=1)
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

item_name = 'data_arrays_h{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(data_arrays, data_files_dir, item_name)
# (841,502,560 bytes)

item_name = 'train_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(train_input, data_files_dir, item_name)

item_name = 'val_input{history_len}_f{rollout_len}'.format(\
                                history_len=history_len, rollout_len=rollout_len)

pickle_this(val_input, data_files_dir, item_name)
