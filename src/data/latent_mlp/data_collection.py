import sys
sys.path.insert(0, './src')
# import os
os.getcwd()
# os.chdir('./DriverActionEstimators')
from importlib import reload
import numpy as np
np.set_printoptions(suppress=True)
import pickle
import os

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

    _, history_sca, future_idm_s, future_m_veh_c, history_e_veh_a = training_data

    train_input = [history_sca[train_samples, :, 2:],
                history_e_veh_a[train_samples, :, 2:]]

    val_input = [history_sca[val_samples, :, 2:],
                history_e_veh_a[val_samples, :, 2:]]

    return train_input, val_input
# %%
"""
Load generated data.
"""
data_id = '045'
dataset_name = 'sim_data_'+data_id
data_files_dir = './src/datasets/'+dataset_name
with open(data_files_dir+'/sim_data.pickle', 'rb') as handle:
    sim_data = pickle.load(handle)
sim_data.shape

# %%

plt.hist(size)
np.array(size)[:, 1].mean()
np.array(size)[:, 1].max()
np.array(size).max()
# %%
"""
Prepare data
"""
history_len = 20 # steps
rollout_len = 30
from data.latent_mlp import data_prep
reload(data_prep)
from data.latent_mlp.data_prep import DataPrepLatentMLP
data_prep = DataPrepLatentMLP()
sim_data, dummy_value_set = data_prep.fill_missing_values(sim_data)
sim_data_scaled, env_scaler, m_scaler = data_prep.scale_data(sim_data)
history_future_seqs = data_prep.sequence(sim_data, history_len, rollout_len)
history_future_seqs_scaled = data_prep.sequence(sim_data_scaled, history_len, rollout_len)
data_arrays = data_prep.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, \
                future_idm_s, future_m_veh_c, history_e_veh_a = data_arrays
# %%
train_input, val_input = prep_data(data_arrays)
train_input[0].shape
train_input[1].shape
# %%
data_files_dir += '/latent_mlp'
if not os.path.exists(data_files_dir):
    os.makedirs(data_files_dir)

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
