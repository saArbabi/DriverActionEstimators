# import os
# import pickle
import matplotlib.pyplot as plt
from importlib import reload
import sys
import data_generator
reload(data_generator)
# reload(plt)
from data_generator import DataGenerator
import numpy as np
import tensorflow as tf

import highway
reload(highway)
from highway import Env

import vehicle_handler
reload(vehicle_handler)
import time
from viewer import Viewer

config = {'lanes_n':6,
        'lane_width':3.7, # m
        'lane_length':400 # m
        }

env = Env(config)
# viewer = Viewer(config)
data_config = {
                # 'future_scaeq_length':40,
                'history_scaeq_length':20,
                'data_frames_n':100,
                'model_type':'belief_net'
                }
data_gen = DataGenerator(env, data_config)
features_origin = data_gen.prep_data()
# features_origin.shape
features_origin.shape

# %%
indxs = {}
feature_names = [
         'episode_id', 'time_step',
         'ego_id', 'leader_id', 'merger_id',
         'ego_decision', 'leader_exists', 'merger_exists',
         'aggressiveness', 'lane_y', 'ego_att',
         'ego_speed', 'leader_speed', 'merger_speed',
         'ego_action', 'leader_action', 'merger_action',
         'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1
indxs['ego_att']
# %%
np.where(features_origin[:, indxs['merger_id']] == features_origin[:, indxs['leader_id']])
features_origin[31972, 0]
# %%
features_origin.shape
features_origin[(features_origin[:, indxs['ego_att']] == 0) & \
                    (features_origin[:, indxs['leader_exists']] == 0) ].shape
features_origin[(features_origin[:, indxs['merger_exists']] == 1)]
this = features_origin[(features_origin[:, indxs['merger_exists']] == 1)]
np.unique(this[:, 0])

np.unique(this[:, 0])
# %%
features_origin[(features_origin[:, indxs['merger_id']] != -1) & \
                    (features_origin[:, indxs['leader_id']] != -1) ].shape
# %%
features_origin[(features_origin[:, indxs['ego_id']] == 59)]
# %%
features_origin[(features_origin[:, indxs['aggressiveness']] == 1) & \
                    (features_origin[:, indxs['ego_att']] == 1) ].shape
features_origin[(features_origin[:, indxs['aggressiveness']] == 0) & \
                    (features_origin[:, indxs['ego_att']] == 1) ].shape

# %%

features_origin[:, indxs['ego_action']].min()
features_origin[:, indxs['ego_action']].std()
features_origin[:, indxs['ego_action']].mean()
# %%
"""
PREPARE DATA
"""
features = features_origin.copy()
# features = features[features[:, indxs['aggressiveness']] == 0.5]
# features[features[:, indxs['merger_exists']] == 1].shape
features = data_gen.fill_missing_values(features)
features_scaled = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, 20, 20)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 20)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_merger_a, future_ego_a = data_arrays


# %%
"""
BALANCE DATA
"""
history_future_usc, history_sca, future_sca, future_idm_s, future_merger_a, future_ego_a = data_arrays
cond = (future_sca[:, :, -1] == 1).any(axis=1)
data_arrays = [np.append(data_array, data_array[cond], axis=0) for data_array in data_arrays]
np.count_nonzero(cond)/future_ego_a.shape[0]
# %%
future_ego_a.shape
future_idm_s.shape

# %%
"""
For debugging - all samples
"""

for i in range(10000000):
# for i in range(100):
    aggressiveness = history_future_usc[i, 0, -1]
    if aggressiveness == 0:
        desired_v = 19.4
        desired_tgap = 2
        min_jamx = 4
        max_act = 0.8
        min_act = 1
    elif aggressiveness == 0.5:
        desired_v = 24.7
        desired_tgap = 1.5
        min_jamx = 2
        max_act = 1.4
        min_act = 2
    elif aggressiveness == 1:
        desired_v = 30
        desired_tgap = 1
        min_jamx = 0
        max_act = 2
        min_act = 3

    vel = future_idm_s[i, :, 2]
    leader_exists = future_idm_s[i, :, -2]
    merger_exists = future_idm_s[i, :, -1]

    dv = future_idm_s[i, :, 3]*leader_exists
    dx = future_idm_s[i, :, 4]*leader_exists + 1000*(1-leader_exists)


    desired_gap = min_jamx + \
    np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

    fl_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
    fl_act = np.clip(fl_act, -3, 3)


    dv = future_idm_s[i, :, 5]*merger_exists
    dx = future_idm_s[i, :, 6]*merger_exists + 1000*(1-merger_exists)
    desired_gap = min_jamx + \
    np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

    fm_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
    fm_act = np.clip(fm_act, -3, 3)
    att_scores = history_future_usc[i, :, -5]
    act =(1-att_scores)*fl_act + att_scores*fm_act
    # features = features[features[:, 6]==0] # merger exists
    loss = abs(act-future_ego_a[i, :, -1])
    if not loss.max() < 0.001:
        print('index:  ', i)
        print(loss.max())
# %%

"""
For debugging - single sample
"""
i = 40273
# history_future_usc[i, 0, :]
aggressiveness = history_future_usc[i, 0, -1]
if aggressiveness == 0:
    desired_v = 19.4
    desired_tgap = 2
    min_jamx = 4
    max_act = 0.8
    min_act = 1
elif aggressiveness == 0.5:
    desired_v = 24.7
    desired_tgap = 1.5
    min_jamx = 2
    max_act = 1.4
    min_act = 2
elif aggressiveness == 1:
    desired_v = 30
    desired_tgap = 1
    min_jamx = 0
    max_act = 2
    min_act = 3

vel = future_idm_s[i, :, 2]
leader_v = future_idm_s[i, :, 3]
merger_v = future_idm_s[i, :, 4]
leader_exists = future_idm_s[i, :, -2]
merger_exists = future_idm_s[i, :, -1]

dv = (vel - leader_v)*leader_exists
dx = future_idm_s[i, :, 5]*leader_exists + 1000*(1-leader_exists)

desired_gap = min_jamx + \
np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

fl_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
fl_act = np.clip(fl_act, -3, 3)

dv = (vel - merger_v)*merger_exists
dx = future_idm_s[i, :, 6]*merger_exists + 1000*(1-merger_exists)
desired_gap = min_jamx + \
np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

fm_act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
fm_act = np.clip(fm_act, -3, 3)
att_scores = history_future_usc[i, :, -5]
history_future_usc[i, :, -5]
act = (1-att_scores)*fl_act + att_scores*fm_act
plt.plot(act)
# %%
att_scores_pred = np.array([1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        1.0000000e+00, 1.0000000e+00, 9.9999881e-01, 9.9386734e-01,
        8.9486158e-01, 7.3602140e-02, 5.1468670e-01, 2.8113881e-01,
        3.3656999e-01, 9.9013418e-01, 9.4325203e-01, 8.9705032e-01,
        9.8999459e-01, 9.9275535e-01, 4.8926911e-01, 9.8763371e-01,
        5.1339842e-02, 9.3817367e-04, 2.2541439e-04, 3.2734834e-03])
act_pred = (1-att_scores_pred)*fl_act + att_scores_pred*fm_act
plt.plot(act_pred)
plt.plot(act)
plt.plot(bad_act)
bad_act = np.array( [[-2.8600504 ],
        [-2.5390313 ],
        [-2.2685897 ],
        [-2.0381284 ],
        [-1.8397862 ],
        [-1.6676115 ],
        [-1.5170228 ],
        [-1.3844308 ],
        [-1.2669896 ],
        [-1.162413  ],
        [-1.0688463 ],
        [-0.98476726],
        [-0.9089168 ],
        [-0.84024453],
        [-0.77786756],
        [-0.7210376 ],
        [-0.66912824],
        [-0.6216008 ],
        [-0.57798433],
        [-0.5378689 ],
        [-0.50089854],
        [-0.46676242],
        [-0.43518108],
        [-0.3595602 ],
        [ 0.07035895],
        [ 0.06437152],
        [ 0.06745518],
        [ 0.06165537],
        [ 0.05996983],
        [ 0.1993992 ],
        [ 0.09106313],
        [ 0.06849393],
        [ 0.17215483],
        [ 0.1972932 ],
        [ 0.03617135],
        [ 0.16665894],
        [ 0.29320046],
        [ 0.28753597],
        [ 0.2819547 ],
        [ 0.27645677]])
# %%
plt.plot(att_scores_pred[28:40])
plt.plot(att_scores[28:40])
# %%


plt.plot(history_future_usc[i, :, 6])
history_future_usc
fl_act
plt.plot(history_future_usc[i, :, 7])
plt.plot(history_future_usc[i, :, 8])
history_future_usc[i, 0, :]
plt.plot(fl_act)
plt.scatter(range(40), fl_act)
plt.plot(fm_act)
# %%
# features = features[features[:, 6]==0] # merger exists
loss = abs(act-future_ego_a[i, :, -1])
if not loss.max() < 0.001:
    print('index:  ', i)
    print(loss.max())
########################################
# %%

"""
EPISODE EVALUATION
"""
# %%

# np.unique(features[features[:, 2] == 21][:, 0])

# features[features[:, 2] == 34]
veh_arr = features[features[:, 0] == 463]
veh_arr[:, indxs['time_step']]
veh_arr[:, indxs['leader_id']]

veh_arr[:, indxs['merger_id']]
veh_arr[:, indxs['ego_id']]
# veh_arr[:, indxs['ego_att']][25]
time_snap_start = veh_arr[0, 1]
time_snap_1 = 332
time_snap_2 = time_snap_1+40
for i in range(veh_arr.shape[-1]):
    plt.figure(figsize=(4, 4))
    plt.plot(veh_arr[:, 1], veh_arr[:, i])
    plt.plot([time_snap_1, time_snap_1],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.plot([time_snap_2, time_snap_2],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.plot([time_snap_start, time_snap_start],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.title(feature_names[i])
    plt.grid()


# %%

"""
RAW FEATURE ANAL
"""
for i in range(features.shape[-1]):
    plt.figure(figsize=(4, 4))
    _ = plt.hist(features[:, i], bins=150)
    plt.title(feature_names[i])
    plt.grid()

# %%


col_names = ['episode_id', 'time_step',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
        'lane_y', 'leader_exists', 'merger_exists']

for i in range(future_sca.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_sca[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%

col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']
# np.count_nonzero(history_future_usc[:, :, 6] == 0)

for i in range(history_future_usc.shape[-1]):
    plt.figure(figsize=(3, 3))
    to_plot = history_future_usc[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()
# %%


# %%

col_names = ['episode_id', 'time_step', 'ego_speed',
                'fl_delta_v', 'fl_delta_x',
                'fm_delta_v', 'fm_delta_x',
                'leader_exists', 'merger_exists']
# type(future_idm_s[i, 0, -3])

for i in range(future_idm_s.shape[-1]):
    plt.figure(figsize=(3, 3))
    _ = plt.hist(future_idm_s[:, -1, i], bins=150)
    plt.title(col_names[i])
    plt.grid()
# %%

# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 256,
    },
    "exp_id": "NA",
    "Note": ""
}

class Trainer():
    def __init__(self, model_type):
        self.model = None
        self.model_type = model_type
        self.train_loss = []
        self.valid_loss = []

        self.train_mseloss = []
        self.train_att_klloss = []
        self.train_idm_klloss = []

        self.test_mseloss = []
        self.test_att_klloss = []
        self.test_idm_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self, model_type=None):
        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModel
        self.model = NeurIDMModel(config)

    def train(self, training_data, epochs):
        self.model.epochs_n = epochs

        _, history_sca, future_sca, future_idm_s,\
                future_merger_a, future_ego_a = training_data

        all_epis = np.unique(history_sca[:, 0, 0])
        train_epis = all_epis[:int(len(all_epis)*0.8)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]

        train_input = [history_sca[train_indxs, :, 2:],
                    future_sca[train_indxs, :, 2:],
                    future_idm_s[train_indxs, :, 2:],
                    future_merger_a[train_indxs, :, 2:],
                    future_ego_a[train_indxs, :, 2:]]

        val_input = [history_sca[val_indxs, :, 2:],
                    future_sca[val_indxs, :, 2:],
                    future_idm_s[val_indxs, :, 2:],
                    future_merger_a[val_indxs, :, 2:],
                    future_ego_a[val_indxs, :, 2:]]

        avg_training_time = epochs * 100/60
        print('it will take about ' , round(avg_training_time), ' mins to complete')
        for epoch in range(epochs):
            t0 = time.time()
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'vae_idm' or self.model_type == 'driver_model':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_att_klloss.append(round(self.model.train_att_klloss.result().numpy().item(), 2))
                self.train_idm_klloss.append(round(self.model.train_idm_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_att_klloss.append(round(self.model.test_att_klloss.result().numpy().item(), 2))
                self.test_idm_klloss.append(round(self.model.test_idm_klloss.result().numpy().item(), 2))
            else:
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
            t1 = time.time()
            print(self.epoch_count, 'epochs completed')
            print('Epoch took: ', round(t1-t0), ' seconds')
            self.epoch_count += 1

    def save_model(self, model_name):
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

model_trainer = Trainer(model_type='driver_model')

# %%
model_trainer.model.vae_loss_weight = 0.1
model_trainer.train(data_arrays, epochs=5)
################## MSE LOSS ##################
fig = plt.figure(figsize=(15, 5))
plt.style.use('default')

mse_axis = fig.add_subplot(131)
att_kl_axis = fig.add_subplot(132)
idm_kl_axis = fig.add_subplot(133)
mse_axis.plot(model_trainer.test_mseloss)
mse_axis.plot(model_trainer.train_mseloss)

mse_axis.grid()
mse_axis.set_xlabel('epochs')
mse_axis.set_ylabel('loss (MSE)')
mse_axis.set_title('MSE')
mse_axis.legend(['test', 'train'])

################## att_kl LOSS ##################
att_kl_axis.plot(model_trainer.test_att_klloss)
att_kl_axis.plot(model_trainer.train_att_klloss)

att_kl_axis.grid()
att_kl_axis.set_xlabel('epochs')
att_kl_axis.set_ylabel('loss (att_kl)')
att_kl_axis.set_title('att_kl')
att_kl_axis.legend(['test', 'train'])

################## idm_kl LOSS ##################
idm_kl_axis.plot(model_trainer.test_idm_klloss)
idm_kl_axis.plot(model_trainer.train_idm_klloss)

idm_kl_axis.grid()
idm_kl_axis.set_xlabel('epochs')
idm_kl_axis.set_ylabel('loss (idm_kl)')
idm_kl_axis.set_title('idm_kl')
idm_kl_axis.legend(['test', 'train'])

# %%
all_epis = np.unique(history_sca[:, 0, 0])
train_epis = all_epis[:int(len(all_epis)*0.8)]

val_epis = np.setdiff1d(all_epis, train_epis)
train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
np.where(train_indxs == 37964)
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]

history_sca.shape
train_indxs.shape
val_examples.shape
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_merger_a = np.float32(future_merger_a)


# %%
def latent_samples(model_trainer, sample_index):
    sdv_actions = future_merger_a[sample_index, :, 2:]
    h_seq = history_sca[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    enc_acts = model_trainer.model.act_encoder(sdv_actions)
    prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
    sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_att_z, sampled_idm_z

def latent_vis():
    fig = plt.figure(figsize=(4, 4))
    # plt.style.use('ggplot')
    plt.style.use('default')
    att_axis = fig.add_subplot(211)
    idm_axis = fig.add_subplot(212)
    sampled_att_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
    aggressiveness = history_future_usc[val_examples, 0, -1]
    color_shade = aggressiveness
    att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, \
                                                c=color_shade, cmap='rainbow')
    idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, \
                                    c=color_shade, cmap='rainbow')

    att_axis.set_ylabel('$z_{att_1}$')
    att_axis.set_xlabel('$z_{att_2}$')
    idm_axis.set_ylabel('$z_{idm_1}$')
    idm_axis.set_xlabel('$z_{idm_1}$')

    return att_axis, idm_axis
att_axis, idm_axis = latent_vis()

# %%
"""
Choose cars based on the latent for debugging
"""
sampled_att_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
sampled_att_z, sampled_idm_z = sampled_att_z.numpy(), sampled_idm_z.numpy()

sampled_att_z
# %%
bad_episodes = []
bad_504 = []
bad_498 = []
bad_zs = np.where(sampled_att_z[:, 0] > 15)[0]
for bad_z in bad_zs:
    exmp_indx = val_examples[bad_z]
    epis = history_future_usc[exmp_indx, 0, 0]
    bad_episodes.append([epis, exmp_indx])
    if epis == 504:
        bad_504.append(exmp_indx)
    if epis == 498:
        bad_498.append(exmp_indx)
min(bad_504)
min(bad_498)
# val_examples[2910]
_ = plt.hist(np.array(bad_episodes)[:, 0], bins=150)

bad_episodes
history_future_usc[71538, :, 1]
history_future_usc[55293, 0, :]
plt.plot(bad_504)
plt.scatter(bad_504,bad_504)
# %%
plt.plot(history_future_usc[55300, :, -6])
71538
for bad_indx in bad_504:
    plt.figure()
    plt.plot(history_future_usc[bad_indx, :, -6])
    plt.title(bad_indx)
    plt.grid()

# %%
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def latent_samples(model_trainer, sample_index):
        sdv_actions = future_merger_a[sample_index, :, 2:]
        h_seq = history_sca[sample_index, :, 2:]
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        enc_acts = model_trainer.model.act_encoder(sdv_actions)
        prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
        sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
        return sampled_att_z, sampled_idm_z

    fig = plt.figure(figsize=(7, 7))
    plt.style.use('ggplot')
    att_axis = fig.add_subplot(211)
    idm_axis = fig.add_subplot(212)


    def animation_frame(i):
        model_trainer.model.vae_loss_weight = 0.1
        model_trainer.train(data_arrays, epochs=1)
        sampled_att_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
        aggressiveness = history_future_usc[val_examples, 0, -1]
        color_shade = aggressiveness
        att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')
        idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')

        att_axis.set_title('Iteration ' + str(i))
        att_axis.set_ylabel('$z_{att_1}$')
        att_axis.set_xlabel('$z_{att_2}$')
        idm_axis.set_ylabel('$z_{idm_1}$')
        idm_axis.set_xlabel('$z_{idm_1}$')

    animation = FuncAnimation(fig, func=animation_frame, \
                              frames=range(1, 81), interval=1)

    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=4, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('latent_evolution.mp4', writer, dpi=250)


# get_animation()
# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def get_ego_att(ego_id, ego_decision, ego_att):
    atten_on_ego = np.where(ego_att == ego_id)
    ego_changing_lane = np.where(ego_decision != 0)
    atten_on_ego_changing_lane = np.intersect1d(atten_on_ego, ego_changing_lane)
    ego_att = np.ones(40)
    ego_att[atten_on_ego_changing_lane] = 0
    return ego_att

hf_usc_indexs = {}
col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']

index = 0
for item_name in col_names:
    hf_usc_indexs[item_name] = index
    index += 1
# %%
Example_pred = 0
i = 0
covered_episodes = []
model_trainer.model.arbiter.attention_temp = 5
traces_n = 20
while Example_pred < 10:
    # sample_index = [timid_drivers[i]]
    sample_index = [val_examples[i]]
    # sample_index = [aggressive_drivers[i]]
    i += 1
    # ego_id = history_future_usc[sample_index, 0, hf_usc_indexs['ego_id']]
    ego_decision = history_future_usc[sample_index, :, hf_usc_indexs['ego_decision']][0]
    ego_att = history_future_usc[sample_index, :, hf_usc_indexs['ego_att']][0]
    merger_exists = history_future_usc[sample_index, :, hf_usc_indexs['merger_exists']][0]
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    # if ego_att.mean() != 0:
    # plt.figure(figsize=(4, 4))
    # plt.plot(ego_decision)
    # plt.plot(ego_att)
    # plt.plot(ego_decision)
    lane_y = history_future_usc[sample_index, :, hf_usc_indexs['lane_y']][0]
    episode = future_idm_s[sample_index, 0, 0][0]
    # future_merger_a[8255, :, 2:]
    # if ego_att[30:].mean() > 0 and ego_att[:20].mean() == 0 and aggressiveness > 0.5:
    # if ego_att[30:].mean() > 0 and ego_att[:20].mean() == 0 and aggressiveness > 0.5:
    # if episode not in covered_episodes and ego_att[30:].mean() > 0 and aggressiveness > 0.5:
    # if episode not in covered_episodes and aggressiveness > 0.5:
    if episode not in covered_episodes and ego_att[:].mean() > 0:
        # if episode not in covered_episodes and 0 < ego_att[:30].mean():
        # if episode not in covered_episodes and ego_att[30:].mean() == 0 and ego_att[:30].mean() == 1:
        covered_episodes.append(episode)
        sdv_actions = vectorise(future_merger_a[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        enc_acts = model_trainer.model.act_encoder(sdv_actions)
        prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
        sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
        att_scores =  model_trainer.model.arbiter([sampled_att_z, enc_h, enc_acts])
        idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])

        act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, future_idm_ss])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

        plt.figure(figsize=(4, 4))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        ego_id = history_future_usc[sample_index, 0, hf_usc_indexs['ego_id']][0]
        time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, ego_id, aggressiveness]]
        plt.text(0.5, 0.5,
                        'episode_id: '+ info[0] +
                        'time_0: '+ info[1] +
                        'ego_id: '+ info[2] +
                        'aggressiveness: '+ info[3]
                            , fontsize=10)
        plt.text(0.1, 0.1, str(idm_params.numpy()[:, :].mean(axis=0)))


        plt.figure(figsize=(4, 4))
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['leader_action']][0], color='purple')
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['ego_action']][0], color='black')
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['merger_action']][0], color='red')
        plt.legend(['leader_action', 'ego_action', 'merger_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(0, 40), act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.5)
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(4, 4))
        # plt.plot(ego_att[:40] , color='black')
        plt.plot(range(0, 40), ego_att, color='red')
        for sample_trace_i in range(traces_n):
           plt.plot(range(0, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
        plt.ylim(-0.1, 1.1)
        plt.title(str(sample_index[0]) + ' -- Attention')
        plt.grid()

        ##########
        # lATENT
        att_axis, idm_axis = latent_vis()
        att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, color='black')
        idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, color='black')

        att_axis.set_ylabel('$z_1$')
        att_axis.set_xlabel('$z_2$')
        ##########

        # plt.plot(desired_vs)
        # plt.grid()
        # plt.plot(desired_tgaps)
        # plt.grid()
        plt.figure(figsize=(4, 4))
        desired_vs = idm_params.numpy()[:, 0]
        desired_tgaps = idm_params.numpy()[:, 1]
        plt.scatter(desired_vs, desired_tgaps, color='grey')

        plt.scatter(24.7, 1.5, color='red')
        plt.xlim(15, 40)
        plt.ylim(0, 3)
        #
        # plt.scatter(30, 1, color='red')
        # plt.xlim(25, 35)
        # plt.ylim(0, 2)

        plt.title(str(sample_index[0]) + ' -- Param')
        plt.grid()

        ##########
        plt.figure(figsize=(4, 4))
        plt.plot(merger_exists, color='black')
        plt.title(str(sample_index[0]) + ' -- merger_exists')
        plt.grid()
        ############
        plt.figure(figsize=(4, 4))
        plt.plot(lane_y[:20], color='black')
        plt.plot(range(0, 40), lane_y, color='red')
        # plt.plot([0, 40], [-0.37, -0.37], color='green')
        # plt.plot([0, 40], [-1, -1], color='red')
        # plt.plot([0, 40], [-1.5, -1.5], color='red')
        plt.title(str(sample_index[0]) + ' -- lane_y')
        plt.grid()
        ############

        Example_pred += 1
# %%
"""Sample anayis
"""
########################
desired_v = 20
desired_tgap = 2
min_jamx = 2.
max_act = 0.8
min_act = 1.4

i = 44366
# vel = future_idm_s[i, :, 2]
# dv = future_idm_s[i, :, 3]
# dx = future_idm_s[i, :, 4]
#
vel = future_idm_s[i, :, 2]
dv = future_idm_s[i, :, 5]
dx = future_idm_s[i, :, 6]
desired_gap = min_jamx + \
 np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

act_pred = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)

desired_v = 19.4
desired_tgap = 2.
min_jamx = 4
max_act = 0.8
min_act = 1

desired_gap = min_jamx + \
 np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

act_true = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
# ego_act = future_ego_a[i, :, -1]

plt.plot(act_pred)
plt.plot(act_true, color='red')
# plt.plot(ego_act, color='red')
plt.grid()

# %%

desired_gap = min_jamx + desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act))
act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
plt.plot(act)
plt.legend(['with clip', 'without clip'])

act_1 = future_ego_a[i, :, -1]
plt.plot(act)
plt.plot(act)
plt.plot(model_act)
plt.scatter(range(20), model_act)
plt.plot(act_1)

plt.grid()
plt.legend(['model output', 'data'])
# features = features[features[:, 6]==0] # merger exists
loss = abs(act-future_ego_a[i, :, -1])
loss
future_idm_s[i, :, :]
# sdv_actions.shape
# %%
"""Single sample Anticipation visualisation
"""


model_trainer.model.arbiter.attention_temp = 20
traces_n = 5
sample_index = [40273]
ego_decision = history_future_usc[sample_index, :, hf_usc_indexs['ego_decision']][0]
ego_att = history_future_usc[sample_index, :, hf_usc_indexs['ego_att']][0]
merger_exists = history_future_usc[sample_index, :, hf_usc_indexs['merger_exists']][0]

lane_y = history_future_usc[sample_index, :, hf_usc_indexs['lane_y']][0]
episode = future_idm_s[sample_index, 0, 0][0]

episode = future_idm_s[sample_index, 0, 0][0]
sdv_actions = vectorise(future_merger_a[sample_index, :, 2:], traces_n)
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
enc_acts = model_trainer.model.act_encoder(sdv_actions)
prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
att_scores =  model_trainer.model.arbiter([sampled_att_z, enc_h, enc_acts])
idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])


act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, future_idm_ss])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

time_axis = np.linspace(0., 4., 40)
plt.figure(figsize=(4, 4))
episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
ego_id = history_future_usc[sample_index, 0, hf_usc_indexs['ego_id']][0]
time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
info = [str(item)+' '+'\n' for item in [episode_id, time_0, ego_id, aggressiveness]]
plt.text(0.5, 0.5,
                'episode_id: '+ info[0] +\
                'time_0: '+ info[1] +\
                'ego_id: '+ info[2] +\
                'aggressiveness: '+ info[3]
                    , fontsize = 15)
plt.text(0.1, 0.1, str(idm_params.numpy()[:, :].mean(axis=0)))

##########
# %%
plt.figure(figsize=(4, 4))
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['leader_action']][0], color='purple')
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['ego_action']][0], color='red')
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['merger_action']][0], color='black')
plt.legend(['Leader', 'Follower', 'Merger'])

for sample_trace_i in range(traces_n):
   plt.plot(time_axis, act_seq[sample_trace_i, :, :].flatten(), \
                                color='grey', alpha=0.5)
plt.title('Vehicle actions')
plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration ($ms^{-2}$)')
plt.ylim(-3, 1)
plt.grid()
##########
# %%
plt.figure(figsize=(4, 4))
plt.plot(time_axis, ego_att, color='red', linewidth=3)
for sample_trace_i in range(traces_n):
   plt.plot(time_axis, att_scores[sample_trace_i, :].flatten(), color='grey', alpha=0.5)
plt.ylim(-0.1, 1.1)
plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('Attentiveness (%)')
plt.title('Driver attentiveness')
plt.legend(['True', 'Predicted'])
plt.grid()
# %%

##########
# lATENT
att_axis, idm_axis = latent_vis()
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, color='black')
idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, color='black')

att_axis.set_ylabel('$z_1$')
att_axis.set_xlabel('$z_2$')

# %%
##########
desired_vs = idm_params.numpy()[:, 0]
desired_tgaps = idm_params.numpy()[:, 1]
min_jamx = idm_params.numpy()[:, 2]
fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(24.7,  1.5, 2, color='red')
# ax.scatter(19.4,  2, 4, color='red')
ax.scatter(desired_vs, desired_tgaps, min_jamx, color='grey')
ax.set_xlim(18, 30)
ax.set_ylim(1, 2)
ax.set_zlim(0, 4)
ax.set_title('Driver disposition')

ax.set_xlabel('Desired speed')
ax.set_ylabel('Desired time gap')
ax.set_zlabel('Minimum jam distance')
plt.legend(['True', 'Predicted'])


# %%

##########
plt.figure(figsize=(4, 4))
plt.plot(merger_exists, color='black')
plt.title(str(sample_index[0]) + ' -- merger_exists')
plt.grid()
############
plt.figure(figsize=(4, 4))
plt.plot(lane_y[:20], color='black')
plt.plot(range(0, 40), lane_y, color='red')
# plt.plot([0, 40], [-0.37, -0.37], color='green')
# plt.plot([0, 40], [-1, -1], color='red')
# plt.plot([0, 40], [-1.5, -1.5], color='red')
plt.title(str(sample_index[0]) + ' -- lane_y')
plt.grid()
############
# %%
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
