# import os
# import pickle
import matplotlib.pyplot as plt
from importlib import reload
import sys
import data_generator
reload(data_generator)
from data_generator import DataGenerator
import numpy as np
import tensorflow as tf

import highway
reload(highway)
from highway import Env

import vehicle_handler
reload(vehicle_handler)

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
# features_origin[features_origin[:, indxs['ego_decision']] != 1].shape
features_origin[:, indxs['ego_action']].min()
features_origin[:, indxs['ego_action']].std()
# %%

# %%
# desired_v = 30
# desired_tgap = 1
# min_jamx = 0
# max_act = 2
# min_act = 3
# for i in range(5000):
#     vel = future_idm_s[i, :, 2]
#     dv = future_idm_s[i, :, 3]
#     dx = future_idm_s[i, :, 4]
#
# desired_v = 19.4
# desired_tgap = 2
# min_jamx = 4
# max_act = 0.8
# min_act = 1
desired_v = 24.7
desired_tgap = 1.5
min_jamx = 2
max_act = 1.4
min_act = 2


desired_v = 26
desired_tgap = 1.7
min_jamx = 2.9
max_act = 1.
min_act = 2

# desired_v:  22.8070831
# desired_v_max:  25.6258011
# desired_v_min:  21.70508
# desired_tgap:  1.00854421
# min_jamx:  3.61159277
# max_act:  1.19065118
# min_act:  2.99607801
for i in range(5000):
    # vel = future_idm_s[i, :, 2]
    # dv = future_idm_s[i, :, 5]
    # dx = future_idm_s[i, :, 6]
    vel = future_idm_s[i, :, 2]
    dv = future_idm_s[i, :, 3]
    dx = future_idm_s[i, :, 4]
    desired_gap = min_jamx + \
    np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

    act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
    # features = features[features[:, 6]==0] # merger exists
    loss = abs(act-future_ego_a[i, :, -1])
    if not loss.max() < 0.1:
        print('index:  ', i)
        print(loss.max())

# %%

i = 600
vel = future_idm_s[i, :, 2]
dv = future_idm_s[i, :, 3]
dx = future_idm_s[i, :, 4]
#
# vel = future_idm_s[i, :, 2]
# dv = future_idm_s[i, :, 5]
# dx = future_idm_s[i, :, 6]
desired_gap = min_jamx + \
 np.clip(desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_act*min_act)), a_min=0,a_max=None)

act = max_act*(1-(vel/desired_v)**4-(desired_gap/dx)**2)
plt.plot(act)

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
sample_index = [600]
traces_n = 1
sdv_actions = vectorise(future_merger_a[sample_index, :, 2:], traces_n)
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
enc_acts = model_trainer.model.act_encoder(sdv_actions)
prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
att_scores =  model_trainer.model.arbiter(sampled_att_z)
idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
idm_params = tf.reshape(idm_params, [1, 1, 5])
idm_params = tf.repeat(idm_params, 20, axis=1)

act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, future_idm_ss])
model_act = act_seq.numpy().flatten()
plt.plot(model_act)
 plt.plot(att_scores.numpy()[0])
# %%

# features = features_origin[(features_origin[:, indxs['ego_att']]==1) &
#                     (features_origin[:, indxs['aggressiveness']]==0)] # aggrss
features.shape
# features = features_origin[features_origin[:, indxs['aggressiveness']]==0.5] # aggrss
features = features_origin

# features.shape
# features[features[:, indxs['merger_exists']] == 0].shape
# features[features[:, indxs['merger_exists']] == 1].shape
features = data_gen.fill_missing_values(features)

features_scaled = data_gen.scale_data(features)
history_future_seqs = data_gen.sequence(features, 20, 20)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 20)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_arrays = [data_array[:5000, :, :] for data_array in data_arrays]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_merger_a, future_ego_a = data_arrays
future_ego_a.shape

# cond = (history_sca[:, :, -1] == 1).any(axis=1)

cond = (history_future_usc[:, :, -3] == 1).any(axis=1)
# cond = (history_future_usc[:, :, -5] == 1).any(axis=1)
# cond = (history_future_usc[:, :, -1] == 0.5).any(axis=1)
# cond = (history_future_usc[:, :, -5] == 1).any(axis=1)
np.count_nonzero(cond)
cond.shape
# data_arrays = [data_array[cond] for data_array in data_arrays]
data_arrays = [np.append(data_array, data_array[cond], axis=0) for data_array in data_arrays]


# %%
history_future_usc[cond].shape
history_future_usc.shape





# %%
indxs = {}
feature_names = [
         'episode_id', 'time_step',
         'ego_id', 'leader_id', 'merger_id',
         'ego_decision', 'leader_exists', 'merger_exists', 'aggressiveness', 'lane_y', 'ego_att',
         'ego_speed', 'leader_speed', 'merger_speed',
         'ego_action', 'leader_action', 'merger_action',
         'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1
indxs['merger_exists']
# %%


# %%
"""
EPISODE EVALUATION
"""
features_origin[features_origin[:, 2] == 110]
veh_arr = features_origin[features_origin[:, 0] == 120]
veh_arr[:, indxs['time_step']]
veh_arr[:, indxs['leader_id']]
veh_arr[:, indxs['merger_id']]
veh_arr[:, indxs['fl_delta_x']]
time_snap_start = veh_arr[0, 1]
# time_snap_1 = 69
# time_snap_2 = 209
for i in range(veh_arr.shape[-1]):
    plt.figure()
    plt.plot(veh_arr[:, 1], veh_arr[:, i])
    # plt.plot([time_snap_1, time_snap_1],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    # plt.plot([time_snap_2, time_snap_2],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.plot([time_snap_start, time_snap_start],[veh_arr[:, i].min(), veh_arr[:, i].max()])
    plt.title(feature_names[i])
    plt.grid()


# plt.ylim([-3, 3])

# %%

"""
RAW FEATURE ANAL
"""
for i in range(features_origin.shape[-1]):
    plt.figure()
    _ = plt.hist(to_plot[:, i], bins=150)
    plt.title(feature_names[i])
    plt.grid()

# %%


col_names = ['episode_id', 'time_step',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'fl_delta_v', 'fl_delta_x', 'fm_delta_v', 'fm_delta_x',
        'lane_y', 'leader_exists', 'merger_exists']

for i in range(future_sca.shape[-1]):
    plt.figure()
    to_plot = history_sca[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    plt.grid()
# %%

col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']
# np.count_nonzero(history_future_usc[:, :, 6] == 0)

for i in range(history_future_usc.shape[-1]):
    plt.figure()
    to_plot = history_future_usc[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    plt.grid()
# %%
sample_index = 9791
time_steps = history_future_usc[sample_index, :, 1]
for i in range(history_future_usc.shape[-1]):
    plt.figure()
    to_plot = history_future_usc[sample_index, :, i].flatten()
    plt.plot(time_steps, to_plot)
    plt.title(col_names[i])
    plt.grid()
# history_future_usc[9770,:,:]

# %%

col_names = ['episode_id', 'time_step', 'ego_speed',
                'fl_delta_v', 'fl_delta_x',
                'fm_delta_v', 'fm_delta_x',
                'leader_exists', 'merger_exists']
# type(future_idm_s[i, 0, -3])

for i in range(future_idm_s.shape[-1]):
    plt.figure()
    _ = plt.hist(future_idm_s[:, -1, i], bins=150)
    plt.title(col_names[i])
    plt.grid()
# %%
col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']

random_samples = np.round(np.random.uniform(0, 3000, 200))
 
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
        self.train_klloss = []

        self.valid_mseloss = []
        self.valid_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self, model_type=None):
        from models.core import driver_model
        reload(driver_model)
        from models.core.driver_model import  NeurIDMModel
        self.model = NeurIDMModel(config)

    def train(self, training_data, epochs):
        train_sample_index = int(len(training_data[0])*0.8)
        self.model.epochs_n = epochs

        _, history_sca, future_sca, future_idm_s,\
                future_merger_a, future_ego_a = training_data


        train_input = [history_sca[0:train_sample_index, :, 2:],
                    future_sca[0:train_sample_index, :, 2:],
                    future_idm_s[0:train_sample_index, :, 2:],
                    future_merger_a[0:train_sample_index, :, 2:],
                    future_ego_a[0:train_sample_index, :, 2:]]

        val_input = [history_sca[train_sample_index:, :, 2:],
                    future_sca[train_sample_index:, :, 2:],
                    future_idm_s[train_sample_index:, :, 2:],
                    future_merger_a[train_sample_index:, :, 2:],
                    future_ego_a[train_sample_index:, :, 2:]]

        for epoch in range(epochs):
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'vae_idm' or self.model_type == 'driver_model':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.valid_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.valid_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))
            else:
                self.train_loss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.valid_loss.append(round(self.model.test_loss.result().numpy().item(), 2))
            print(self.epoch_count, 'epochs completed')
            self.epoch_count += 1

    def save_model(self, model_name):
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

model_trainer = Trainer(model_type='driver_model')

# %%

model_trainer.model.vae_loss_weight = 0.1
model_trainer.train(data_arrays, epochs=5)
plt.figure()
plt.plot(model_trainer.valid_mseloss)
plt.plot(model_trainer.train_mseloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (MSE)')
plt.title('MSE')

plt.figure()
plt.plot(model_trainer.valid_klloss)
plt.plot(model_trainer.train_klloss)
plt.legend(['val', 'train'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss (KL)')
plt.title('KL')


# %%

# %%
# val_1 = model_trainer.valid_mseloss
# val_2 = model_trainer.valid_mseloss
# val_3 = model_trainer.valid_mseloss
plt.plot(val_1, color='red')
plt.plot(val_2)
plt.plot(val_3)
# %%

np.random.seed(2020)
train_sample_index = int(len(history_sca)*0.8)
val_examples = range(train_sample_index, len(history_sca))
# val_examples = range(0, len(history_sca))

history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_merger_a = np.float32(future_merger_a)

timid_drivers = []
normal_drivers = []
aggressive_drivers = []
for sample_index in val_examples:
    if history_future_usc[sample_index, -1, -1] == 0:
       timid_drivers.append(sample_index)
    elif history_future_usc[sample_index, -1, -1] == 0.5:
       normal_drivers.append(sample_index)
    elif history_future_usc[sample_index, -1, -1] == 1:
       aggressive_drivers.append(sample_index)
history_sca.shape
len(timid_drivers)
len(normal_drivers)
len(aggressive_drivers)
# %%
def latent_samples(model_trainer, sample_index):
    sdv_actions = future_merger_a[sample_index, :, 2:]
    h_seq = history_sca[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    enc_acts = model_trainer.model.act_encoder(sdv_actions)
    prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
    sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_att_z, sampled_idm_z



fig = plt.figure(figsize=(7, 7))
att_axis = fig.add_subplot(211)
idm_axs = fig.add_subplot(212)

sampled_att_z, sampled_idm_z = latent_samples(model_trainer, aggressive_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='red')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='red')

sampled_att_z, sampled_idm_z = latent_samples(model_trainer, timid_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='green')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='green')

sampled_att_z, sampled_idm_z = latent_samples(model_trainer, normal_drivers)
att_axis.scatter(sampled_att_z[:, 0], sampled_att_z[:, 1], s=15, alpha=0.3, color='orange')
idm_axs.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, color='orange')

att_axis.set_ylabel('$z_1$')
att_axis.set_xlabel('$z_2$')

# %%
sampled_att_z = sampled_att_z.numpy()
sampled_att_z[:, 0].max()
sampled_att_z[17]
bad_samples = np.where(sampled_att_z[:, 1] > 6)

normal_drivers[38]
# %%
bad_epis = []
for drv in bad_samples[0]:
    print(timid_drivers[drv])
    indx = timid_drivers[drv]
    bad_epis.append(history_sca[indx, 0, 0])
bad_epis

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

# model_trainer.model.idm_sim.arbiter.attention_temp = 20
model_trainer.model.arbiter.attention_temp = 20

Example_pred = 0
traces_n = 20
i = 0
covered_episodes = []
indxs = {}
col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']

index = 0
for item_name in col_names:
    indxs[item_name] = index
    index += 1

while Example_pred < 20:
    # sample_index = [timid_drivers[i]]
    sample_index = [normal_drivers[i]]
    # sample_index = [aggressive_drivers[i]]
    i += 1
    # ego_id = history_future_usc[sample_index, 0, indxs['ego_id']]
    ego_decision = history_future_usc[sample_index, :, indxs['ego_decision']][0]
    ego_att = history_future_usc[sample_index, :, indxs['ego_att']][0]
    merger_exists = history_future_usc[sample_index, :, indxs['merger_exists']][0]
    # if ego_att.mean() != 0:
    # plt.figure()
    # plt.plot(ego_decision)
    # plt.plot(ego_att)
    # plt.plot(ego_decision)
    lane_y = history_future_usc[sample_index, :, indxs['lane_y']][0]
    episode = future_idm_s[sample_index, 0, 0][0]
    # future_merger_a[8255, :, 2:]
    # if episode not in covered_episodes and 0 < ego_att[:30].mean() < 1:
    if episode not in covered_episodes and 0 < ego_att[:30].mean():
        # if episode not in covered_episodes and ego_att[30:].mean() == 0 and ego_att[:30].mean() == 1:
        covered_episodes.append(episode)
        sdv_actions = vectorise(future_merger_a[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        enc_acts = model_trainer.model.act_encoder(sdv_actions)
        prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
        sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
        att_scores =  model_trainer.model.arbiter(sampled_att_z)

        idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
        idm_params = tf.reshape(idm_params, [traces_n, 1, 5])
        idm_params = tf.repeat(idm_params, 20, axis=1)

        act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, future_idm_ss])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

        plt.figure()
        episode_id = history_future_usc[sample_index, 0, indxs['episode_id']][0]
        ego_id = history_future_usc[sample_index, 0, indxs['ego_id']][0]
        time_0 = history_future_usc[sample_index, 0, indxs['time_step']][0]
        aggressiveness = history_future_usc[sample_index, 0, indxs['aggressiveness']][0]
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, ego_id, aggressiveness]]
        plt.text(0.5, 0.5,
                        'episode_id: '+ info[0] +\
                        'time_0: '+ info[1] +\
                        'ego_id: '+ info[2] +\
                        'aggressiveness: '+ info[3]
                            , fontsize = 15)
        plt.text(0.1, 0.1, str(idm_params.numpy()[:, 0, :].mean(axis=0)))


        plt.figure()
        plt.plot(history_future_usc[sample_index, :, indxs['leader_action']][0], color='purple')
        plt.plot(history_future_usc[sample_index, :, indxs['ego_action']][0], color='black')
        plt.plot(history_future_usc[sample_index, :, indxs['merger_action']][0], color='red')
        plt.legend(['leader_action', 'ego_action', 'merger_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), \
                                        color='grey', alpha=0.5)
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure()
        plt.plot(ego_att[:20] , color='black')
        plt.plot(range(20, 40), ego_att[20:], color='red')
        for sample_trace_i in range(traces_n):
           plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
        plt.ylim(-0.1, 1.1)
        plt.title(str(sample_index[0]) + ' -- Attention')
        plt.grid()

        ##########

        # plt.plot(desired_vs)
        # plt.grid()
        # plt.plot(desired_tgaps)
        # plt.grid()
        plt.figure()
        desired_vs = idm_params.numpy()[:, 0, 0]
        desired_tgaps = idm_params.numpy()[:, 0, 1]
        plt.scatter(desired_vs, desired_tgaps, color='grey')

        plt.scatter(19.4, 2, color='green')
        # plt.scatter(25, 1.4, color='orange')
        # plt.scatter(30, 1, color='red')
        plt.xlim(15, 40)
        plt.ylim(0, 3)
        #
        # plt.scatter(30, 1, color='red')
        # plt.xlim(25, 35)
        # plt.ylim(0, 2)

        plt.title(str(sample_index[0]) + ' -- Param')
        plt.grid()

        ##########
        plt.figure()
        plt.plot(merger_exists, color='black')
        plt.title(str(sample_index[0]) + ' -- merger_exists')
        plt.grid()
        ############
        plt.figure()
        plt.plot(lane_y[:20], color='black')
        plt.plot(range(20, 40), lane_y[20:], color='red')
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
desired_tgap = 1.4
min_jamx = 2.4
max_act = 1.4
min_act = 2.3

i = 8237
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
indxs = {}
col_names = ['episode_id', 'time_step', 'ego_id',
        'ego_speed', 'leader_speed', 'merger_speed',
        'ego_action', 'leader_action', 'merger_action',
        'lane_y', 'ego_att', 'leader_exists', 'merger_exists',
        'ego_decision', 'aggressiveness']

index = 0
for item_name in col_names:
    indxs[item_name] = index
    index += 1

model_trainer.model.arbiter.attention_temp = 20
traces_n = 20
sample_index = [7559]
# sample_index = [aggressive_drivers[i]]
# ego_id = history_future_usc[sample_index, 0, indxs['ego_id']]
ego_decision = history_future_usc[sample_index, :, indxs['ego_decision']][0]
ego_att = history_future_usc[sample_index, :, indxs['ego_att']][0]
merger_exists = history_future_usc[sample_index, :, indxs['merger_exists']][0]
# if ego_att.mean() != 0:
# plt.figure()
# plt.plot(ego_decision)
# plt.plot(ego_att)
# plt.plot(ego_decision)
lane_y = history_future_usc[sample_index, :, indxs['lane_y']][0]
episode = future_idm_s[sample_index, 0, 0][0]

sdv_actions = vectorise(future_merger_a[sample_index, :, 2:], traces_n)
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
enc_acts = model_trainer.model.act_encoder(sdv_actions)
prior_param = model_trainer.model.belief_net([enc_h, enc_acts], dis_type='prior')
sampled_att_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
att_scores =  model_trainer.model.arbiter(sampled_att_z)

idm_params = model_trainer.model.idm_layer([sampled_idm_z, enc_h])
idm_params = tf.reshape(idm_params, [traces_n, 1, 5])
idm_params = tf.repeat(idm_params, 20, axis=1)

act_seq = model_trainer.model.idm_sim.rollout([att_scores, idm_params, future_idm_ss])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

plt.figure()
episode_id = history_future_usc[sample_index, 0, indxs['episode_id']][0]
ego_id = history_future_usc[sample_index, 0, indxs['ego_id']][0]
time_0 = history_future_usc[sample_index, 0, indxs['time_step']][0]
aggressiveness = history_future_usc[sample_index, 0, indxs['aggressiveness']][0]
info = [str(item)+' '+'\n' for item in [episode_id, time_0, ego_id, aggressiveness]]
plt.text(0.5, 0.5,
                'episode_id: '+ info[0] +\
                'time_0: '+ info[1] +\
                'ego_id: '+ info[2] +\
                'aggressiveness: '+ info[3]
                    , fontsize = 15)
plt.text(0.1, 0.1, str(idm_params.numpy()[:, 0, :].mean(axis=0)))


plt.figure()
plt.plot(history_future_usc[sample_index, :, indxs['leader_action']][0], color='purple')
plt.plot(history_future_usc[sample_index, :, indxs['ego_action']][0], color='black')
plt.plot(history_future_usc[sample_index, :, indxs['merger_action']][0], color='red')
plt.legend(['leader_action', 'ego_action', 'merger_action'])

for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), act_seq[sample_trace_i, :, :].flatten(), \
                                color='grey', alpha=0.5)
   # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

# plt.ylim(-3, 3)
plt.title(str(sample_index[0]) + ' -- Action')
plt.grid()

plt.figure()
plt.plot(ego_att[:20] , color='black')
plt.plot(range(20, 40), ego_att[20:], color='red')
for sample_trace_i in range(traces_n):
   plt.plot(range(20, 40), att_scores[sample_trace_i, :].flatten(), color='grey')
plt.ylim(-0.1, 1.1)
plt.title(str(sample_index[0]) + ' -- Attention')
plt.grid()

##########

# plt.plot(desired_vs)
# plt.grid()
# plt.plot(desired_tgaps)
# plt.grid()
plt.figure()
desired_vs = idm_params.numpy()[:, 0, 0]
desired_tgaps = idm_params.numpy()[:, 0, 1]
plt.scatter(desired_vs, desired_tgaps, color='grey')

plt.scatter(19.4, 2, color='green')
# plt.scatter(25, 1.4, color='orange')
# plt.scatter(30, 1, color='red')
plt.xlim(15, 40)
plt.ylim(0, 3)
#
# plt.scatter(30, 1, color='red')
# plt.xlim(25, 35)
# plt.ylim(0, 2)

plt.title(str(sample_index[0]) + ' -- Param')
plt.grid()

##########
plt.figure()
plt.plot(merger_exists, color='black')
plt.title(str(sample_index[0]) + ' -- merger_exists')
plt.grid()
############
plt.figure()
plt.plot(lane_y[:20], color='black')
plt.plot(range(20, 40), lane_y[20:], color='red')
# plt.plot([0, 40], [-0.37, -0.37], color='green')
# plt.plot([0, 40], [-1, -1], color='red')
# plt.plot([0, 40], [-1.5, -1.5], color='red')
plt.title(str(sample_index[0]) + ' -- lane_y')
plt.grid()
############
