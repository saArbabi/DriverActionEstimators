import matplotlib.pyplot as plt
from importlib import reload
import pickle
import sys

reload(plt)
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
np.set_printoptions(suppress=True)


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
    def __init__(self, training_data, model_type, model_name):
        self.model = None
        self.model_type = model_type
        self.model_name = model_name

        self.train_mseloss = []
        self.train_klloss = []

        self.test_mseloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()
        self.prep_data(training_data)

    def initiate_model(self):
        if self.model_name == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import  NeurIDMModel
            self.model = NeurIDMModel(config)

        elif self.model_name == 'h_lat_f_act':
            from models.core import h_lat_f_act
            reload(h_lat_f_act)
            from models.core.h_lat_f_act import  NeurLatentModel
            self.model = NeurLatentModel(config)

        elif self.model_name == 'lstm_model':
            from models.core import lstm
            reload(lstm)
            from models.core.lstm import  Encoder
            self.model = Encoder(config)

        elif self.model_name == 'mlp_model':
            from models.core import mlp
            reload(mlp)
            from models.core.mlp import  MLP
            self.model = MLP(config)

        with open('./models/experiments/scaler.pickle', 'rb') as handle:
            self.model.forward_sim.scaler = pickle.load(handle)

    def prep_data(self, training_data):
        all_epis = np.unique(training_data[0][:, 0, 0])
        np.random.seed(2021)
        np.random.shuffle(all_epis)
        train_epis = all_epis[:int(len(all_epis)*0.8)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_indxs = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]

        _, history_sca, future_sca, future_idm_s,\
                    future_m_veh_a, future_e_veh_a = training_data

        if self.model_type == 'cvae':
            self.train_input = [history_sca[train_indxs, :, 2:],
                        future_sca[train_indxs, :, 2:],
                        future_idm_s[train_indxs, :, 2:],
                        future_m_veh_a[train_indxs, :, 2:],
                        future_e_veh_a[train_indxs, :, 2:]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                        future_sca[val_indxs, :, 2:],
                        future_idm_s[val_indxs, :, 2:],
                        future_m_veh_a[val_indxs, :, 2:],
                        future_e_veh_a[val_indxs, :, 2:]]

        elif self.model_type == 'lstm_model':
            self.train_input = [history_sca[train_indxs, :, 2:],
                                        future_e_veh_a[train_indxs, 0, -1]]

            self.val_input = [history_sca[val_indxs, :, 2:],
                                        future_e_veh_a[val_indxs, 0, -1]]

        elif self.model_type == 'mlp_model':
            history_sca = np.squeeze(history_sca)
            future_e_veh_a = np.squeeze(future_e_veh_a)

            self.train_input = [history_sca[train_indxs, 2:],
                                        future_e_veh_a[train_indxs, -1]]

            self.val_input = [history_sca[val_indxs, 2:],
                                        future_e_veh_a[val_indxs, -1]]

    def train(self, epochs):
        # self.model.epochs_n = epochs
        avg_training_time = (45+epochs*15)/60
        print('it will take about ' , round(avg_training_time), ' mins to complete')
        for epoch in range(epochs):
            t0 = time.time()
            self.model.train_loop(self.train_input)
            self.model.test_loop(self.val_input, epoch)
            if self.model_type == 'cvae':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))
            else:
                self.train_mseloss.append(round(self.model.train_loss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_loss.result().numpy().item(), 2))
            t1 = time.time()
            print(self.epoch_count, 'epochs completed')
            print('Epoch took: ', round(t1-t0), ' seconds')
            self.epoch_count += 1

    def save_model(self, model_name):
        model_name += '_epo_'+str(self.epoch_count)
        exp_dir = './models/experiments/'+model_name+'/model'
        self.model.save_weights(exp_dir)

model_trainer = Trainer(data_arrays, model_type='cvae', model_name='h_lat_f_act')
# model_trainer.train(epochs=1)
# exp_dir = './models/experiments/'+'driver_model'+'/model'
# model_trainer.model.load_weights(exp_dir).expect_partial()
# model_trainer = Trainer(data_arrays, model_type='lstm_model')
# model_trainer = Trainer(data_arrays, model_type='mlp_model')

# %%
#
# model_trainer.train(epochs=5)
#
# fig = plt.figure(figsize=(15, 5))
plt.style.use('default')
#
# mse_axis = fig.add_subplot(131)
# kl_axis = fig.add_subplot(132)
# idm_kl_axis = fig.add_subplot(133)
# mse_axis.plot(model_trainer.test_mseloss)
# mse_axis.plot(model_trainer.train_mseloss)
# %%
all_epis = np.unique(history_sca[:, 0, 0])
np.random.seed(2021)
np.random.shuffle(all_epis)
train_epis = all_epis[:int(len(all_epis)*0.8)]
val_epis = np.setdiff1d(all_epis, train_epis)
train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
history_sca.shape
train_indxs.shape
val_examples.shape
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_a = np.float32(future_m_veh_a)
# %%
model_trainer.model.vae_loss_weight = 0.1
model_trainer.train(epochs=5)
################## MSE LOSS ##################
fig = plt.figure(figsize=(15, 5))
# plt.style.use('default')

mse_axis = fig.add_subplot(121)
kl_axis = fig.add_subplot(122)
mse_axis.plot(model_trainer.test_mseloss)
mse_axis.plot(model_trainer.train_mseloss)

mse_axis.grid()
mse_axis.set_xlabel('epochs')
mse_axis.set_ylabel('loss (MSE)')
mse_axis.set_title('MSE')
mse_axis.legend(['test', 'train'])

################## kl LOSS ##################
kl_axis.plot(model_trainer.test_klloss)
kl_axis.plot(model_trainer.train_klloss)

kl_axis.grid()
kl_axis.set_xlabel('epochs')
kl_axis.set_ylabel('loss (kl)')
kl_axis.set_title('kl')
kl_axis.legend(['test', 'train'])

ax = latent_vis()
model_trainer.save_model('h_lat_f_act')


# %%
"""
Save model and scalers etc.
"""
import pickle
from matplotlib import rcParams
#
# model_trainer.save_model('driver_model')
# model_trainer.save_model('lstm_model')
# model_trainer.save_model('mlp_model')
with open('./models/experiments/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)
with open('./models/experiments/dummy_value_set.pickle', 'wb') as handle:
    pickle.dump(dummy_value_set, handle)
# %%
# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_samples(model_trainer, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    sdv_actions = future_m_veh_a[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
    sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_z

def latent_vis():
    fig = pyplot.figure(figsize=(4, 6))
    examples_to_vis = np.random.choice(val_examples, 10000, replace=False)

    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    sampled_z = latent_samples(model_trainer, examples_to_vis)
    aggressiveness = history_future_usc[examples_to_vis, 0, -1]
    color_shade = aggressiveness
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    ax.tick_params(pad=1)
    ax.grid(False)
    # ax.view_init(30, 50)
    #===============
    #  Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    axins = inset_axes(ax,
                        width="5%",
                        height="90%",
                        loc='right',
                        borderpad=-2
                       )

    fig.colorbar(att_sc, cax=axins)
    cbar = fig.colorbar(att_sc, cax=axins)
    ax.tick_params(pad=1)
    ax.grid(False)
    ax.view_init(30, 50)
    # ax.set_xlabel('$z_{1}$', labelpad=1)
    # ax.set_ylabel('$z_{2}$', labelpad=1)
    # ax.set_zlabel('$z_{3}$', labelpad=1)
    plt.subplots_adjust(wspace=0.2, hspace=None)
latent_vis()
# plt.savefig("latent.png", dpi=500)

# %%
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])

# %%

# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def get_e_veh_att(e_veh_id, e_veh_decision, e_veh_att):
    atten_on_ego = np.where(e_veh_att == e_veh_id)
    e_veh_changing_lane = np.where(e_veh_decision != 0)
    atten_on_e_veh_changing_lane = np.intersect1d(atten_on_ego, e_veh_changing_lane)
    e_veh_att = np.ones(40)
    e_veh_att[atten_on_e_veh_changing_lane] = 0
    return e_veh_att

hf_usc_indexs = {}
col_names = ['episode_id', 'time_step', 'e_veh_id',
        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
        'e_veh_action', 'f_veh_action', 'm_veh_action',
        'ef_delta_v', 'ef_delta_x', 'em_delta_v', 'em_delta_x',
        'em_delta_y', 'e_veh_att', 'f_veh_exists', 'm_veh_exists',
        'e_veh_decision', 'aggressiveness']

index = 0
for item_name in col_names:
    hf_usc_indexs[item_name] = index
    index += 1

# %%

Example_pred = 0
i = 0
covered_episodes = []
model_trainer.model.forward_sim.attention_temp = 20
traces_n = 20
sepcific_examples = [100000]
# for i in bad_examples[0]:
# for i in sepcific_examples:
# for i in bad_zs:
# for i in bad_examples[0][0:10]:
while Example_pred < 20:
    "ENSURE ONLY VAL SAMPLES CONSIDERED"

    sample_index = [val_examples[i]]
    # sample_index = [i]
    i += 1
    e_veh_decision = history_future_usc[sample_index, :, hf_usc_indexs['e_veh_decision']][0]
    e_veh_att = history_future_usc[sample_index, :, hf_usc_indexs['e_veh_att']][0]
    m_veh_exists = history_future_usc[sample_index, :, hf_usc_indexs['m_veh_exists']][0]
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = history_future_usc[sample_index, :, hf_usc_indexs['em_delta_y']][0]
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode not in covered_episodes and aggressiveness > 0.8:
    # if episode not in covered_episodes:
    # if 4 == 4:
    # #
    #
    if episode not in covered_episodes and e_veh_att[:25].mean() == 0 and \
            e_veh_att[20:55].mean() > 0:

    # if episode not in covered_episodes and aggressiveness == 0.5:
        covered_episodes.append(episode)
        sdv_actions = vectorise(future_m_veh_a[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
        sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
        act_seq = model_trainer.model.forward_sim.rollout([sampled_z, \
                                                    future_idm_ss, sdv_actions])
        act_seq = act_seq.numpy()

        plt.figure(figsize=(3, 3))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
        plt.text(0.5, 0.5,
                        'episode_id: '+ info[0] +
                        'time_0: '+ info[1] +
                        'e_veh_id: '+ info[2] +
                        'aggressiveness: '+ info[3]
                            , fontsize=10)


        plt.figure(figsize=(3, 3))
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['f_veh_action']][0], color='purple')
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['e_veh_action']][0], color='black')
        plt.plot(history_future_usc[sample_index, :, hf_usc_indexs['m_veh_action']][0], color='red')
        plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(0, 60), act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.5)
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(3, 3))
        Example_pred += 1
# %%

"""Single sample Anticipation visualisation
"""
# model_trainer.model.arbiter.attention_temp = 5
traces_n = 50
model_trainer.model.forward_sim.attention_temp = 20
sample_index = [22751]
e_veh_decision = history_future_usc[sample_index, :, hf_usc_indexs['e_veh_decision']][0]
e_veh_att = history_future_usc[sample_index, :, hf_usc_indexs['e_veh_att']][0]
m_veh_exists = history_future_usc[sample_index, :, hf_usc_indexs['m_veh_exists']][0]

em_delta_y = history_future_usc[sample_index, :, hf_usc_indexs['em_delta_y']][0]
episode = future_idm_s[sample_index, 0, 0][0]

episode = future_idm_s[sample_index, 0, 0][0]
sdv_actions = vectorise(future_m_veh_a[sample_index, :, 2:], traces_n)
sdv_actions.shape
# sdv_actions[:, :, 0] = 0
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
enc_h = model_trainer.model.h_seq_encoder(h_seq)
prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
idm_params = model_trainer.model.idm_layer(sampled_z)
act_seq, att_scores = model_trainer.model.forward_sim.rollout([sampled_z, \
                                            idm_params, future_idm_ss, sdv_actions])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

time_axis = np.linspace(0., 6., 60)
plt.figure(figsize=(3, 3))
episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
plt.text(0.5, 0.5,
                'episode_id: '+ info[0] +\
                'time_0: '+ info[1] +\
                'e_veh_id: '+ info[2] +\
                'aggressiveness: '+ info[3]
                    , fontsize = 15)
plt.text(0.1, 0.1, str(idm_params.numpy()[:, :].mean(axis=0)))

##########
# %%
plt.figure(figsize=(3, 2))
# plt.legend(['Leader', 'Follower', 'Merger'])
for sample_trace_i in range(traces_n):
   plt.plot(time_axis[20:], act_seq[sample_trace_i, :, :].flatten(), \
                    color='grey', alpha=0.5, linewidth=0.5, label='_nolegend_', linestyle='-')
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['e_veh_action']][0], color='red')
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['m_veh_action']][0], \
                                                    linestyle='--', color='black')
plt.plot(time_axis, history_future_usc[sample_index, :, hf_usc_indexs['f_veh_action']][0], color='purple',linestyle='-')

# plt.title('Vehicle actions')
plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration ($ms^{-2}$)')
plt.ylim(-2, 1)
plt.yticks([1, 0., -1, -2])
plt.xticks([0., 2, 4, 6])

plt.grid(alpha=0.1)
plt.legend(['Ego', 'Merger', 'Leader'])

##########
# plt.savefig("example_actions.png", dpi=500)

# %%
plt.figure(figsize=(3, 2))
for sample_trace_i in range(traces_n):
   plt.plot(time_axis[20:], att_scores[sample_trace_i, :].flatten(), \
            color='grey', alpha=0.5, linewidth=0.5, label='_nolegend_', linestyle='-')
plt.plot(time_axis, e_veh_att, color='red', linewidth=1, linestyle='-')

plt.ylim(-0.05, 1.05)
plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('$w$')
# plt.title('Driver attentiveness')
plt.legend(['True attnetion'])
plt.xticks([0., 2, 4, 6])
plt.yticks([0., 0.5, 1])
# plt.xlim(0, 1)
plt.minorticks_off()
plt.grid(alpha=0.1)
# plt.savefig("example_attention.png", dpi=500)


# %%

##########
# lATENT
ax= latent_vis()
ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, color='black')
idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, color='black')

ax.set_ylabel('$z_1$')
ax.set_xlabel('$z_2$')

# %%
##########
desired_vs = idm_params.numpy()[:, 0]
desired_tgaps = idm_params.numpy()[:, 1]
b_max = idm_params.numpy()[:, -1]
fig = pyplot.figure(figsize=(3, 2))
ax = Axes3D(fig)

ax.scatter(29.2,  1., 2.6, color='red')
ax.scatter(desired_vs, desired_tgaps, b_max, color='grey')
ax.set_xlim(28, 30)
ax.set_ylim(1, 2)
ax.set_zlim(2, 3)
ax.set_xticks([28., 29, 30])
ax.set_yticks([1, 1.5, 2.])
ax.set_zticks([2, 2.5, 3])
# ax.set_title('Driver disposition')
# ax.minorticks_off()

ax.set_xlabel('$v_{des}$', labelpad=0)
ax.set_ylabel('$T_{des}$', labelpad=1)
ax.set_zlabel('$b_{max}$', labelpad=3)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.tick_params(axis='x', which='major', pad=0)
ax.tick_params(axis='y', which='major', pad=0)
ax.tick_params(axis='z', which='major', pad=0)
ax.grid(False)

plt.legend(['True parameter', 'Predicted parameters'])
plt.savefig("example_params.png", dpi=500)



# %%
a = 7
b = a
id(a)
id(b)
##########
plt.figure(figsize=(3, 3))
plt.plot(m_veh_exists, color='black')
plt.title(str(sample_index[0]) + ' -- m_veh_exists')
plt.grid()
############
plt.figure(figsize=(3, 3))
plt.plot(em_delta_y[:20], color='black')
plt.plot(range(0, 60), em_delta_y, color='red')
# plt.plot([0, 40], [-0.37, -0.37], color='green')
# plt.plot([0, 40], [-1, -1], color='red')
# plt.plot([0, 40], [-1.5, -1.5], color='red')
plt.title(str(sample_index[0]) + ' -- em_delta_y')
plt.grid()
############
# %%

import random

"""
LATENT ANIMATION
"""
def get_animation():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def latent_samples(model_trainer, sample_index):
        sdv_actions = future_m_veh_a[sample_index, :, 2:]
        h_seq = history_sca[sample_index, :, 2:]
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
        sampled_z, sampled_idm_z = model_trainer.model.belief_net.sample_z(prior_param)
        return sampled_z, sampled_idm_z

    fig = plt.figure(figsize=(7, 7))
    plt.style.use('ggplot')
    ax = fig.add_subplot(211)
    idm_axis = fig.add_subplot(212)


    def animation_frame(i):
        model_trainer.model.vae_loss_weight = 0.1
        model_trainer.train(data_arrays, epochs=1)
        sampled_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
        aggressiveness = history_future_usc[val_examples, 0, -1]
        color_shade = aggressiveness
        ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')
        idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')

        ax.set_title('Iteration ' + str(i))
        ax.set_ylabel('$z_{att_1}$')
        ax.set_xlabel('$z_{att_2}$')
        idm_axis.set_ylabel('$z_{idm_1}$')
        idm_axis.set_xlabel('$z_{idm_1}$')

    animation = FuncAnimation(fig, func=animation_frame, \
                              frames=range(1, 81), interval=1)

    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=4, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('latent_evolution.mp4', writer, dpi=250)


# get_animation()
