import matplotlib.pyplot as plt
from importlib import reload
import pickle
import sys
import tensorflow as tf
reload(plt)
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import beta

import numpy as np
np.set_printoptions(suppress=True)
import os


# %%
mean = 0.5
precision = 5
alpha_param = precision*mean
beta_param = precision*(1-mean)
gen_samples = np.random.beta(alpha_param, beta_param, 100)*3
# gen_samples =  18 + np.random.beta(alpha_param, beta_param, 50)*14
plt.xlim(0, 3)

_ = plt.hist(gen_samples, bins=150)
gen_samples.std()


# %%
"""
Data prep
Note:
If you change sequence length, you need also change visualisation ranges AND
the fetch_traj method.
"""
import data_generator
reload(data_generator)
from data_generator import DataGeneratorMerge
data_gen = DataGeneratorMerge()
with open('./models/experiments/sim_data_009.pickle', 'rb') as handle:
    features = pickle.load(handle)
features, dummy_value_set = data_gen.fill_missing_values(features)
features_scaled, scaler = data_gen.scale_data(features)

history_future_seqs = data_gen.sequence(features, 20, 20)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 20)
data_arrays = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_a, future_e_veh_a = data_arrays

future_idm_s[0, 0, :]
future_idm_s.shape

# %%
"""
BALANCE DATA
"""
history_future_usc, history_sca, future_sca, future_idm_s, future_m_veh_a, future_e_veh_a = data_arrays
cond = (history_future_usc[:, :, -3] == 1).any(axis=1)
data_arrays = [np.append(data_array, data_array[cond], axis=0) for data_array in data_arrays]
balance_value = np.count_nonzero((history_future_usc[:, :, -3] == 1).any(axis=1))/\
                        np.count_nonzero((history_future_usc[:, :, -3] != 1).any(axis=1))
print(balance_value)
# %%
indxs = {}
feature_names = [
         'episode_id', 'time_step',
         'e_veh_id', 'f_veh_id', 'm_veh_id',
         'e_veh_decision', 'f_veh_exists', 'm_veh_exists', 'e_veh_att',
         'e_veh_glob_x', 'f_veh_glob_x', 'm_veh_glob_x',
         'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
         'e_veh_action', 'f_veh_action', 'm_veh_action',
         'aggressiveness', 'desired_v',
         'desired_tgap', 'min_jamx', 'max_act', 'min_act',
         'el_delta_v', 'el_delta_x', 'em_delta_v', 'em_delta_x', 'em_delta_y']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1
indxs['e_veh_att']
indxs['desired_v']
features[:, indxs['el_delta_x']].min()
features[:, indxs['em_delta_x']].min()
# %%
epis = 47
for param_name in [ 'aggressiveness', 'desired_v',
                            'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    print(param_name, ' ', features[features[:, 0] == epis][0, indxs[param_name]])
a = {'go':3}
a.keys()
# %%
veh_id = 63
for param_name in [ 'aggressiveness', 'desired_v',
                            'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    print(param_name, ' ', features[features[:, 2] == veh_id][0, indxs[param_name]])
# %%
history_future_usc[history_future_usc[:, :, 2]==47]
history_sca[history_sca[:, :, 0]==302]
history_sca[history_sca[:, :, 0]==302]
history_sca[47731, 0, 0:3]

history_future_usc[0, 0]
history_future_usc[0, 0]
import pickle
#
# %%

data_id = '_009'
file_name = 'scaler'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(scaler, handle)
else:
    print('This data id exists')

file_name = 'dummy_value_set'+data_id+'.pickle'
file_address = './models/experiments/'+file_name
if not os.path.exists(file_address):
    with open(file_address, 'wb') as handle:
        pickle.dump(dummy_value_set, handle)
else:
    print('This data id exists')
# %%
future_m_veh_a.shape
future_m_veh_a.shape
# plt.plot(history_future_usc[0, :, 6])
plt.plot(history_future_usc[10450, :, 6])
# %%

col_names = ['episode_id', 'time_step', 'e_veh_id',
        'e_veh_speed', 'f_veh_speed', 'm_veh_speed',
        'e_veh_action', 'f_veh_action', 'm_veh_action',
        'ef_delta_v', 'ef_delta_x', 'em_delta_v', 'em_delta_x',
        'em_delta_y', 'e_veh_att', 'f_veh_exists', 'm_veh_exists',
        'e_veh_decision', 'aggressiveness']
# np.count_nonzero(history_future_usc[:, :, 6] == 0)

for i in range(history_future_usc.shape[-1]):
    plt.figure(figsize=(5, 3))
    to_plot = history_future_usc[:, :, i].flatten()
    _ = plt.hist(to_plot, bins=150)
    plt.title(col_names[i])
    # plt.grid()

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
config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 256,
    },
    "exp_id": "NA",
    "Note": ""
}

class Trainer():
    def __init__(self, model_type, model_name):
        self.model = None
        self.model_type = model_type
        self.model_name = model_name

        self.train_mseloss = []
        self.train_klloss = []

        self.test_mseloss = []
        self.test_klloss = []
        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self):
        if self.model_name == 'driver_model':
            from models.core import driver_model
            reload(driver_model)
            from models.core.driver_model import  NeurIDMModel
            self.model = NeurIDMModel(config)

        with open('./models/experiments/scaler_009.pickle', 'rb') as handle:
            self.model.forward_sim.scaler = pickle.load(handle)

    def prep_data(self, training_data):
        all_epis = np.unique(training_data[0][:, 0, 0])
        np.random.seed(2021)
        np.random.shuffle(all_epis)
        # train_epis = all_epis[:int(len(all_epis)*0.1)]
        # val_epis = train_epis

        train_epis = all_epis[:int(len(all_epis)*0.7)]
        val_epis = np.setdiff1d(all_epis, train_epis)
        train_indxs = np.where(training_data[0][:, 0:1, 0] == train_epis)[0]
        val_indxs = np.where(training_data[0][:, 0:1, 0] == val_epis)[0]


        _, history_sca, future_sca, future_idm_s,\
                    future_m_veh_a, future_e_veh_a = training_data

        if self.model_type == 'cvae':
            train_input = [history_sca[train_indxs, :, 2:],
                        future_sca[train_indxs, :, 2:],
                        future_idm_s[train_indxs, :, 2:],
                        future_m_veh_a[train_indxs, :, 2:],
                        future_e_veh_a[train_indxs, :, 2:]]

            val_input = [history_sca[val_indxs, :, 2:],
                        future_sca[val_indxs, :, 2:],
                        future_idm_s[val_indxs, :, 2:],
                        future_m_veh_a[val_indxs, :, 2:],
                        future_e_veh_a[val_indxs, :, 2:]]

        return train_input, val_input

    def train(self, train_input, val_input, epochs):
        for epoch in range(epochs):
            self.epoch_count += 1
            self.model.train_loop(train_input)
            self.model.test_loop(val_input, epoch)
            if self.model_type == 'cvae':
                self.train_mseloss.append(round(self.model.train_mseloss.result().numpy().item(), 2))
                self.train_klloss.append(round(self.model.train_klloss.result().numpy().item(), 2))
                self.test_mseloss.append(round(self.model.test_mseloss.result().numpy().item(), 2))
                self.test_klloss.append(round(self.model.test_klloss.result().numpy().item(), 2))

            print(self.epoch_count, 'epochs completed')
            # print('vae_loss_weight', self.model.vae_loss_weight)
            # self.model.vae_loss_weight += 0.03

    def save_model(self, model_name, exp_id):
        model_name += exp_id + '_epo_'+str(self.epoch_count)
        print(model_name)
        exp_dir = './models/experiments/'+model_name+'/model'
        if not os.path.exists('./models/experiments/'+model_name):
            self.model.save_weights(exp_dir)
        else:
            print('This model is already saved')

tf.random.set_seed(2021)
model_trainer = Trainer(model_type='cvae', model_name='driver_model')
train_input, val_input = model_trainer.prep_data(data_arrays)
# model_trainer.train(epochs=1)
exp_dir = './models/experiments/'+'h_z_f_idm_act067_epo_20'+'/model'
model_trainer.model.load_weights(exp_dir).expect_partial()
# model_trainer = Trainer(data_arrays, model_type='lstm_model')``
# model_trainer = Trainer(data_arrays, model_type='mlp_model')
# model_trainer.train(train_input, val_input, epochs=1)
# model_trainer.test_mseloss
# train_input = None
# latent_samples(model_trainer, val_examples[0:10])

# 1.5*(1-(25/15)**4 - 3)
# Using 'auto'/'sum_over_batch_size' reduction type.
# %%
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
train_epis = all_epis[:int(len(all_epis)*0.7)]
val_epis = np.setdiff1d(all_epis, train_epis)
# np.where(train_epis == 302)
train_indxs = np.where(history_future_usc[:, 0:1, 0] == train_epis)[0]
val_examples = np.where(history_future_usc[:, 0:1, 0] == val_epis)[0]
history_sca.shape
train_indxs.shape
val_examples.shape
history_sca = np.float32(history_sca)
future_idm_s = np.float32(future_idm_s)
future_m_veh_a = np.float32(future_m_veh_a)
# np.count_nonzero(np.isnan(history_sca))
# %%
model_trainer.model.vae_loss_weight = 0.1
model_trainer.model.forward_sim.attention_temp = 5
################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
# model_trainer.train(epochs=10)
model_trainer.train(train_input, val_input, epochs=5)
################## ##### ##################
################## ##### ##################
################## ##### ##################

################## MSE LOSS ###############
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
print(model_trainer.test_mseloss[-1])
# ax = latent_vis(2000)
# %%

model_trainer.model.config


# %%
sampled_zs = latent_samples(model_trainer, val_examples[0:1000])
idm_params = model_trainer.model.idm_layer(sampled_zs).numpy()
idm_params.shape
idm_params[:, 0].max()

idm_params
# %%
model_trainer.save_model('h_z_f_idm_act', '068')

# %%
"""
Find bad examples
"""

import tensorflow as tf
# examples_to_vis = val_examples
# val_examples.shape
def get_avg_loss_across_sim(examples_to_vis):
    val_input = [history_sca[examples_to_vis , :, 2:],
                future_sca[examples_to_vis, :, 2:],
                future_idm_s[examples_to_vis, :, 2:],
                future_m_veh_a[examples_to_vis, :, 2:]]
    act_pred, pri_params, pos_params = model_trainer.model(val_input)
    loss = (tf.square(tf.subtract(act_pred, future_e_veh_a[examples_to_vis, :, 2:])))
    # loss = (tf.abs(tf.subtract(act_pred, future_e_veh_a[examples_to_vis, :, 2:])))
    loss = tf.reduce_mean(loss, axis=1).numpy()
    return loss

loss = get_avg_loss_across_sim(val_examples[0:10000])
_ = plt.hist(loss, bins=150)
# _ = plt.hist(loss[loss<0.1], bins=150)
bad_examples = np.where(loss >0.1)



# %%
# from matplotlib import rcParams

# %%
x = np.linspace(12, 50, 1000)
# y = np.exp(x)
y = (25/x)**4
# plt.plot(x, y)

# y = 1/(1+np.exp(-1*x))
# y = 15 +  20/(1+np.exp(-1*x))
plt.plot(x, y)
# %%


y = np.log(1+np.exp(x))
plt.plot(x, y, color='red')


# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_samples(model_trainer, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    sdv_actions = future_m_veh_a[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
    sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
    return sampled_z

def latent_vis(n_z_samples):
    fig = pyplot.figure(figsize=(4, 6))
    examples_to_vis = np.random.choice(val_examples, n_z_samples, replace=False)

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
latent_vis(2000)
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
var = 0.1
vels_samples = []
traj_n = 5
for traj_i in range(traj_n):
    vels = []
    acts = []
    vel = 20
    for i in range(20):
        act = np.random.normal(0, var)
        vels.append(vel)
        acts.append(act)
        vel += act*0.1

    vels_samples.append(vels)
# plt.ylim(17, 23)
for traj in vels_samples:
    plt.plot(traj)
plt.plot([20]*20)
plt.grid()
# plt.plot(acts)

# %%
"""
Choose cars based on the latent for debugging
"""
sampled_z, sampled_idm_z = latent_samples(model_trainer, val_examples)
sampled_z, sampled_idm_z = sampled_z.numpy(), sampled_idm_z.numpy()

sampled_z
# %%
bad_episodes = []
bad_504 = []
bad_498 = []
# bad_zs = np.where((sampled_idm_z[:, 0] < -2) & (sampled_idm_z[:, 0] > -5))[0]
bad_zs = np.where((sampled_z[:, 1] > 5))[0]
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
for bad_indx in bad_504:
    plt.figure()
    plt.plot(history_future_usc[bad_indx, :, -6])
    plt.title(bad_indx)
    plt.grid()

# %%

# %%
"""Anticipation visualisation
"""
def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    # data: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 19, axis=1)
    return traj.flatten()

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
zzz = sampled_z.numpy()
zzz[:, 0].std()
zzz[:, 1].std()
zzz[:, 2].std()

# %%
plt.plot(range(20))
plt.xticks(np.arange(0, 20, 5))
plt.grid(axis='x')
# plt.xaxis.grid()

# %%
np.where((history_future_usc[:, 0, 0] == 8) & \
                             (history_future_usc[:, 0, 1] == 488))

# %%

Example_pred = 0
i = 0
covered_episodes = []
model_trainer.model.forward_sim.attention_temp = 20
traces_n = 50
np.where((history_future_usc[:, 0, 2] == 63))
sepcific_examples = [ 227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,
         238,  239,  240,  752,  753,  754,  755,  756,  766,  767,  768,
         769,  770,  771,  772,  773,  774,  775,  776,  777,  778,  926,
         927,  928,  929,  930,  931,  932,  933,  934,  935,  936,  937,
         938,  939,  940,  941,  942,  943,  944,  945, 1419, 1420, 1421,
        1422, 1423, 1424, 1425, 1430, 1431, 2073, 2074, 2075, 2076, 2077,
        2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088,
        2089, 2090, 2091, 6134, 6136, 6137, 6138, 6139, 7267, 7268, 7269,
        7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280,
        7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7289, 7290, 8336,
        8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8347,
        8348, 8349, 8350]

# for i in bad_examples[0]:
# for i in sepcific_examples:
# for i in bad_zs:
# for i in bad_examples[0][0:10]:
while Example_pred < 20:
    "ENSURE ONLY VAL SAMPLES CONSIDERED"
    sample_index = [val_examples[i]]
    # sample_index = [train_indxs[i]]
    # sample_index = [i]
    i += 1
    e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
    m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
    aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
    em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
    episode = future_idm_s[sample_index, 0, 0][0]
    # if episode not in covered_episodes and aggressiveness > 0.8:
    # if episode not in covered_episodes and 0.6 > aggressiveness > 0.4:
    # if episode not in covered_episodes:
    # if 4 == 4:
    # traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
    # if episode == 21 and sample_index[0] > 3300:

    # #
    #
    # if episode == 258 and sample_index[0] > 40262:
    # if episode not in covered_episodes and e_veh_att[:35].mean() == 0 and \
    #         e_veh_att[20:60].mean() > 0 and  0.5 > aggressiveness:
    # if episode not in covered_episodes and e_veh_att[:35].mean() == 0 and \
    #         e_veh_att[20:60].mean() > 0 and  np.abs(traj).max() > 0.7:
    # if episode not in covered_episodes and \
    #         e_veh_att.mean() > 0 and  0.4 < aggressiveness < 0.6:
    # if episode not in covered_episodes and aggressiveness == 0.5:
    # if episode not in covered_episodes and m_veh_exists[:20].mean() == 0 and \
    #         e_veh_att.mean() > 0:
    # if episode not in covered_episodes and \
    #         m_veh_exists[:20].mean() == 0 and e_veh_att[25:35].mean() > 0:
    # if episode not in covered_episodes and e_veh_att.mean() > 0:
    if episode not in covered_episodes and e_veh_att.mean() > 0 \
                            and e_veh_att[:20].mean() == 0:
    # avg_speed = future_idm_s[sample_index, :, 2].mean()
    # if episode not in covered_episodes and aggressiveness > 0.8 \
    #                         and avg_speed < 25:

        covered_episodes.append(episode)
        sdv_actions = vectorise(future_m_veh_a[sample_index, :, 2:], traces_n)
        h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
        future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
        enc_h = model_trainer.model.h_seq_encoder(h_seq)
        prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
        sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
        idm_params = model_trainer.model.idm_layer([sampled_z, h_seq[:,-1,:]])
        act_seq, att_scores = model_trainer.model.forward_sim.rollout([sampled_z, \
                                                    idm_params, future_idm_ss, sdv_actions])
        act_seq, att_scores = act_seq.numpy(), att_scores.numpy()

        plt.figure(figsize=(5, 3))
        episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
        e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
        time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
        info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
        plt.text(0.1, 0.5,
                        'episode_id: '+ info[0] +
                        'time_0: '+ info[1] +
                        'e_veh_id: '+ info[2] +
                        'aggressiveness: '+ info[3]
                            , fontsize=10)

        true_params = []
        for param_name in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
            true_params.append(round(features[features[:, 0] == episode][0, indxs[param_name]], 2))
        plt.text(0.1, 0.3, 'true: '+ str(true_params)) #True
        plt.text(0.1, 0.1, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))

        plt.figure(figsize=(5, 3))
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
        plt.plot(traj, color='purple')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
        plt.plot(traj, color='black')
        traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
        plt.plot(traj, color='red')
        plt.legend(['f_veh_action', 'e_veh_action', 'm_veh_action'])

        for sample_trace_i in range(traces_n):
           plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(),
                                        color='grey', alpha=0.5)
           # plt.plot(range(19, 39), act_seq[sample_trace_i, :, :].flatten(), color='grey')

        # plt.ylim(-3, 3)
        plt.title(str(sample_index[0]) + ' -- Action')
        plt.grid()

        plt.figure(figsize=(5, 3))
        # plt.plot(e_veh_att[:40] , color='black')
        plt.plot(range(39), e_veh_att, color='red')
        plt.plot([19, 19], [0, 1], color='black')
        plt.plot([39, 39], [0, 1], color='black')

        for sample_trace_i in range(traces_n):
           plt.plot(range(19, 39), att_scores[sample_trace_i, :].flatten(), color='grey')
        plt.title(str(sample_index[0]) + ' -- Attention')

        try:
            precision = 15
            alpha_param = precision*aggressiveness
            beta_param = precision*(1-aggressiveness)
            start_point = np.where(m_veh_exists[1:]-m_veh_exists[0:-1] == 1)[0][0]
            plt.plot([start_point, start_point], [0, 1], color='black', linestyle='--')
            end_point = start_point + 45
            max_prob_point = start_point + aggressiveness*45
            plt.plot([max_prob_point, max_prob_point], [0, 1], color='red', linestyle='--')
            # plt.plot([start_point+22.5, start_point+22.5], [0, 1], color='red', linestyle='--')
            x = np.linspace(start_point, end_point, 100)
            p = beta.pdf(np.linspace(0.01, 0.99, 100), alpha_param, beta_param)
            p = p/p.max()
            plt.plot(np.linspace(start_point, end_point, 100), p, color='purple')
            plt.xlim(0, 60)
        except:
            pass

        # if 0 <= aggressiveness <= 1/3:
        #     mean_dis = 0.15
        # elif 1/3 <= aggressiveness <= 2/3:
        #     mean_dis = 0.45
        # elif 2/3 <= aggressiveness:
        #     mean_dis = 0.75
        # att_max_likelihood = mean_dis*35 + \
        #                         np.where(m_veh_exists[1:]-m_veh_exists[0:-1] == 1)[0][0]


        ##########
        """
        # lATENT
        ax = latent_vis()
        ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2], s=15, color='black')
        ##########

        # plt.plot(desired_vs)
        # plt.grid()
        # plt.plot(desired_tgaps)
        # plt.grid()
        plt.figure(figsize=(5, 3))
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
        """

        ##########
        plt.figure(figsize=(5, 3))
        plt.plot(m_veh_exists, color='black')
        plt.title(str(sample_index[0]) + ' -- m_veh_exists')
        plt.grid()
        ######\######
        plt.figure(figsize=(5, 3))
        plt.plot(range(39), em_delta_y, color='red')
        # plt.plot([0, 40], [-0.37, -0.37], color='green')
        # plt.plot([0, 40], [-1, -1], color='red')
        # plt.plot([0, 40], [-1.5, -1 .5], color='red')
        plt.title(str(sample_index[0]) + ' -- em_delta_y')
        plt.grid()
        ############
        Example_pred += 1
# %%

"""Single sample Anticipation visualisation
"""

# model_trainer.model.arbiter.attention_temp = 5
traces_n = 100
model_trainer.model.forward_sim.attention_temp = 5
sample_index = [39884]
e_veh_att = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_att'])
m_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_exists'])
f_veh_exists = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_exists'])
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
em_delta_y = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['em_delta_y'])
episode = future_idm_s[sample_index, 0, 0][0]

sdv_actions = vectorise(future_m_veh_a[sample_index, :, 2:], traces_n)
# sdv_actions[:, 10, -1] = 0
h_seq = vectorise(history_sca[sample_index, :, 2:], traces_n)
future_idm_ss = vectorise(future_idm_s[sample_index, :, 2:], traces_n)
future_idm_ss[0, 0]
enc_h = model_trainer.model.h_seq_encoder(h_seq)
prior_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
sampled_z = model_trainer.model.belief_net.sample_z(prior_param)
#
# min_act = idm_params.numpy()[:, -2]
# plt.scatter(min_act, [0]*100)
# plt.scatter(1.43, 0, color='red')
idm_params = model_trainer.model.idm_layer(sampled_z)
# idm_params = tf.ones([100, 5])*[27.53, 1.14, 4, 1.93, 2.8]
# idm_params = tf.ones([100, 5])*[18., 1.11, 4, 1., 1]
# idm_params = idm_params.numpy()
# idm_params[:, 4] = 1.4
act_seq, att_scores = model_trainer.model.forward_sim.rollout([sampled_z, \
                                            idm_params, future_idm_ss, sdv_actions])
act_seq, att_scores = act_seq.numpy(), att_scores.numpy()
att_ = att_scores.flatten()

plt.figure(figsize=(5, 3))
episode_id = history_future_usc[sample_index, 0, hf_usc_indexs['episode_id']][0]
e_veh_id = history_future_usc[sample_index, 0, hf_usc_indexs['e_veh_id']][0]
time_0 = history_future_usc[sample_index, 0, hf_usc_indexs['time_step']][0]
aggressiveness = history_future_usc[sample_index, 0, hf_usc_indexs['aggressiveness']][0]
info = [str(item)+' '+'\n' for item in [episode_id, time_0, e_veh_id, aggressiveness]]
plt.text(0.1, 0.5,
        'episode_id: '+ info[0] +
        'time_0: '+ info[1] +
        'e_veh_id: '+ info[2] +
        'aggressiveness: '+ info[3]
            , fontsize=10)
true_params = []
for param_name in ['desired_v', 'desired_tgap', 'min_jamx', 'max_act', 'min_act']:
    true_params.append(round(features[features[:, 0] == episode][0, indxs[param_name]], 2))
plt.text(0.1, 0.3, 'true: '+ str(true_params)) #True
plt.text(0.1, 0.1, 'pred: '+ str(idm_params.numpy()[:, :].mean(axis=0).round(2)))
##########
# %%
# plt.figure(figsize=(10, 10))
time_axis = np.linspace(0., 4., 39)
plt.figure(figsize=(5, 3))
# plt.legend(['Leader', 'Follower', 'Merger'])

for sample_trace_i in range(traces_n):
   plt.plot(time_axis[19:], act_seq[sample_trace_i, :, :].flatten(), \
                    color='grey', alpha=0.5, linewidth=0.5, label='_nolegend_', linestyle='-')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['e_veh_action'])
plt.plot(time_axis, traj, color='red')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['m_veh_action'])
plt.plot(time_axis, traj, linestyle='--', color='black')
traj = fetch_traj(history_future_usc, sample_index, hf_usc_indexs['f_veh_action'])
plt.plot(time_axis, traj, color='purple',linestyle='-')
# plt.plot([3,3],[-2,1])
# plt.title('Vehicle actions')
# plt.fill_between([0,2],[-3,-3], [3,3], color='lightgrey')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration ($ms^{-2}$)')
# plt.ylim(-0.1, 0.1)
# plt.yticks([1, 0., -1, -2])
# plt.xticks([0., 2, 4, 6])

plt.grid()
# plt.grid(alpha=0.1)
plt.legend(['Ego', 'Merger', 'Leader'])

##########
# plt.savefig("example_actions.png", dpi=500)

# %%
plt.figure(figsize=(10, 10))
# plt.figure(figsize=(3, 2))
for sample_trace_i in range(traces_n):
   plt.plot(time_axis[19:], att_scores[sample_trace_i, :].flatten(), \
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
plt.figure(figsize=(5, 3))
plt.plot(m_veh_exists, color='black')
plt.title(str(sample_index[0]) + ' -- m_veh_exists')
plt.grid()
############
plt.figure(figsize=(5, 3))
plt.plot(em_delta_y[:20], color='black')
plt.plot(range(29, 39), em_delta_y, color='red')
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
