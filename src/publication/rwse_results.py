import pickle
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)


# %%
"""
used methods
"""
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [traces_n, rollout_len]
    Return shape [1, rollout_len]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_veh_err(index, model_name):
    """
    Input shpae [veh_n, traces_n, rollout_len, state_index]
    Return shape [veh_n, rollout_len]
    """
    posx_true = snips_true[model_name][:,:,:,index]
    posx_pred = snips_pred[model_name][:,:,:,index]

    vehs_err_arr = [] # vehicles error array
    veh_n = snips_true[model_name].shape[0]
    for i in range(veh_n):
        vehs_err_arr.append(get_trace_err(posx_pred[i, :, :], posx_true[i, :, :]))
    return np.array(vehs_err_arr)

def get_rwse(vehs_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(vehs_err_arr, axis=0)**0.5

"""
Load recordings
"""
indxs = {}
logged_state_names = [
                'time_step', 'episode_id', 'veh_id', 'trace', 'glob_x',
                'speed', 'act_long', 'min_delta_x', 'att_real']

index = 0
for item_name in logged_state_names:
    indxs[item_name] = index
    index += 1

real_collections = {}
ima_collections = {}
collision_logs = {}
runtimes = {}
model_names = ['mlp_05', 'lstm_05', 'latent_mlp_22', 'neural_045', 'neural_idm_367']
mc_run_name = 'rwse'

for model_name in model_names:
    exp_dir = './src/evaluation/mc_collections/'+ mc_run_name + '/' + model_name

    with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open(exp_dir+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)

model_paper_name = {} # model names used for paper
paper_indxs_names = {}
names = ['MLP', 'LSTM', 'Latent-MLP', 'CVAE', 'NIDM']
for i, model_name in enumerate(model_names):
    model_paper_name[model_name] = names[i]
# 4%%
"""
Each trajectory snippet is rollout_len time steps long
"""
rollout_len = 100
snips_true = {}
snips_pred = {}
for model_name in model_names:
    snips_true[model_name] = [] # shape: (car_count, traces_n, rollout_len, 8)
    snips_pred[model_name] = [] # shape: (car_count, 1, rollout_len, 9)
for model_name in model_names:
    for epis_id, epis_dic in real_collections[model_name].items():
        for veh_id, veh_dic in real_collections[model_name][epis_id].items():
            _true = np.array(real_collections[model_name][epis_id][veh_id])
            _true = _true[:,:rollout_len, :]
            flatten_ima = []
            for trace in range(len(ima_collections[model_name][epis_id][veh_id])):
                flatten_ima.append(\
                    ima_collections[model_name][epis_id][veh_id][trace][:rollout_len])

            _pred = np.array(flatten_ima)
            _pred = _pred[:,:rollout_len, :]
            snips_true[model_name].append(_true)
            snips_pred[model_name].append(_pred)
    snips_pred[model_name] = np.array(snips_pred[model_name])
    snips_true[model_name] = np.array(snips_true[model_name])

# %%
""" Data to plot
"""
time_vals = np.linspace(0, 10, rollout_len)
rwse_x = {}
rwse_v = {}
for model_name in model_names:
    vehs_err_arr = get_veh_err(indxs['glob_x'], model_name)
    error_total = get_rwse(vehs_err_arr)
    rwse_x[model_name] = error_total

    vehs_err_arr = get_veh_err(indxs['speed'], model_name)
    error_total = get_rwse(vehs_err_arr)
    rwse_v[model_name] = error_total

# %%
""" RWSE plot setup
"""
MEDIUM_SIZE = 22
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 1,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# to change default colormap
plt.rcParams["image.cmap"] = "Set1"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

linewidth = 3

fig = plt.figure(figsize=(10, 7))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.)

speed_axis.set_ylabel('RWSE speed (m/s)', labelpad=10)
speed_axis.set_xlabel('Time horizon (s)')
position_axis.set_ylabel('RWSE position (m)', labelpad=5)

speed_axis.minorticks_off()
speed_axis.set_ylim(-0.1, 4.5)
speed_axis.set_xlim(-0.1, 10.1)
speed_axis.set_yticks([0, 2, 4])

position_axis.set_yticks([0, 10, 20])
position_axis.set_xlim(-0.1, 10.1)
position_axis.set_xticklabels([])

"""
rwse x position
"""
for model_name in model_names:
    if model_name == 'neural_idm_367' or model_name == 'latent_mlp_22':
        position_axis.plot(time_vals, rwse_x[model_name], \
                           label=model_paper_name[model_name], linestyle='--', linewidth=linewidth)
    else:
        position_axis.plot(time_vals, rwse_x[model_name], label=model_paper_name[model_name], linewidth=linewidth)
"""
rwse speed
"""
for model_name in model_names:
    if model_name == 'neural_idm_367' or model_name == 'latent_mlp_22':
        speed_axis.plot(time_vals, rwse_v[model_name], \
                           label=model_paper_name[model_name], linestyle='--', linewidth=linewidth)
    else:
        speed_axis.plot(time_vals, rwse_v[model_name], label=model_paper_name[model_name], linewidth=linewidth)


position_axis.legend(fontsize=18, ncol=1, edgecolor='black')
plt.savefig("rwse.pdf", dpi=500, bbox_inches='tight')

# %%
