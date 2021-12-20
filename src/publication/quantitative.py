import pickle
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# %%
"""
Load recordings
"""
# %%
indxs = {}
feature_names = [
                'time_step', 'episode_id', 'veh_id', 'trace', 'glob_x',
                'speed', 'act_long', 'min_delta_x', 'att_real']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1

real_collections = {}
ima_collections = {}
collision_logs = {}
# model_names = ['neural_idm_107', 'neural_029', 'latent_mlp_01']
# model_names = ['latent_mlp_02', 'neural_idm_113']
model_names = ['neural_032','neural_idm_117', 'latent_mlp_07', 'mlp_01', 'lstm_01']
for model_name in model_names:
    exp_dir = './src/models/experiments/'+model_name+'/eval'

    with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open(exp_dir+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)

    try:
        with open(exp_dir+'/collision_log.pickle', 'rb') as handle:
            collision_logs[model_name] = pickle.load(handle)
    except:
        collision_logs[model_name] = []

# %%
"""
Each trajectory snippet is steps_n time steps long.
"""
steps_n = 50
snips_true = {}
snips_pred = {}
for model_name in model_names:
    snips_true[model_name] = [] # shape: (car_count, trace_n, steps_n, 8)
    snips_pred[model_name] = [] # shape: (car_count, 1, steps_n, 9)
for model_name in model_names:
    for epis_id, epis_dic in real_collections[model_name].items():
        for veh_id, veh_dic in real_collections[model_name][epis_id].items():
            _true = np.array(real_collections[model_name][epis_id][veh_id])
            _true = _true[:,:steps_n, :]
            flatten_ima = []
            for trace in range(len(ima_collections[model_name][epis_id][veh_id])):
                flatten_ima.append(\
                    ima_collections[model_name][epis_id][veh_id][trace][:steps_n])

            _pred = np.array(flatten_ima)
            _pred = _pred[:,:steps_n, :]
            # xposition_error = rwse(pred_traces, true_trace)
            snips_true[model_name].append(_true)
            snips_pred[model_name].append(_pred)
    snips_pred[model_name] = np.array(snips_pred[model_name])
    snips_true[model_name] = np.array(snips_true[model_name])

# snips_pred['neural_idm_107'].shape

# %%
"""
Vis true vs pred state for models.
Note:
Models being compared qualitatively must have the same history_len.
"""
state_index = indxs['act_long']
# state_index = indxs['speed']
# model_name = 'neural_idm_105'
# model_name = 'neural_028'
error_squared = []

for i in range(40):
    plt.figure()
    epis_id = snips_true[model_names[0]][i,0,0,1]
    veh_id = snips_true[model_names[0]][i,0,0,2]
    state_true = snips_true[model_names[0]][i,0,:,state_index]
    plt.plot(state_true, color='red')
    plt.title(str(i)+'   Episode_id:'+str(epis_id)+\
                                                '   Veh_id:'+str(veh_id))
    for model_name in ['neural_idm_117', 'latent_mlp_07']:
        if model_name == 'neural_idm_117':
            color = 'blue'
        else:
            color = 'orange'

        for trace in range(1):
            state_pred = snips_pred[model_name][i,trace,:,state_index]
            plt.plot(state_pred, label=model_name, color=color)
    plt.legend()

# %%
"""
Vis speeds true vs pred for specific vehicle trace
"""
state_true = snips_true[model_name][2,0,:,state_index]
state_pred = snips_pred[model_name][2,0,:,state_index]
plt.plot(state_true, color='red')
plt.plot(state_pred, color='grey')
plt.scatter(range(steps_n), state_true, color='red')
plt.scatter(range(steps_n), state_pred, color='grey')
plt.grid()
# %%
"""
Vis error values
"""
i = 0
plt.figure()
for item in error_squared:
    if item.max() > 0.5:
        plt.plot(item, label=i)
    else:
        plt.plot(item)
    i += 1
plt.legend()

# %%
"""
used methods
"""
# plt.plot(xposition_error)
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [trace_n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_veh_err(index, model_name):
    """
    Input shpae [veh_n, trace_n, steps_n, state_index]
    Return shape [veh_n, steps_n]
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

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(p*np.log(p/q))

def get_state_kl(trajs, bins):
    """
    Returns kl divergence for two historgrams formed of
    piecewise uniform rectacngles with n bins of equal width.
    """
    EPSILON = 1e-7# to avoid zeros

    true_traj, pred_traj = trajs
    pred_hist, bin_edges = np.histogram(pred_traj, bins)
    true_hist, _ = np.histogram(true_traj, bin_edges)
    pred_hist = pred_hist + EPSILON
    true_hist = true_hist + EPSILON
    bin_width = 1/pred_hist.sum()
    pred_prob = pred_hist*bin_width
    bin_width = 1/true_hist.sum()
    true_prob = true_hist*bin_width
    kl_val = kl(true_prob, pred_prob)
    return kl_val
# %%
"""visualise traj for debugging
"""
# plt.plot(snips_pred[model_name][1,0,:,4])
# car_id = 1
car_index = 0
state_index = indxs['speed']
legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
# legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
for model_name, label in zip(model_names, legends):
    plt.plot(snips_pred[model_name][car_index,0,:,state_index], label=label)
plt.plot(snips_true[model_names[0]][car_index,0,:,state_index], color='red')
# plt.plot(snips_true[model_names[0]][car_index,0,:,state_index], color='red')
# plt.ylim(18, 20)

plt.grid()
plt.legend()
# %%
# model_name = 'h_lat_f_idm_act'
# model_name = 'h_lat_f_act'
# model_name = 'h_lat_act'
car_index = 3
state_index = 5

plt.plot(snips_pred[model_name][car_index,0,:,state_index])
plt.plot(snips_true[model_name][car_index,0,:,state_index], color='red')
minval = snips_pred[model_name][car_index,0,:,state_index].min()
maxval = snips_pred[model_name][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
# %%

# %%

# %%

# params = {
#           'font.size' : 20,
#           'font.family' : 'EB Garamond',
#           }
# plt.rcParams.update(params)
# plt.style.use(['science', 'ieee'])
# %%
"""
rwse x position
"""
time_vals = np.linspace(0, 5, steps_n)

fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.05)
# for model_name in model_names:
for model_name in model_names:
    vehs_err_arr = get_veh_err(indxs['glob_x'], model_name)
    error_total = get_rwse(vehs_err_arr)
    position_axis.plot(time_vals, error_total, label=model_name)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
position_axis.set_ylabel('RWSE position (m)')
# position_axis.set_xlabel('Time horizon (s)')
position_axis.legend(model_names)
position_axis.minorticks_off()
# position_axis.set_ylim(0, 5)
position_axis.set_xticklabels([])
# 0%%
"""
rwse speed
"""
# legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']
for model_name in model_names:
    vehs_err_arr = get_veh_err(indxs['speed'], model_name)
    error_total = get_rwse(vehs_err_arr)
    speed_axis.plot(time_vals, error_total, label=model_name)

speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
# speed_axis.set_ylim(0, 2)
# speed_axis.set_yticks([0, 0.2, 0.4])
# speed_axis.legend(legends)
speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=3)
# plt.savefig("rwse.png", dpi=500)
# %%
# vehs_err_arr = get_veh_err(indxs['speed'], model_names[1])
# error_total = get_rwse(vehs_err_arr)
# vehs_err_arr.mean()
# vehs_err_arr.mean()
# %%
""""KL DIV # min_delta_x"""
BINS = 100
state_index = indxs['min_delta_x']
state_index = indxs['speed']
state_index = indxs['act_long']
# _ = plt.hist(pred_min_gaps, bins=bins_count, color='green', alpha=0.8, range=(0, 220))
# _ = plt.hist(true_min_gaps, bins=bins_count, facecolor="None", edgecolor='black', linewidth=1.5, range=(0, 220))
kl_divergences = {}
snips_true[model_name].shape
for model_name in model_names:
    true_min_gaps = snips_true[model_name][:, :, :, state_index]
    true_min_gaps = np.repeat(true_min_gaps, 2, axis=1).flatten()
    pred_min_gaps = snips_pred[model_name][:, :, :, state_index].flatten()
    trajs = [true_min_gaps, pred_min_gaps]
    kl_divergences[model_name] = get_state_kl(trajs, BINS)
kl_divergences

# %%

plt.figure(figsize=(10, 3))
plt.bar(kl_divergences.keys(), kl_divergences.values())

# %%
""""Collision counts"""
collision_counts = {}

for model_name in model_names:
    collision_counts[model_name] = len(collision_logs[model_name])
collision_counts
