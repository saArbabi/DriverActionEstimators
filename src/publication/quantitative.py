import pickle
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# -%%
"""
Load recordings
"""
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
runtimes = {}
# model_names = ['neural_idm_238', 'neural_037', 'latent_mlp_12', 'mlp_03', 'lstm_03']
model_names = ['neural_idm_326', 'neural_idm_327', 'neural_044', 'latent_mlp_22','mlp_05', 'lstm_05']
model_names = ['neural_idm_326', 'neural_idm_353', 'neural_idm_353__', 'neural_idm_355']
# model_names = ['neural_idm_326']
model_names = ['neural_045', 'neural_idm_355', 'latent_mlp_22','mlp_05', 'lstm_05']
# model_names = ['neural_idm_320', 'neural_040', 'lstm_04']
# model_names = ['neural_045', 'neural_idm_355']
mc_run_name = 'rwse'

for model_name in model_names:
    exp_dir = './src/evaluation/mc_collections/'+ mc_run_name + '/' + model_name

    with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open(exp_dir+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)

    with open(exp_dir+'/runtime.pickle', 'rb') as handle:
        runtimes[model_name] = pickle.load(handle)

    try:
        with open(exp_dir+'/collision_log.pickle', 'rb') as handle:
            collision_logs[model_name] = pickle.load(handle)
    except:
        collision_logs[model_name] = []

# 4%%
"""
Each trajectory snippet is steps_n time steps long.
"""
steps_n = 50
snips_true = {}
snips_pred = {}
for model_name in model_names:
    snips_true[model_name] = [] # shape: (car_count, traces_n, steps_n, 8)
    snips_pred[model_name] = [] # shape: (car_count, 1, steps_n, 9)
for model_name in model_names:
    for epis_id, epis_dic in real_collections[model_name].items():
        for veh_id, veh_dic in real_collections[model_name][epis_id].items():
            _true = np.array(real_collections[model_name][epis_id][veh_id])
            _true = _true[:,:steps_n, :]
            # if _true[:, :, -1].mean() == 0 or _true[:, :, -1].mean() == 1:
            #     continue
            flatten_ima = []
            for trace in range(len(ima_collections[model_name][epis_id][veh_id])):
                flatten_ima.append(\
                    ima_collections[model_name][epis_id][veh_id][trace][:steps_n])

            _pred = np.array(flatten_ima)
            _pred = _pred[:,:steps_n, :]
            # xposition_error = rwse(pred_traces, true_trace)
            snips_true[model_name].append(_true)
            snips_pred[model_name].append(_pred)
    # snips_pred[model_name] = np.array(snips_pred[model_name])[14:, :, :, :]
    # snips_true[model_name] = np.array(snips_true[model_name])[14:, :, :, :]
    snips_pred[model_name] = np.array(snips_pred[model_name])
    snips_true[model_name] = np.array(snips_true[model_name])
list(snips_pred.values())[0].shape
# %%

"""
Vis true vs pred state for models.
Note:
Models being compared qualitatively must have the same history_len.
"""
state_index = indxs['speed']
state_index = indxs['act_long']
for i in range(190, 207):
    epis_id = snips_true[model_names[-1]][i,0,0,1]
    veh_id = snips_true[model_names[-1]][i,0,0,2]
    state_true = snips_true[model_names[-1]][i,0,:,state_index]
    for model_name in model_names:
        plt.figure()
        plt.plot(state_true, color='red', linestyle='--', label=model_name)
        plt.title(str(i)+'   Episode_id:'+str(epis_id)+\
                                                    '   Veh_id:'+str(veh_id))


        for trace in range(10):
            state_pred = snips_pred[model_name][i,trace,:,state_index]
            # plt.plot(state_pred, color='grey')
            plt.plot(state_pred, label=trace)
        plt.legend()

# %%

"""
Vis true vs pred state for models.
Note:
Models being compared qualitatively must have the same history_len.
"""
state_index = indxs['speed']
state_index = indxs['act_long']
for i in range(14):
    plt.figure()
    epis_id = snips_true[model_names[-1]][i,0,0,1]
    veh_id = snips_true[model_names[-1]][i,0,0,2]
    state_true = snips_true[model_names[-1]][i,0,:,state_index]
    plt.plot(state_true, color='red', linestyle='--')
    plt.title(str(i)+'   Episode_id:'+str(epis_id)+\
                                                '   Veh_id:'+str(veh_id))
    for model_name in model_names:
        for trace in range(1):
            state_pred = snips_pred[model_name][i,trace,:,state_index]
            # plt.plot(state_pred, color='grey')
            plt.plot(state_pred, label=model_name)
    plt.legend()

# %%
"""
used methods
"""
# plt.plot(xposition_error)
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [traces_n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_veh_err(index, model_name, car_id_to_rwse):
    """
    Input shpae [veh_n, traces_n, steps_n, state_index]
    Return shape [veh_n, steps_n]
    """
    if type(car_id_to_rwse) == int:
        posx_true = snips_true[model_name][car_id_to_rwse:car_id_to_rwse+1,:,:,index]
        posx_pred = snips_pred[model_name][car_id_to_rwse:car_id_to_rwse+1,:,:,index]
    else:
        posx_true = snips_true[model_name][:,:,:,index]
        posx_pred = snips_pred[model_name][:,:,:,index]

    vehs_err_arr = [] # vehicles error array
    veh_n = posx_true.shape[0]
    for i in range(veh_n):
        vehs_err_arr.append(get_trace_err(posx_pred[i, :, :], posx_true[i, :, :]))
    return np.array(vehs_err_arr)

def get_rwse(vehs_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(vehs_err_arr, axis=0)**0.5

# %%

"""
rwse x position
"""
time_vals = np.linspace(0, 5, steps_n)
car_id_to_rwse = 'all'
# car_id_to_rwse = 109

fig = plt.figure(figsize=(8, 6))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.1)
# for model_name in model_names:
for model_name in model_names:
    vehs_err_arr = get_veh_err(indxs['glob_x'], model_name, car_id_to_rwse)
    error_total = get_rwse(vehs_err_arr)
    if model_name == 'neural_idm_180':
        position_axis.plot(time_vals, error_total, \
                           label=model_name, linestyle='--')
    else:
        position_axis.plot(time_vals, error_total, label=model_name)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
position_axis.set_ylabel('RWSE position (m)', labelpad=10)
# position_axis.set_xlabel('Time horizon (s)')
position_axis.minorticks_off()
# position_axis.set_ylim(0, 5)
position_axis.set_xticklabels([])
# 0%%
"""
rwse speed
"""

# legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']
for model_name in model_names:
    vehs_err_arr = get_veh_err(indxs['speed'], model_name, car_id_to_rwse)
    error_total = get_rwse(vehs_err_arr)
    if model_name == 'neural_idm_180':
        speed_axis.plot(time_vals, error_total, \
                           label=model_name, linestyle='--')
    else:
        speed_axis.plot(time_vals, error_total, label=model_name)
speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)', labelpad=10)
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
# speed_axis.set_ylim(0, 2)
# speed_axis.set_yticks([0, 1, 2, 3])
speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)
# %%


speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)

# %%
""""Collision counts"""
collision_counts = {}
for model_name in model_names:
    count = len(collision_logs[model_name])
    # collision_counts[model_name] = [count, count/10]
    collision_counts[model_name] = [count, count]
collision_counts
