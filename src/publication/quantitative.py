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
model_names = ['h_z_f_idm_act_097']
for model_name in model_names:
    exp_dir = './src/models/experiments/'+model_name+'/eval'

    with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open(exp_dir+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)
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
            _true.shape
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

snips_pred['h_z_f_idm_act_097'].shape
snips_true['h_z_f_idm_act_097'].shape

# %%
"""
Vis speeds true vs pred
"""
state_index = indxs['speed']
model_name = 'h_z_f_idm_act_097'

error_squared = []
for i in range(4):
    plt.figure()
    for trace in range(1):
        epis_id = snips_true[model_name][i,0,0,1]
        veh_id = snips_true[model_name][i,0,0,2]
        state_true = snips_true[model_name][i,0,:,state_index]
        state_pred = snips_pred[model_name][i,trace,:,state_index]
        error_squared.append((state_true-state_pred)**2)
        plt.plot(state_true, color='red')
        plt.plot(state_pred, color='grey')
        plt.title(str(i)+'   Episode_id:'+str(epis_id)+\
                                                    '   Veh_id:'+str(veh_id))
# %%
"""
Vis speeds true vs pred for specific vehicle trace
"""
state_true = snips_true['test2'][10,0,:,state_index]
state_pred = snips_pred['test2'][10,0,:,state_index]
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
rwse methods
"""
# plt.plot(xposition_error)
def per_veh_rwse(pred_traces, true_trace):
    """
    Input shpae [trace_n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)**0.5

def get_rwse(index, model_name):
    posx_true = snips_true[model_name][:,:,:,index]
    posx_pred = snips_pred[model_name][:,:,:,index]

    rwse_collection = []
    veh_n = snips_true[model_name].shape[0]
    for i in range(veh_n):
        rwse_collection.append(per_veh_rwse(posx_pred[i, :, :], posx_true[i, :, :]))
    rwse_collection = np.array(rwse_collection)
    # mean across snippets (axis=0)

    error_total = np.mean(rwse_collection, axis=0)
    return error_total

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
time_vals = np.linspace(0, 6, steps_n)

fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.05)
# for model_name in model_names:
for model_name, label in zip(model_names, model_names):
    error_total = get_rwse(indxs['glob_x'], model_name)
    position_axis.plot(time_vals, error_total, label=label)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
position_axis.set_ylabel('RWSE position (m)')
# position_axis.set_xlabel('Time horizon (s)')
position_axis.legend(model_names)
position_axis.minorticks_off()
# position_axis.set_ylim(0, 5)
position_axis.set_xticklabels([])
# 9%%
"""
rwse speed
"""
# legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']

for model_name, label in zip(model_names, model_names):
    error_total = get_rwse(indxs['speed'], model_name)
    speed_axis.plot(time_vals, error_total, label=label)

speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
# speed_axis.set_ylim(0, 2)
# speed_axis.set_yticks([0, 0.2, 0.4])
# speed_axis.legend(legends)
speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=3)
# plt.savefig("rwse.png", dpi=500)



# %%
"""
gap dist
"""

plt.figure()
bins_count = 50
# time_lapse = 199
model_name = 'h_lat_f_act1000'
true_min_gaps = snips_true[model_name][:, :, -1].flatten()

pred_min_gaps = np.mean(snips_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=bins_count, color='blue', range=(0, 220))
_ = plt.hist(true_min_gaps, bins=bins_count, facecolor="None", edgecolor='black', linewidth=1.5, range=(0, 220))

#
plt.figure()

model_name = 'h_lat_f_idm_act1000'
pred_min_gaps = np.mean(snips_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=bins_count, color='green', alpha=0.8, range=(0, 220))
_ = plt.hist(true_min_gaps, bins=bins_count, facecolor="None", edgecolor='black', linewidth=1.5, range=(0, 220))



# %%
import numpy as np

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


values1 = [1.346112,1.337432,1.246655]
values2 = [1.033836,1.082015,1.117323]

KL(values1, values2)