import pickle
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

reload(plt)
import matplotlib.pyplot as plt


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

# model_name = 'driver_model'
# model_name = 'lstm_model'
real_collections = {}
ima_collections = {}
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act']
model_names = ['test2', 'test3', 'test4', 'h_z_f_idm_act095_epo_25']
for model_name in model_names:
    with open('./publication_results/'+model_name+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open('./publication_results/'+model_name+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)
# %%

"""
snip data for the 20s horizon.
"""
_pred.shape
len(ima_collections[model_name][veh_id][1])
len(real_collections['h_lat_act_o'])

np.array(real_collections[model_name][veh_id])
len(ima_collections[model_name][veh_id][0])
len(ima_collections[model_name][veh_id][1])
ima_collections[model_name][veh_id][0][0]
ima_collections[model_name][veh_id][1][-1]
# %%
snip_collection_true = {}
snip_collection_pred = {}
for model_name in model_names:
    snip_collection_true[model_name] = []
    snip_collection_pred[model_name] = []
horizon_steps_n = 60
for model_name in model_names:
    for epis_id, epis_dic in real_collections[model_name].items():
        for veh_id, veh_dic in real_collections[model_name][epis_id].items():
            _true = np.array(real_collections[model_name][epis_id][veh_id])
            _true = _true[:,:horizon_steps_n, :]
            _true.shape
            flatten_ima = []
            for trace in range(len(ima_collections[model_name][epis_id][veh_id])):
                flatten_ima.append(\
                    ima_collections[model_name][epis_id][veh_id][trace][:horizon_steps_n])

            _pred = np.array(flatten_ima)
            _pred = _pred[:,:horizon_steps_n, :]
            # xposition_error = rwse(pred_traces, true_trace)
            snip_collection_true[model_name].append(_true)
            snip_collection_pred[model_name].append(_pred)
    snip_collection_pred[model_name] = np.array(snip_collection_pred[model_name])
    snip_collection_true[model_name] = np.array(snip_collection_true[model_name])

snip_collection_pred['test3'].shape
snip_collection_true['test4'].shape
snip_collection_pred['test4'].shape
# snip_collection_true[model_names[0]][0 , 0, :, indxs['glob_x']] += 5
snip_collection_pred['test3'][0,0,0,:]
snip_collection_pred['test4'][0,0,0,:]
snip_collection_pred['test4'][0,0,0,:]
# %%
"""
Vis speeds true vs pred
"""
state_index = indxs['speed']
model_name = 'h_z_f_idm_act095_epo_25'

collection = []
for i in range(15):
    plt.figure()
    for trace in range(5):
        epis_id = snip_collection_true[model_name][i,0,0,1]
        veh_id = snip_collection_true[model_name][i,0,0,2]
        state_true = snip_collection_true[model_name][i,0,:,state_index]
        state_pred = snip_collection_pred[model_name][i,trace,:,state_index]
        collection.append((state_true-state_pred)**2)
        plt.plot(state_true, color='red')
        plt.plot(state_pred, color='grey')
        plt.title(str(i)+'   Episode_id:'+str(epis_id)+'   Veh_id:'+str(veh_id))
# %%
"""
Vis speeds true vs pred for specific vehicle trace
"""
state_true = snip_collection_true['test2'][10,0,:,state_index]
state_pred = snip_collection_pred['test2'][10,0,:,state_index]
plt.plot(state_true, color='red')
plt.plot(state_pred, color='grey')
plt.scatter(range(60), state_true, color='red')
plt.scatter(range(60), state_pred, color='grey')
plt.grid()
# %%
"""
Vis error values
"""
i = 0
plt.figure()
for item in collection:
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
    Input shpae [trace_n, horizon_steps_n]
    Return shape [1, horizon_steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)**0.5

def get_rwse(index, model_name):
    posx_true = snip_collection_true[model_name][:,:,:,index]
    posx_pred = snip_collection_pred[model_name][:,:,:,index]

    rwse_collection = []
    veh_n = snip_collection_true[model_name].shape[0]
    for i in range(veh_n):
        rwse_collection.append(per_veh_rwse(posx_pred[i, :, :], posx_true[i, :, :]))
    rwse_collection = np.array(rwse_collection)
    # mean across snippets (axis=0)

    error_total = np.mean(rwse_collection, axis=0)
    return error_total

# %%
"""visualise traj for debugging
"""
# plt.plot(snip_collection_pred[model_name][1,0,:,4])
# car_id = 1
car_index = 0
state_index = indxs['speed']
legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
# legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
for model_name, label in zip(model_names, legends):
    plt.plot(snip_collection_pred[model_name][car_index,0,:,state_index], label=label)
plt.plot(snip_collection_true[model_names[0]][car_index,0,:,state_index], color='red')
# plt.plot(snip_collection_true[model_names[0]][car_index,0,:,state_index], color='red')
# plt.ylim(18, 20)

plt.grid()
plt.legend()
# %%
# model_name = 'h_lat_f_idm_act'
model_name = 'h_lat_f_act'
# model_name = 'h_lat_act'
car_index = 4
state_index = 5

plt.plot(snip_collection_pred[model_name][car_index,0,:,state_index])
plt.plot(snip_collection_true[model_name][car_index,0,:,state_index], color='red')
minval = snip_collection_pred[model_name][car_index,0,:,state_index].min()
maxval = snip_collection_pred[model_name][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
# %%
state_index = -1
plt.plot(snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index])
plt.plot(snip_collection_true[model_name][car_index,0,:,state_index], color='red')
minval = snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index].min()
maxval = snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
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
time_vals = np.linspace(0, 6, 60)

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
# %%
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
true_min_gaps = snip_collection_true[model_name][:, :, -1].flatten()

pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=bins_count, color='blue', range=(0, 220))
_ = plt.hist(true_min_gaps, bins=bins_count, facecolor="None", edgecolor='black', linewidth=1.5, range=(0, 220))

#
plt.figure()

model_name = 'h_lat_f_idm_act1000'
pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
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
