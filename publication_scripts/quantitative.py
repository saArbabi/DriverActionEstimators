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
# model_name = 'driver_model'
# model_name = 'lstm_model'
real_collections = {}
ima_collections = {}
model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act']
# model_names = ['h_lat_f_idm_act']
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
    for veh_id in real_collections[model_name].keys():
        _true = np.array(real_collections[model_name][veh_id])[:, :]
        if _true.shape[0] >= horizon_steps_n :
            _true = _true[:horizon_steps_n, :]
            flatten_ima = []
            for trace in range(len(ima_collections[model_name][veh_id])):
                flatten_ima.append(ima_collections[model_name][veh_id][trace][:horizon_steps_n])

            _pred = np.array(flatten_ima)[:, :, :]
            # xposition_error = rwse(pred_traces, true_trace)
            snip_collection_true[model_name].append(_true)
            snip_collection_pred[model_name].append(_pred)
    snip_collection_pred[model_name] = np.array(snip_collection_pred[model_name])
    snip_collection_true[model_name] = np.array(snip_collection_true[model_name])

snip_collection_pred['h_lat_f_idm_act'].shape

# %%
len(flatten_ima[0][0])
snip_collection_pred['h_lat_f_idm_act'].shape
real_collections['h_lat_f_idm_act'].keys()
real_collections['h_lat_act'].keys()
np.array(real_collections['h_lat_f_idm_act1000'][10]).shape

.shape
snip_collection_true['h_lat_f_idm_act'].shape
snip_collection_pred['h_lat_act_o'].shape
snip_collection_pred['h_lat_f_id'].shape

# %%
"""
rwse methods
"""
# plt.plot(xposition_error)
def per_veh_rwse(pred_traces, true_trace):
    return np.mean((pred_traces - true_trace)**2, axis=0)**0.5

def get_rwse(index, model_name):
    posx_true = snip_collection_true[model_name][:,:,index]
    posx_pred = snip_collection_pred[model_name][:,:,:,index]

    rwse_collection = []
    veh_n = snip_collection_true[model_name].shape[0]
    for i in range(veh_n):
        rwse_collection.append(per_veh_rwse(posx_pred[i, :, :], posx_true[i, :]))
    rwse_collection = np.array(rwse_collection)

    error_total = np.mean(rwse_collection, axis=0)
    return error_total

#
# %%
"""visualise traj for debugging
"""
# plt.plot(snip_collection_pred[model_name][1,0,:,4])
car_index = 1
state_index = 4
legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
# legends = ['NIDM', 'Latent-Seq', 'Latent-Single']
for model_name, label in zip(model_names, legends):
    plt.plot(snip_collection_pred[model_name][car_index,0,:,state_index], label=label)
plt.plot(snip_collection_true[model_names[0]][car_index,:,state_index], color='red')
# plt.plot(snip_collection_true[model_names[0]][car_index,:,state_index], color='red')

plt.grid()
plt.legend()
# %%
plt.plot(snip_collection_true[model_names[0]][car_index,:,state_index])
plt.plot(snip_collection_true[model_names[-1]][car_index,:,state_index])
snip_collection_pred[model_names[-1]][car_index, 0,0,1]
snip_collection_pred[model_names[0]][car_index, 0,0,1]

# %%


model_name = 'h_lat_f_idm_act'
# model_name = 'h_lat_f__act'
# model_name = 'h_lat_act'
car_index = 4
state_index = 5
snip_collection_pred[model_name][car_index, 0,0,1]
snip_collection_pred[model_name][car_index, 0,0,0]

plt.plot(snip_collection_pred[model_name][car_index,0,:,state_index])
plt.plot(snip_collection_true[model_name][car_index,:,state_index], color='red')
minval = snip_collection_pred[model_name][car_index,0,:,state_index].min()
maxval = snip_collection_pred[model_name][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
# %%
# model_name = 'h_lat_f_idm_act'
model_name = 'h_lat_f_act'
# model_name = 'h_lat_act'
car_index = 4
state_index = 5
snip_collection_pred[model_name][car_index, 0,0,1]
snip_collection_pred[model_name][car_index, 0,0,0]

plt.plot(snip_collection_pred[model_name][car_index,0,:,state_index])
plt.plot(snip_collection_true[model_name][car_index,:,state_index], color='red')
minval = snip_collection_pred[model_name][car_index,0,:,state_index].min()
maxval = snip_collection_pred[model_name][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
# %%
state_index = -1
plt.plot(snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index])
plt.plot(snip_collection_true[model_name][car_index,:,state_index], color='red')
minval = snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index].min()
maxval = snip_collection_pred['h_lat_f_idm_act'][car_index,0,:,state_index].max()
for i in range(0, 100, 30):
    plt.plot([i, i], [minval, maxval], alpha=0.7, color='grey')
# %%

snip_collection_true[model_name][0,0,1]
0.006*19
# %%

# params = {
#           'font.size' : 20,
#           'font.family' : 'EB Garamond',
#           }
# plt.rcParams.update(params)
# plt.style.use(['science', 'ieee'])
# d%%
"""
rwse x position
"""
time_vals = np.linspace(0, 6, 60)

"""
rwse x position
"""
fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.05)
# for model_name in model_names:
for model_name, label in zip(model_names, legends):
    error_total = get_rwse(3, model_name)
    position_axis.plot(time_vals, error_total, label=label)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
position_axis.set_ylabel('RWSE position (m)')
# position_axis.set_xlabel('Time horizon (s)')
# position_axis.selegend(legends)
position_axis.minorticks_off()
position_axis.set_ylim(0, 5)
position_axis.set_xticklabels([])
# x%%
"""
rwse speed
"""
# legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']

for model_name, label in zip(model_names, legends):
    error_total = get_rwse(4, model_name)
    speed_axis.plot(time_vals, error_total, label=label)

speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
speed_axis.set_ylim(0, 2)
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
