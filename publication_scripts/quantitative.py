import pickle
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

reload(plt)
import matplotlib.pyplot as plt

def rwse(pred_traces, true_trace):
    return np.mean((pred_traces - true_trace)**2, axis=0)*0.5


# %%
"""
Load recordings
"""
# model_name = 'driver_model'
# model_name = 'lstm_model'
real_collections = {}
ima_collections = {}
model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']
for model_name in model_names:
    with open('./publication_results/'+model_name+'/real_collection.pickle', 'rb') as handle:
        real_collections[model_name] = pickle.load(handle)

    with open('./publication_results/'+model_name+'/ima_collection.pickle', 'rb') as handle:
        ima_collections[model_name] = pickle.load(handle)

# %%

"""
snip data for the 20s horizon.
"""
# %%
snip_collection_true = {}
snip_collection_pred = {}
for model_name in model_names:
    snip_collection_true[model_name] = []
    snip_collection_pred[model_name] = []

horizon_steps_n = 50
for model_name in model_names:
    for veh_id in real_collections[model_name].keys():
        _true = np.array(real_collections[model_name][veh_id])[:, :]
        if _true.shape[0] >= horizon_steps_n:
            _true = _true[:horizon_steps_n, :]
            _pred = np.array(ima_collections[model_name][veh_id])[:, :horizon_steps_n, :]
            # xposition_error = rwse(pred_traces, true_trace)
            snip_collection_true[model_name].append(_true)
            snip_collection_pred[model_name].append(_pred)
    len(snip_collection_pred)
    snip_collection_pred[model_name] = np.array(snip_collection_pred[model_name])
    snip_collection_true[model_name] = np.array(snip_collection_true[model_name])

# %%

snip_collection_pred['driver_model'].shape
snip_collection_true.shape

# %%
"""
rwse
"""
# plt.plot(xposition_error)
def per_veh_rwse(pred_traces, true_trace):
    return np.mean((pred_traces - true_trace)**2, axis=0)*0.5

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
fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.05)
for model_name in model_names:
    error_total = get_rwse(3, model_name)
    position_axis.plot(np.linspace(0., 5., 50), error_total)

legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']
position_axis.set_ylabel('RWSE position (m)')
# position_axis.set_xlabel('Time horizon (s)')
# position_axis.selegend(legends)
position_axis.minorticks_off()
position_axis.set_ylim(0, 4)
position_axis.set_xticklabels([])
# s%%
"""
rwse speed
"""
# legends = ['NIDM', 'LSTM-MDN', 'MLP-MDN']

for model_name, label in zip(model_names, legends):
    error_total = get_rwse(4, model_name)
    speed_axis.plot(np.linspace(0., 5., 50), error_total, label=label)

speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
speed_axis.set_ylim(0, 0.43)
speed_axis.set_yticks([0, 0.2, 0.4])
# speed_axis.legend(legends)
speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=3)
# plt.savefig("rwse.png", dpi=500)



# %%
"""
gap dist
"""
plt.figure()
model_name = 'driver_model_l2_double'
true_min_gaps = snip_collection_true[model_name][:, :, -1].flatten()
_ = plt.hist(true_min_gaps, bins=30, color='white', edgecolor='black', linewidth=1.5)
pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=30, color='green', alpha=0.5)

plt.figure()
model_name = 'lstm_model'
true_min_gaps = snip_collection_true[model_name][:, :, -1].flatten()
_ = plt.hist(true_min_gaps, bins=30, color='white', edgecolor='black', linewidth=1.5)
pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=30, color='green', alpha=0.5)


# %%
import numpy as np

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


values1 = [1.346112,1.337432,1.246655]
values2 = [1.033836,1.082015,1.117323]

KL(values1, values2)
