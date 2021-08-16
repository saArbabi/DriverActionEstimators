import pickle
import matplotlib.pyplot as plt
import numpy as np

def rwse(pred_traces, true_trace):
    return np.mean((pred_traces - true_trace)**2, axis=0)*0.5


# %%
"""
Load recordings
"""
# model_name = 'driver_model'
model_name = 'lstm_model'

with open('./publication_results/'+model_name+'/real_collection.pickle', 'rb') as handle:
    real_collection = pickle.load(handle)

with open('./publication_results/'+model_name+'/ima_collection.pickle', 'rb') as handle:
    ima_collection = pickle.load(handle)

# %%

"""
snip data for the 20s horizon.
"""

snip_collection_true = []
snip_collection_pred = []
horizon_steps_n = 100
for veh_id in real_collection.keys():
    _true = np.array(real_collection[veh_id])[:, :]
    if _true.shape[0] >= horizon_steps_n:
        _true = _true[:horizon_steps_n, :]
        _pred = np.array(ima_collection[veh_id])[:, :horizon_steps_n, :]
        # xposition_error = rwse(pred_traces, true_trace)
        snip_collection_true.append(_true)
        snip_collection_pred.append(_pred)
len(snip_collection_pred)
snip_collection_pred = np.array(snip_collection_pred)
snip_collection_true = np.array(snip_collection_true)
snip_collection_pred.shape
snip_collection_true.shape

veh_n = snip_collection_true.shape[0]
veh_n
# %%
"""
rwse x position
"""
# plt.plot(xposition_error)
posx_true = snip_collection_true[:,:,3]
posx_pred = snip_collection_pred[:,:,:,3]
def per_veh_rwse(pred_traces, true_trace):
    return np.mean((pred_traces - true_trace)**2, axis=0)*0.5

rwse_collection = []
for i in range(veh_n):
    rwse_collection.append(per_veh_rwse(posx_pred[i, :, :], posx_true[i, :]))
rwse_collection = np.array(rwse_collection)

error_total = np.mean(rwse_collection, axis=0)
plt.plot(error_total)


# %%
"""
rwse speed
"""
speed_pred = snip_collection_pred[:,:,:,4]
speed_true = snip_collection_true[:,:,4]

rwse_collection = []
for i in range(veh_n):
    rwse_collection.append(per_veh_rwse(speed_pred[i, :, :], speed_true[i, :]))
rwse_collection = np.array(rwse_collection)

error_total = np.mean(rwse_collection, axis=0)
plt.plot(error_total)

# %%
"""
gap dist
"""
pred_min_gaps = np.mean(snip_collection_pred[:, :, :, -1], axis=1).flatten()
np.mean(snip_collection_pred[:, :, :, -1], axis=1).shape
true_min_gaps = snip_collection_true[:, :, -1].flatten()
_ = plt.hist(true_min_gaps, bins=30, color='white', edgecolor='black', linewidth=1.5)
_ = plt.hist(pred_min_gaps, bins=30, color='green', alpha=0.5)


# %%
