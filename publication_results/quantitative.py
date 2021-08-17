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
# model_name = 'lstm_model'
real_collections = {}
ima_collections = {}
model_names = ['driver_model', 'lstm_model', 'mlp_model']
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

snip_collection_pred.shape
snip_collection_true.shape

# %%
"""
rwse x position
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

for model_name in model_names:
    error_total = get_rwse(3, model_name)
    plt.plot(error_total)

plt.legend(model_names)
# %%
"""
rwse speed
"""
for model_name in model_names:
    error_total = get_rwse(4, model_name)
    plt.plot(error_total)

# %%
"""
gap dist
"""
true_min_gaps = snip_collection_true['driver_model'][:, :, -1].flatten()
plt.figure()
model_name = 'driver_model'
_ = plt.hist(true_min_gaps, bins=30, color='white', edgecolor='black', linewidth=1.5)
pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=30, color='green', alpha=0.5)

plt.figure()
model_name = 'lstm_model'
_ = plt.hist(true_min_gaps, bins=30, color='white', edgecolor='black', linewidth=1.5)
pred_min_gaps = np.mean(snip_collection_pred[model_name][:, :, :, -1], axis=1).flatten()
_ = plt.hist(pred_min_gaps, bins=30, color='green', alpha=0.5)


# %%
