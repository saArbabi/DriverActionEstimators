import pickle
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

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

    with open(exp_dir+'/runtime.pickle', 'rb') as handle:
        runtimes[model_name] = pickle.load(handle)

    try:
        with open(exp_dir+'/collision_log.pickle', 'rb') as handle:
            collision_logs[model_name] = pickle.load(handle)
    except:
        collision_logs[model_name] = []

model_paper_name = {} # model names used for paper
klplot_legend_names = {}
names = ['MLP', 'LSTM', 'Latent-MLP', 'CVAE', 'NIDM']
for i, model_name in enumerate(model_names):
    model_paper_name[model_name] = names[i]

names = ['Headway', 'Speed', 'Long. Acceleration']
for i, _name in enumerate(['min_delta_x', 'speed', 'act_long']):
    klplot_legend_names[_name] = names[i]

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
""""KL DIV # min_delta_x"""
BINS = 100
state_index = indxs['min_delta_x']
state_index = indxs['speed']
kl_divergences = {}
for model_name in model_names:
    kl_collection = {}
    for index in ['min_delta_x', 'speed', 'act_long']:
        state_index = indxs[index]
        true_min_gaps = snips_true[model_name][:, :, :, state_index]
        true_min_gaps = np.repeat(true_min_gaps, 2, axis=1).flatten()
        pred_min_gaps = snips_pred[model_name][:, :, :, state_index].flatten()
        trajs = [true_min_gaps, pred_min_gaps]
        kl_collection[index] = get_state_kl(trajs, BINS)
    kl_divergences[model_name] = kl_collection
kl_divergences
# %%
""" plot setup
"""
plt.style.use('ieee')
plt.rcParams["font.family"] = "Times New Roman"
MEDIUM_SIZE = 11
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

fig, ax = plt.subplots(figsize=(7, 3))
ax.grid(axis='y', alpha=0.5)
ax.axes.xaxis.set_ticks([], minor=True)
ax.axes.yaxis.set_ticks([], minor=True)
ax.set_xticklabels(model_paper_name.values())
plt.tick_params(top=False)
ax.legend(klplot_legend_names.values(), loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=False, shadow=False, edgecolor=None, ncol=5)

width = 0.5  # the width of the bars
loc = 1
bar_colors = ['blue','green','red']
state_names = ['Headway', 'Speed', 'Long. Accel']
xlabel_loc = []
for model_name in model_names:
    for i, index in enumerate(['min_delta_x', 'speed', 'act_long']):
        if model_paper_name[model_name] == 'MLP':
            label = state_names[i]
        else:
            label = None

        if i == 1:
            xlabel_loc.append(loc)

        ax.bar(loc, kl_divergences[model_name][index], width, linewidth=0.3, label=label, color=bar_colors[i])
        loc += width+0.05

    loc += 0.5

ax.set_xticks(xlabel_loc)
ax.legend(edgecolor='black')
fig.tight_layout()

# plt.savefig("kl_bar_chart.png", dpi=500)


# %%
""""Collision counts"""
collision_counts = {}
snips_pred[model_name].shape
total_traces_n = 2100
for model_name in model_names:
    count = len(collision_logs[model_name])
    collision_counts[model_name] = [count, 100*count/total_traces_n]
collision_counts
# %%

avg_runtime = {}
for model_name in model_names:
    avg = np.array(runtimes[model_name])[:, -1].mean()
    avg_runtime[model_name] = avg
avg_runtime
