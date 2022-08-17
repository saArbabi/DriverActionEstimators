import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
import tensorflow as tf
import numpy as np
import pandas as pd

class EventFiles():
    def __init__(self):
        self.losses = {'train_losses':{'displacement_loss':[], \
                               'action_loss':[], 'kl_loss':[], 'tot_loss':[]}, \
                       'test_losses':{'displacement_loss':[], \
                              'action_loss':[], 'kl_loss':[], 'tot_loss':[]}}
    def get_event_paths(self, exp_dir):
        event_items = os.listdir(exp_dir+'/logs/train')
        train_event_paths = []
        for event in event_items:
            path = os.path.join(exp_dir+'/logs/train', event)
            train_event_paths.append(path)

        event_items = os.listdir(exp_dir+'/logs/test')
        test_event_paths = []
        for event in event_items:
            path = os.path.join(exp_dir+'/logs/test', event)
            test_event_paths.append(path)
        return train_event_paths, test_event_paths

    def read_event_files(self, exp_dir):
        train_losses = {'displacement_loss':[], 'action_loss':[], \
                                                'kl_loss':[], 'tot_loss':[]}
        test_losses = {'displacement_loss':[], 'action_loss':[], \
                                                'kl_loss':[], 'tot_loss':[]}
        train_event_paths, test_event_paths = self.get_event_paths(exp_dir)

        for train_event_path in train_event_paths:
            for e in tf.compat.v1.train.summary_iterator(train_event_path):
                for value in e.summary.value:
                    if value.tag == 'displacement_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'action_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'kl_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'tot_loss':
                        train_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())

        for test_event_path in test_event_paths:
            for e in tf.compat.v1.train.summary_iterator(test_event_path):
                for value in e.summary.value:
                    if value.tag == 'displacement_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'action_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'kl_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
                    if value.tag == 'tot_loss':
                        test_losses[value.tag].append(tf.make_ndarray(value.tensor).tolist())
        return train_losses, test_losses

    def read_losses(self, exp_dir):
        train_losses, test_losses = self.read_event_files(exp_dir)
        self.losses = {'train_losses':{'displacement_loss':[], \
                               'action_loss':[], 'kl_loss':[], 'tot_loss':[]}, \
                       'test_losses':{'displacement_loss':[], \
                              'action_loss':[], 'kl_loss':[], 'tot_loss':[]}}
        self.losses['train_losses']['displacement_loss'] = train_losses['displacement_loss']
        self.losses['test_losses']['displacement_loss'] = test_losses['displacement_loss']
        self.losses['train_losses']['action_loss'] = train_losses['action_loss']
        self.losses['test_losses']['action_loss'] = test_losses['action_loss']
        self.losses['train_losses']['kl_loss'] = train_losses['kl_loss']
        self.losses['test_losses']['kl_loss'] = test_losses['kl_loss']
        self.losses['train_losses']['tot_loss'] = train_losses['tot_loss']
        self.losses['test_losses']['tot_loss'] = test_losses['tot_loss']

exp_dir = './src/models/experiments/' + 'neural_idm_' + 'test'
event_files = EventFiles()
# %%
model_names = []

for seed_value in range(1, 11):
    exp_id = '367_seed_' + str(seed_value)
    model_name = 'neural_idm_'+exp_id
    model_names.append(model_name)

model_names
# %%
""" Data to plot
"""
train_losses = {}
test_losses = {}
for model_name in model_names:
    event_files.read_losses('./src/models/experiments/' + model_name)
    train_losses[model_name] = event_files.losses['train_losses']
    test_losses[model_name] = event_files.losses['test_losses']
    print(test_losses[model_name]['displacement_loss'][-1])

def get_ll_space(ll_dic, ll_name, loss_weight):
    all_ll = np.array([ll_dic[model_name][ll_name] for model_name in model_names])*loss_weight
    ll_stdev = all_ll.std(axis=0)
    ll_mean = all_ll.mean(axis=0)
    max_bound = ll_mean+ll_stdev
    min_bound = ll_mean-ll_stdev

    ll_mean = pd.DataFrame(ll_mean).rolling(10).mean().values.flatten()[20:]
    max_bound = pd.DataFrame(max_bound).rolling(10).mean().values.flatten()[20:]
    min_bound = pd.DataFrame(min_bound).rolling(10).mean().values.flatten()[20:]
    return ll_mean, max_bound, min_bound

# %%
""" plot total loss
"""
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 18,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 18
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

fig, ax = plt.subplots(figsize=(8, 4))

ll_mean, max_bound, min_bound = get_ll_space(test_losses, 'tot_loss', 1)
plt.plot(ll_mean, color='blue')
plt.fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='blue', alpha=0.2)

ll_mean, max_bound, min_bound = get_ll_space(train_losses, 'tot_loss', 1)
plt.plot(ll_mean, color='red')
plt.fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='red', alpha=0.2)


plt.ylabel('$\mathcal{L}_{Total}$')
plt.xlabel('Iterations')
plt.ylim([-0.005, 0.15])

#
plt.legend(['Training set', 'Validation set'], ncol=1)
plt.savefig("loss_plot.pdf", bbox_inches='tight')


# %%
""" plot all losses
"""
plt.rcParams["font.family"] = "Times New Roman"
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 18,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 18
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

subplot_xcount = 2
subplot_ycount = 2
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(8, 5))
# fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.0)
# axs[0, 0].spines['bottom'].set_visible(False)
# axs[0, 0].xaxis.set_tick_params(which="both", top=False)
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])
# axs[0, 0].xaxis.set_visible(False)
axs[1, 0].set_xlabel('Iterations')
axs[1, 1].set_xlabel('Iterations')
axs[1, 1].ticklabel_format(axis='x', style='sci')

for ax1, ax2 in zip(axs[0], axs[1]):
    ax1.tick_params(top=True, right=True, direction='in')
    ax2.tick_params(top=True, right=True, direction='in')

axs[0, 0].set_ylabel('$\mathcal{L}_{\mathrm{x}}$')
axs[0, 0].set_ylim([0, 0.005])
axs[0, 0].set_yticks([0, 0.005, 0.01])

axs[0, 1].set_ylabel('$\mathcal{L}_{\mathrm{KL}}$')
axs[0, 1].set_yticks([0.25, 0.75])
axs[0, 1].set_ylim([0, 0.9])

axs[1, 0].set_ylabel('$\mathcal{L}_a$')
axs[1, 0].set_ylim([0, 0.15])
axs[1, 0].set_yticks([0., 0.1])
axs[1, 0].set_xticks([0., 4000, 8000])

axs[1, 1].set_ylabel('$\mathcal{L}_{\mathrm{Total}}$')
axs[1, 1].set_ylim([0, 0.15])
axs[1, 1].set_yticks([0., 0.1])
axs[1, 1].set_xticks([0., 4000, 8000])

# x%%

################## displacement_loss LOSS ####    ###########
ll_mean, max_bound, min_bound = get_ll_space(train_losses, 'displacement_loss', 10)
axs[0, 0].plot(ll_mean, color='red')
axs[0, 0].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='red', alpha=0.2)

ll_mean, max_bound, min_bound = get_ll_space(test_losses, 'displacement_loss', 10)
axs[0, 0].plot(ll_mean, color='blue')
axs[0, 0].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='blue', alpha=0.2)
################## action_loss LOSS ####    ###########
ll_mean, max_bound, min_bound = get_ll_space(train_losses, 'action_loss', 1)
axs[1, 0].plot(ll_mean, color='red')
axs[1, 0].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='red', alpha=0.2)

ll_mean, max_bound, min_bound = get_ll_space(test_losses, 'action_loss', 1)
axs[1, 0].plot(ll_mean, color='blue')
axs[1, 0].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='blue', alpha=0.2)
################## kl LOSS ##################
ll_mean, max_bound, min_bound = get_ll_space(train_losses, 'kl_loss', 1)
axs[0, 1].plot(ll_mean, color='red')
axs[0, 1].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='red', alpha=0.2)

ll_mean, max_bound, min_bound = get_ll_space(test_losses, 'kl_loss', 1)
axs[0, 1].plot(ll_mean, color='blue')
axs[0, 1].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='blue', alpha=0.2)
################## Total LOSS ##################
ll_mean, max_bound, min_bound = get_ll_space(train_losses, 'tot_loss', 1)
axs[1, 1].plot(ll_mean, color='red')
axs[1, 1].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='red', alpha=0.2)

ll_mean, max_bound, min_bound = get_ll_space(test_losses, 'tot_loss', 1)
axs[1, 1].plot(ll_mean, color='blue')
axs[1, 1].fill_between(range(max_bound.shape[0]), max_bound, min_bound, color='blue', alpha=0.2)


legend = fig.legend(['Training set', 'Validation set'], loc='upper center', bbox_to_anchor=(0.5, 1),
       ncol=2, fancybox=True)

frame = legend.get_frame()
frame.set_edgecolor('black')
plt.savefig("loss_plot.pdf", bbox_inches='tight')
