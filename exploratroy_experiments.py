import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from model import Encoder
from importlib import reload
import tensorflow as tf
import time
# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "enc_units": 100,
     "dec_units": 100,
     "epochs_n": 50,
     "components_n": 5,
    "batch_size": 50,
    "allowed_error": 0.3,
},
"data_config": {"obs_n": 20,
                "pred_step_n": 7,
                "step_size": 3,
                "Note": "lat/long motion not considered jointly"
                # "Note": "jerk as target"

},
"exp_id": "NA",
"Note": ""
}
# %%


###
episode_len = 30
x_init = 100
lead_x = [x_init]
for i in range(episode_len):
    lead_x.append(lead_x[i]+10*0.1)
###
# %%
def get_desired_gap(vel, dv):
    gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                    (2*np.sqrt(max_acc*max_decc))
    return gap

def act(vel, obs):
    desired_gap = get_desired_gap(vel, obs['dv'])
    acc = max_acc*(1-(vel/desired_v)**4-\
                                        (desired_gap/obs['dx'])**2)
    return sorted([-3, acc, 3])[1]
# idm params
desired_v=12 # m/s
desired_tgap=2.8 # s
min_jamx=4 # m
max_acc=3 # m/s^2
max_decc=3 # m/s^2
xs = []
ys = []
sample_size = 1000
x_range = range(0, 80)
v_range = range(5, 10)

for sample_i in range(sample_size):
    rear_x_init = np.random.choice(x_range)
    rear_v_init = np.random.choice(v_range)
    rear_x = [rear_x_init]
    rear_v = [rear_v_init]
    sample_xs = []
    for i in range(episode_len):
        obs = {'dx':lead_x[i]-rear_x[i], 'dv':10-rear_v[i]}
        acc = act(rear_v[i], obs)
        vel = rear_v[i] + acc * 0.1
        x = rear_x[i] + rear_v[i] * 0.1 \
                                    + 0.5 * acc * 0.1 **2
        rear_x.append(x)
        rear_v.append(vel)
        sample_xs.append([rear_v[i], 10-rear_v[i], lead_x[i]-rear_x[i]])
    xs.append(sample_xs)
    ys.append([acc])
len(ys)
len(xs)
# plt.plot(np.array(xs)[6,:,0])
# plt.plot(rear_x)
# plt.plot(rear_v)

# %%
from exploratory.models import idm_neural
reload(idm_neural)
from exploratory.models.idm_neural import  Encoder

exp_trains = {}
exp_vals = {}
durations = {}
def train_exp(durations, exp_trains, exp_vals, config, exp_name):

    if exp_name in (exp_trains or exp_vals):
        raise  KeyError("Experiment already completed")

    train_loss = []
    valid_loss = []

    idm_model = Encoder(config, model_use='training')

    train_indx = int(sample_size*0.8)
    for epoch in range(20):
        # model.train_loop([xs, ys])
        idm_model.train_loop([xs[0:train_indx], ys[0:train_indx]])
        idm_model.test_loop([xs[train_indx:], ys[train_indx:]], epoch)
        train_loss.append(round(idm_model.train_loss.result().numpy().item(), 2))
        valid_loss.append(round(idm_model.test_loss.result().numpy().item(), 2))
        # modelEvaluate(model, validation_data, config)
        print(epoch, 'epochs completed')
        # print('train_loss', train_loss[-1])
        # print('valid_loss', valid_loss[-1])

    exp_trains[exp_name] = train_loss
    exp_vals[exp_name] = valid_loss
    durations[exp_name] = 0


    return idm_model, durations, exp_trains, exp_vals

# train_debugger()
idm_model, durations, exp_trains, exp_vals = train_exp(durations, exp_trains,
                                        exp_vals, config, 'exp002')

plt.plot(exp_vals['exp002'])
plt.plot(exp_trains['exp002'])
plt.legend(['val', 'train'])


# %%
ys[0][0]
xs[0][-1]
 ]
# idm_model =
# %%

idm_model.model_use = 'inference'
idm_model.batch_size = 1
input = np.array(xs[200])
input.shape = (1, 30, 3)
idm_model(input)

# .numpy()[0][0]+
