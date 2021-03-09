import numpy as np
import pickle
# %%
idm_param = {
                'desired_v':12, # m/s
                'desired_tgap':1.5, # s
                'min_jamx':2, # m
                'max_acc':1.4, # m/s^2
                'max_decc':2, # m/s^2
                }
"""
Synthetic data specs
"""
xs = []
ys = []
history_len = 30
lead_x_init = 100
desired_v = idm_param['desired_v']
desired_tgap = idm_param['desired_tgap']
min_jamx = idm_param['min_jamx']
max_acc = idm_param['max_acc']
max_decc = idm_param['max_decc']
sample_size = 1000
x_range = range(40, 80)
v_range = range(5, 10)
leader_v = 10

def get_desired_gap(vel, dv):
    gap = min_jamx + desired_tgap*vel+(vel*dv)/ \
                                    (2*np.sqrt(max_acc*max_decc))
    return gap

def act(vel, obs):
    desired_gap = get_desired_gap(vel, obs['dv'])
    acc = max_acc*(1-(vel/desired_v)**4-\
                                        (desired_gap/obs['dx'])**2)
    return sorted([-3, acc, 3])[1]


lead_x = [lead_x_init]
for i in range(history_len):
    lead_x.append(lead_x[i]+leader_v*0.1)

for sample_i in range(sample_size):
    rear_lead_x_init = np.random.choice(x_range)
    rear_v_init = np.random.choice(v_range)
    rear_x = [rear_lead_x_init]
    rear_v = [rear_v_init]
    sample_xs = []
    for i in range(history_len):
        obs = {'dx':lead_x[i]-rear_x[i], 'dv':leader_v-rear_v[i]}
        acc = act(rear_v[i], obs)
        vel = rear_v[i] + acc * 0.1
        x = rear_x[i] + rear_v[i] * 0.1 \
                                    + 0.5 * acc * 0.1 **2
        rear_x.append(x)
        rear_v.append(vel)
        sample_xs.append([rear_v[i], leader_v-rear_v[i], lead_x[i]-rear_x[i]])
    xs.append(sample_xs)
    ys.append([acc])

with open("./exploratory/x_vals", "wb") as fp:
    pickle.dump(xs, fp)

with open("./exploratory/y_vals", "wb") as fp:
    pickle.dump(ys, fp)
