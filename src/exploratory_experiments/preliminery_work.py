import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from models.core.preprocessing import utils
from math import hypot
import json
from importlib import reload

cwd = os.getcwd()

# %%
"""
m_df - merge_mveh_df
y_df - yield_mveh_df
"""
col = ['id','frm','scenario','lane_id',
                                'bool_r','bool_l','pc','vel',
                                'a_long','act_lat','a_lat','e_class',
                                'ff_id','ff_long','ff_lat','ff_v',
                                'fl_id','fl_long','fl_lat','fl_v',
                                'bl_id','bl_long','bl_lat','bl_v',
                                'fr_id','fr_long','fr_lat','fr_v',
                                'br_id','br_long','br_lat','br_v',
                                'bb_id','bb_long','bb_lat','bb_v']

col_drop = ['bool_r','bool_l','a_lat','a_long',
                    'fr_id','fr_long','fr_lat','fr_v',
                    'fl_id','fl_long','fl_lat','fl_v']
datasets = {
        "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
        "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
        "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
        "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
        "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
        "i80_3": "trajdata_i80_trajectories-0515-0530.txt"}

col_df_all = ['id','frm','scenario','lane_id','length','x_front','y_front','class']

# %%
feature_set = pd.read_csv('./datasets/feature_set.txt', delimiter=' ',
                        header=None, names=col).drop(col_drop,axis=1)

df_all = pd.read_csv('./datasets/df_all.txt', delimiter=' ',
                                                            header=None, names=col_df_all)

 # %%
 # 2215 i80_2 5088.0 1543 1548.0 1538.0 1539.0 74

follower = feature_set[(feature_set['id'] == 1548) & (feature_set['scenario'] == \
                                                    'i80_2')].reset_index(drop=True)
leader_1 = feature_set[(feature_set['id'] == 1537) & (feature_set['scenario'] == \
                                                    'i80_2')].reset_index(drop=True)
leader_2 = feature_set[(feature_set['id'] == 1538) & (feature_set['scenario'] == \
                                                    'i80_2')].reset_index(drop=True)
leader_3 = feature_set[(feature_set['id'] == 1543) & (feature_set['scenario'] == \
                                                    'i80_2')].reset_index(drop=True)

ff_id[ff_id.diff() != 0].index
lane = follower['lane_id']
frm = follower['frm']
frm[0]
chane_point_1 = frm[471]
chane_point_2 = frm[608]
ff_id = follower['ff_id']
# %%
overlap = 20
vel_1 = leader_1[(leader_1['frm'] >= 4480) &
                (leader_1['frm'] <= 4951+overlap)]['vel']
plt.plot(vel_1.values)
plt.plot(follower[0:471+overlap]['vel'].values)
plt.legend(['leader','follower'])
plt.scatter([overlap, len(vel_1)-overlap],[follower['vel'][0],follower['vel'][471]])
plt.xlabel('frame(N)')
plt.ylabel('speed($ms_{-1}$)')
# %%
follower[0:1]['vel']
follower[0:1]['ff_v']
follower[0:1]['ff_long']
# IDM params:


# %%
desired_vel = 5.6
desired_tgap = 2.8
vel_0 = 5.6
min_jamx = 0
max_acc = 3
max_decc = 3

def get_desired_gap(vel, dv):
    return min_jamx + desired_tgap*vel+(vel*dv)/(2*np.sqrt(max_acc*max_decc))

def IDM_act(dx, dv, vel, IDM_param):
    """outputs IDM's action for the current state
    """
    # acc = max_acc*(1-(get_desired_gap(vel, dv)/dx)**2)
    acc = max_acc*(1-(vel/desired_vel)**4-(get_desired_gap(vel, dv)/dx)**2)

    return acc

vels = [5.6]
dxs = [15.5]
accs = []
for i in range(471):
    leader_state = leader_1.iloc[i]
    # now
    dv = leader_state['vel'] - vels[i]
    # compute
    acc = IDM_act(dxs[i], dv, vels[i], IDM_param=None)
    accs.append(acc)
    # next
    vel = vels[i]+acc*0.1
    vels.append(vel)
    dx = dxs[i] + dv*0.1
    dxs.append(dx)


plt.plot(leader_1[0:471]['vel'])
plt.plot(follower[0:471+overlap]['vel'].values)
plt.plot(vels)

plt.legend(['leader','follower_true', 'follower_IDM'])
plt.xlabel('frame(N)')
plt.ylabel('speed($ms_{-1}$)')
# %%

plt.plot(accs)
plt.plot(dxs)
# %%
vel_2 = leader_2[(leader_2['frm'] >= 4951-overlap) &
                (leader_2['frm'] <= 5088+overlap)]['vel']
plt.plot(vel_2.values)
plt.plot(follower[471-overlap:608+overlap]['vel'].values)
plt.legend(['leader','follower'])
plt.scatter([overlap, len(vel_2)-overlap],[follower['vel'][471],follower['vel'][608]])
plt.xlabel('frame(N)')
plt.ylabel('speed($ms_{-1}$)')

# %%
vel_3 = leader_3[(leader_3['frm'] >= 5088-overlap)]['vel']
plt.plot(vel_3.values)
plt.plot(follower[608-overlap:]['vel'].values)
plt.legend(['leader','follower'])
plt.scatter([overlap, len(vel_3)-overlap],[follower['vel'][608],follower['vel'][797-overlap]])
plt.xlabel('frame(N)')
plt.ylabel('speed($ms_{-1}$)')

# %%

# %%
plt.plot(lane.values)
plt.plot(follower['vel'].values)
plt.plot(leader['vel'].values[0:200])
plt.plot(lane)
