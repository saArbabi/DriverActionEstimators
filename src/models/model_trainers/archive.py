"""Some code that is most probably useless
"""
"""
Driver model - lstm
"""
history_future_seqs = data_gen.sequence(features, 20, 1)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 20, 1)
data_list = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_list = [data_array[:5000, :, :] for data_array in data_list]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_list

future_e_veh_a.shape
# %%
"""
Driver model - mlp
"""
history_future_seqs = data_gen.sequence(features, 1, 1)
history_future_seqs_scaled = data_gen.sequence(features_scaled, 1, 1)
data_list = data_gen.split_data(history_future_seqs, history_future_seqs_scaled)
# data_list = [data_array[:5000, :, :] for data_array in data_list]

history_future_usc, history_sca, future_sca, future_idm_s, \
                future_m_veh_c, future_e_veh_a = data_list

future_e_veh_a.shape

history_sca.flatten().shape
future_e_veh_a[0]
history_future_usc[0]
#########################################################
#########################################################
