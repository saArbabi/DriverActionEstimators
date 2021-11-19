def latent_samples(model_trainer, sample_index):
    h_seq = history_sca[sample_index, :, 2:]
    merger_cs = future_m_veh_c[sample_index, :, 2:]
    enc_h = model_trainer.model.h_seq_encoder(h_seq)
    latent_dis_param = model_trainer.model.belief_net(enc_h, dis_type='prior')
    sampled_z = model_trainer.model.belief_net.sample_z(latent_dis_param)
    return sampled_z

def latent_vis(n_z_samples):
    fig = pyplot.figure(figsize=(4, 6))
    examples_to_vis = np.random.choice(val_examples, n_z_samples, replace=False)
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    sampled_z = latent_samples(model_trainer, examples_to_vis)
    aggressiveness = history_future_usc[examples_to_vis, 0, -7]
    color_shade = aggressiveness
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    ax.tick_params(pad=1)
    ax.grid(False)
    # ax.view_init(30, 50)
    #===============
    #  Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    att_sc = ax.scatter(sampled_z[:, 0], sampled_z[:, 1], sampled_z[:, 2],
                  s=5, c=color_shade, cmap='rainbow', edgecolors='black', linewidth=0.2)

    axins = inset_axes(ax,
                        width="5%",
                        height="90%",
                        loc='right',
                        borderpad=-2
                       )

    fig.colorbar(att_sc, cax=axins)
    cbar = fig.colorbar(att_sc, cax=axins)
    ax.tick_params(pad=1)
    ax.grid(False)
    ax.view_init(30, 50)
    # ax.set_xlabel('$z_{1}$', labelpad=1)
    # ax.set_ylabel('$z_{2}$', labelpad=1)
    # ax.set_zlabel('$z_{3}$', labelpad=1)
    plt.subplots_adjust(wspace=0.2, hspace=None)

def vectorise(step_row, traces_n):
    return np.repeat(step_row, traces_n, axis=0)

def fetch_traj(data, sample_index, colum_index):
    # data: [sample_index, time, feature]
    traj = np.delete(data[sample_index, :, colum_index:colum_index+1], 19, axis=1)
    return traj.flatten()

def pickle_this(item_address, item, item_name):
    item_address += item_name+'.pickle'
