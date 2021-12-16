"""
These methods need to be placed in the correct scrips. You may refactor them
when they are needed.
"""
def get_animation():
    """To show case how the latent spaces changes with more training.
    """
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers

    def latent_samples(model, sample_index):
        merger_cs = future_m_veh_c[sample_index, :, 2:]
        h_seq = history_sca[sample_index, :, 2:]
        enc_h = model.h_seq_encoder(h_seq)
        latent_dis_param = model.belief_net(enc_h, dis_type='prior')
        sampled_z, sampled_idm_z = model.belief_net.sample_z(latent_dis_param)
        return sampled_z, sampled_idm_z

    fig = plt.figure(figsize=(7, 7))
    plt.style.use('ggplot')
    ax = fig.add_subplot(211)
    idm_axis = fig.add_subplot(212)

    def animation_frame(i):
        model.vae_loss_weight = 0.1
        model.train(data_arrays, epochs=1)
        sampled_z, sampled_idm_z = latent_samples(model, val_samples)
        aggressiveness = history_future_usc[val_samples, 0, -1]
        color_shade = aggressiveness
        ax.scatter(sampled_z[:, 0], sampled_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')
        idm_axis.scatter(sampled_idm_z[:, 0], sampled_idm_z[:, 1], s=15, alpha=0.3, \
                                                    c=color_shade, cmap='rainbow')

        ax.set_title('Iteration ' + str(i))
        ax.set_ylabel('$z_{att_1}$')
        ax.set_xlabel('$z_{att_2}$')
        idm_axis.set_ylabel('$z_{idm_1}$')
        idm_axis.set_xlabel('$z_{idm_1}$')

    animation = FuncAnimation(fig, func=animation_frame, \
                              frames=range(1, 81), interval=1)

    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=4, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('latent_evolution.mp4', writer, dpi=250)
# get_animation()
# %%
def get_animation():
    """To show traffic simulation
    """
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
    from matplotlib.animation import FuncAnimation, writers
    for i_ in range(1000):
        env.step()

    def animation_frame(i):
        viewer.render(env.vehicles)
        env.step()
        # return line,

    animation = FuncAnimation(viewer.fig, func=animation_frame, \
                              frames=range(100), interval=1000)


    # setting up wrtiers object
    Writer = writers['ffmpeg']
    writer = Writer(fps=25, metadata={'artist': 'Me'}, bitrate=3000)
    animation.save('sim_example.mp4', writer, dpi=250)
# get_animation()
# plt.show()
# %%
