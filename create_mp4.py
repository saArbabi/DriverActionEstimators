# %%
def get_animation():
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
