from envs.merge import EnvMerge
from viewer import Viewer
import matplotlib.pyplot as plt
import numpy as np

def main():
    config = {'lanes_n':2,
            'lane_width':3.75, # m
            'lane_length':300 # m
            }
    env = EnvMerge(config)
    env.initialize_env(61)

    viewer = Viewer(config)
    # np.random.seed(0)
    while True:
        # if env.time_step > 200:
        # if env.time_step > 640:
        if env.time_step >= 0:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            viewer.render(env.vehicles)
            # viewer.render(env.vehicles, None)
            print(env.time_step)

        env.step()
        # print(env.time_step)

if __name__=='__main__':
    main()
#
# def get_animation():
#     plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/sa00443/ffmpeg_programs/ffmpeg.exe'
#     from matplotlib.animation import FuncAnimation, writers
#     for i_ in range(1000):
#         env.step()
#
#     def animation_frame(i):
#         viewer.render(env.vehicles)
#         env.step()
#         # return line,
#
#     animation = FuncAnimation(viewer.fig, func=animation_frame, \
#                               frames=range(200), interval=1000)
#
#
#     # setting up wrtiers object
#     Writer = writers['ffmpeg']
#     writer = Writer(fps=10, metadata={'artist': 'Me'}, bitrate=3000)
#     animation.save('sim_example.mp4', writer, dpi=250)
# get_animation()
