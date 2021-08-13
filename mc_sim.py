"""
Collisions
Hard brakes
RWSE
"""
import os
# os.chdir('../../')
# print('directory: ' + os.getcwd())
# directory: C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim

from highway import EnvMC
from viewer import ViewerMC
import matplotlib.pyplot as plt
import copy

def main():
    config = {'lanes_n':6,
            'lane_width':3.75, # m
            'lane_length':400 # m
            }
    env = EnvMC(config)
    viewer = ViewerMC(config)
    while True:
        if env.time_step > 100:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            print(env.time_step)
            viewer.render(env.real_vehicles, env.ima_vehicles)
            viewer.info_plot(env.real_mc_log, env.ima_mc_log)
        env.step()
        # print(env.ima_vehicles[0].vehicle_type)
        # print(env.ima_vehicles[0].act_long)
        # print(env.ima_vehicles[0].speed)
        # print(env.ima_vehicles[0].id)
        # print(env.time_step)

# env.ima_vehicles[1].__dict__.items()

if __name__=='__main__':
    main()
