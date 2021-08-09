"""
Collisions
Hard brakes
RWSE
"""
import os
# os.chdir('../../')
# print('directory: ' + os.getcwd())
# directory: C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim

from highway import EnvLaneKeepMC
from viewer import ViewerLaneKeep
import matplotlib.pyplot as plt
import copy

def main():
    config = {'lanes_n':1,
            'lane_width':3.75, # m
            'lane_length':400 # m
            }
    env = EnvLaneKeepMC(config)
    viewer = ViewerLaneKeep(config)

    while True:
        user_input = input()
        if user_input == 'n':
            sys.exit()

        print(env.time_step)
        env.step()
        viewer.render(env.real_vehicles, env.ima_vehicles)
        # print(env.ima_vehicles[0].vehicle_type)
        # print(env.ima_vehicles[0].act_long)
        # print(env.ima_vehicles[0].speed)
        # print(env.ima_vehicles[0].id)
        # print(env.time_step)

# env.ima_vehicles[1].__dict__.items()

if __name__=='__main__':
    main()
