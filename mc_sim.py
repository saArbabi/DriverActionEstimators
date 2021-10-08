"""
Collisions
Hard brakes
RWSE
"""
import os
# os.chdir('../../')
# print('directory: ' + os.getcwd())
# directory: C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim

from envs.merge_mc import EnvMergeMC
from viewer import ViewerMC
import numpy as np
# from vehicles.neural_vehicles import NeuralIDMVehicle, NeurLatentVehicle
# import tensorflow as tf

def main():
    config = {'lanes_n':2,
            'lane_width':3.75, # m
            'lane_length':600 # m
            }
    env = EnvMergeMC(config)
    # env.neural_vehicle = LSTMVehicle()
    # env.neural_vehicle = MLPVehicle()
    # env.neural_vehicle = NeuralIDMVehicle()
    # env.neural_vehicle = NeurLatentVehicle()
    viewer = ViewerMC(config)
    np.random.seed(0)
    # np.random.seed(2021)
    # tf.random.set_seed(0)
    env.debugging_mode = True
    # env.debugging_mode = False
    while True:
        if env.time_step > 0:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            print(env.time_step)
            viewer.render(env.real_vehicles, env.ima_vehicles)
            if env.debugging_mode:
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
