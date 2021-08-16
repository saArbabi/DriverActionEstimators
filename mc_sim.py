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
import numpy as np
from vehicles.neural_vehicles import NeuralIDMVehicle, LSTMVehicle

def main():
    config = {'lanes_n':6,
            'lane_width':3.75, # m
            'lane_length':800 # m
            }
    env = EnvMC(config)
    env.neural_vehicle = NeuralIDMVehicle()
    viewer = ViewerMC(config)
    np.random.seed(2021)
    while True:
        if env.time_step > 50:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            print(env.time_step)
            viewer.render(env.real_vehicles, env.ima_vehicles)
            # if 10 in env.real_mc_log:
            #     viewer.info_plot(env.real_mc_log, env.ima_mc_log)
        env.step()
        # print(env.ima_vehicles[0].vehicle_type)
        # print(env.ima_vehicles[0].act_long)
        # print(env.ima_vehicles[0].speed)
        # print(env.ima_vehicles[0].id)
        # print(env.time_step)

# env.ima_vehicles[1].__dict__.items()

if __name__=='__main__':
    main()
