"""
Collisions
Hard brakes
RWSE
"""
import sys
sys.path.insert(0, './src')
from envs.merge_mc import EnvMergeMC
from viewer import ViewerMC
import numpy as np
from vehicles.neural_vehicles import NeuralIDMVehicle, NeurLatentVehicle
import tensorflow as tf

def main():
    config = {'lanes_n':2,
            'lane_width':3.75, # m
            'lane_length':300 # m
            }
    env = EnvMergeMC(config)
    episode_id = 20
    env.initialize_env(episode_id)
    model_name = 'h_z_f_idm_act_101'
    epoch_count = '20'
    data_id = '026'
    # env.neural_vehicle = MLPVehicle()
    # env.neural_vehicle = NeurLatentVehicle()
    # env.neural_vehicle = LSTMVehicle()
    env.neural_vehicle = NeuralIDMVehicle()
    env.neural_vehicle.initialize_agent(
                        model_name, epoch_count, data_id)
    viewer = ViewerMC(config)
    tf.random.set_seed(10)
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
        if env.collision_detected:
            print('collision_detected')
            sys.exit()

if __name__=='__main__':
    main()
