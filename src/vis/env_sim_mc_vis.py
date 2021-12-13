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
import tensorflow as tf

def main():
    config = {'lanes_n':2,
            'lane_width':3.75, # m
            'lane_length':300 # m
            }
    env = EnvMergeMC(config)
    episode_id = 510
    np.random.seed(episode_id)
    env.transition_time = np.random.randint(0, 50) # vehicle_type = 'neural'

    env.initialize_env(episode_id)
    # model_name = 'neural_028'
    # model_name = 'neural_idm_107'
    model_name = 'latent_mlp_01'
    epoch_count = '15'
    data_id = '028'

    model_vehicle_map = {'neural_idm_107': 'NeuralIDMVehicle',
            'neural_029': 'NeuralVehicle',
            'latent_mlp_01': 'LatentMLPVehicle'
                                            }
    if model_vehicle_map[model_name] == 'NeuralVehicle':
        from vehicles.neural.neural_vehicle import NeuralVehicle
        env.neural_vehicle = NeuralVehicle()
    elif model_vehicle_map[model_name] == 'NeuralIDMVehicle':
        from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
        env.neural_vehicle = NeuralIDMVehicle()
    elif model_vehicle_map[model_name] == 'LatentMLPVehicle':
        from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
        env.neural_vehicle = LatentMLPVehicle()

    env.neural_vehicle.initialize_agent(
                        model_name, epoch_count, data_id)
    viewer = ViewerMC(config)
    tf.random.set_seed(0)
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
            sys.exit()

if __name__=='__main__':
    main()
