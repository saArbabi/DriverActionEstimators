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

    # model_name = 'neural_028'
    model_name = 'neural_idm_119'
    # model_name = 'latent_mlp_02'
    # model_name = 'mlp_01'
    # model_name = 'lstm_01'
    data_id = '031'
    history_len = 30 # choose this based on the model with longest history
    rollout_len = 50

    model_vehicle_map = {
            'neural_idm_119': 'NeuralIDMVehicle',
            'neural_032': 'NeuralVehicle',
            'latent_mlp_08': 'LatentMLPVehicle',
            'mlp_01': 'MLPVehicle',
            'lstm_01': 'LSTMVehicle'}
    if model_vehicle_map[model_name] == 'NeuralVehicle':
        epoch_count = '20'
        from vehicles.neural.neural_vehicle import NeuralVehicle
        env.neural_vehicle = NeuralVehicle()
    elif model_vehicle_map[model_name] == 'NeuralIDMVehicle':
        epoch_count = '20'
        from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
        env.neural_vehicle = NeuralIDMVehicle()
    elif model_vehicle_map[model_name] == 'LatentMLPVehicle':
        epoch_count = '20'
        from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
        env.neural_vehicle = LatentMLPVehicle()
    elif model_vehicle_map[model_name] == 'MLPVehicle':
        epoch_count = '20'
        from vehicles.neural.mlp_vehicle import MLPVehicle
        env.neural_vehicle = MLPVehicle()
    elif model_vehicle_map[model_name] == 'LSTMVehicle':
        epoch_count = '10'
        from vehicles.neural.lstm_vehicle import LSTMVehicle
        env.neural_vehicle = LSTMVehicle()

    episode_id = 504
    trace = 0
    np.random.seed(episode_id)
    env.trans_time = np.random.randint(\
                history_len, history_len*2) # controller ==> 'neural'
    env.neural_vehicle.initialize_agent(
                        model_name, epoch_count, data_id)
    env.initialize_env(episode_id)
    viewer = ViewerMC(config)
    tf.random.set_seed(trace)
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
