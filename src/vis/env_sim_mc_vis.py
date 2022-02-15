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
import json

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvMergeMC(config)

    # model_name = 'latent_mlp_12'
    model_name = 'neural_idm_238'
    model_name = 'neural_037'
    # model_name = 'latent_mlp_02'
    # model_name = 'mlp_02'
    # model_name = 'lstm_02'
    data_id = '046'
    history_len = 20 # choose this based on the model with longest history
    rollout_len = 50

    model_vehicle_map = {
            'neural_idm_238': 'NeuralIDMVehicle',
            'neural_037': 'NeuralVehicle',
            'latent_mlp_12': 'LatentMLPVehicle',
            'mlp_02': 'MLPVehicle',
            'lstm_02': 'LSTMVehicle'}

    if model_vehicle_map[model_name] == 'NeuralVehicle':
        epoch_count = '10'
        from vehicles.neural.neural_vehicle import NeuralVehicle
        env.neural_vehicle = NeuralVehicle()
    elif model_vehicle_map[model_name] == 'NeuralIDMVehicle':
        epoch_count = '10'
        from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
        env.neural_vehicle = NeuralIDMVehicle()
    elif model_vehicle_map[model_name] == 'LatentMLPVehicle':
        epoch_count = '15'
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

    episode_id = 509 # wrong switch to 1
    # episode_id = 505
    # episode_id = 506 # late switch
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
        # if env.collision_detected:
        #     sys.exit()

if __name__=='__main__':
    main()
