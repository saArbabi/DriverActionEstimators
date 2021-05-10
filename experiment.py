import os
import pickle
import sys
from factory.environment import Env
from factory.vehicles import *

def set_follower(lane_id, model_type, model_name, driver_type):
    config = {
             "model_config": {
                 "learning_rate": 1e-3,
                "batch_size": 50,
                },
                "exp_id": "NA",
                "Note": ""}

    exp_dir = './models/experiments/'+model_name+'/model'

    # with open('./models/experiments/scaler.pickle', 'rb') as handle:
    #     scaler = pickle.load(handle)

    if model_type == 'dnn':
        from models.core.dnn import  Encoder
        model = Encoder(config)
        model.load_weights(exp_dir).expect_partial()
        follower = DNNVehicle(id='neural', lane_id=lane_id, x=40, v=20,
                        driver_type=driver_type, model=model)

    if model_type == 'lstm':
        from models.core.lstm import  Encoder
        model = Encoder(config)
        model.load_weights(exp_dir).expect_partial()
        follower = LSTMVehicle(id='neural', lane_id=lane_id, x=40, v=20,
                        driver_type=driver_type, model=model)

    if model_type == 'lstm_idm':
        from models.core.lstm_idm import  Encoder
        model = Encoder(config, model_use='inference')
        model.load_weights(exp_dir).expect_partial()
        follower = LSTMIDMVehicle(id='neural', lane_id=lane_id, x=40, v=20,
                        driver_type=driver_type, model=model)

    if  model_type == 'lstm_seq_idm':
        from models.core.lstm_seq_idm import  Encoder
        model = Encoder(config, model_use='inference')
        model.load_weights(exp_dir).expect_partial()
        follower = LSTMIDMVehicle(id='neural', lane_id=lane_id, x=40, v=20,
                        driver_type=driver_type, model=model)

    if  model_type == 'driver_model':
        from models.core.driver_model import  Encoder
        model = Encoder(config, model_use='inference')
        model.load_weights(exp_dir).expect_partial()
        follower = NeurIDM(id='neural', lane_id=lane_id, x=40, v=20,
                        driver_type=driver_type, model=model)
    # follower.scaler = scaler
    return follower


env = Env()
#
# model_type='lstm_seq_idm'
# model_name='lstm_seq_idm'
# model_type='dnn'
# model_type='lstm'
# model_type='lstm_idm'

# model_type='lstm_idm'
model_name='lstm_seq2s_idm'
model_type='lstm_seq_idm'
model_type='driver_model'
model_name='driver_model'

leader1 = LeadVehicle(id='leader', lane_id=3, x=100, v=20)
leader2 = LeadVehicle(id='leader', lane_id=2, x=100, v=20)
leader3 = LeadVehicle(id='leader', lane_id=1, x=100, v=20)

neural_IDM1 = set_follower(lane_id=3, model_type=model_type, model_name=model_name,\
                                                            driver_type='aggressive_idm')
neural_IDM2 = set_follower(lane_id=2, model_type=model_type, model_name=model_name,\
                                                            driver_type='normal_idm')
neural_IDM3 = set_follower(lane_id=1, model_type=model_type, model_name=model_name,\
                                                            driver_type='timid_idm')

follower_IDM1 = IDMVehicle(id='aggressive_idm', lane_id=3, x=40, v=20, driver_type='aggressive_idm')
follower_IDM2 = IDMVehicle(id='normal_idm', lane_id=2, x=40, v=20, driver_type='normal_idm')
follower_IDM3 = IDMVehicle(id='timid_idm', lane_id=1, x=40, v=20, driver_type='timid_idm')

follower_IDM1.lead_vehicle = leader1
follower_IDM2.lead_vehicle = leader2
follower_IDM3.lead_vehicle = leader3

neural_IDM1.lead_vehicle = leader1
neural_IDM2.lead_vehicle = leader2
neural_IDM3.lead_vehicle = leader3
env.vehicles = [
                neural_IDM1,
                neural_IDM2,
                neural_IDM3,
                follower_IDM1,
                follower_IDM2,
                follower_IDM3,
                leader1,
                leader2,
                leader3]

env.render(model_type)
for i in range(5000):
    env.step()
    env.render()

    if env.elapsed_time > 0 and  round(env.elapsed_time, 1) % 10 == 0:
        answer = input('Continue?')
        if answer == 'n':
            sys.exit()
