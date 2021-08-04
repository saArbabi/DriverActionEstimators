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
from viewer import Viewer
import matplotlib.pyplot as plt
import copy

def main():
    config = {'lanes_n':6,
            'lane_width':3.75, # m
            'lane_length':600 # m
            }
    env = EnvMC(config)
    viewer_real = Viewer(config)
    viewer_imagined = Viewer(config)

    while True:
        user_input = input()
        if user_input == 'n':
            sys.exit()
        # try:
        #     viewer.focus_on_this_vehicle = user_input
        # except:
        #     pass


        # neural_actions = []
        # for vehicle in env.neural_vehicles:
        #     vehicle.neighbours = vehicle.my_neighbours(env.vehicles)
        #     actions = vehicle.act(None)
        #     neural_actions.append(actions)

        env.step()
        viewer_real.render(env.real_vehicles)
        viewer_imagined.render(env.ima_vehicles)
        print(env.ima_vehicles[0].vehicle_type)
        print(env.ima_vehicles[0].act_long)
        print(env.ima_vehicles[0].speed)
        # print(env.ima_vehicles[0].id)
        # print(env.time_step)

# env.ima_vehicles[1].__dict__.items()

if __name__=='__main__':
    main()
# list(env.vehicles[0].__dict__.items())[0]
# vehicle_dir = dir(env.vehicles[0])
# %%
# C:\Users\sa00443\OneDrive - University of Surrey\190805 OneDrive Backup\Implementations\mcts_merge\sim\publication_results\quantitative
# class ViewerMC(Viewer):
#     def __init__(self, config):
#         super().__init__(config)
# attrnames = [attrname for attrname in dir(vehicle)
#
import types

class Dog:
    def __init__(self):
        self.a = 5
    def bark(self):
        print ("WOOF")
    def silence(self):
        print('wholy')

boby = Dog()
boby.bark() # WOOF




boby.bark() # WoOoOoF!!
def modify(item_):
    def _bark(self):
        self.a += 5
        self.silence()
        print ("WoOoOoF!!")
        return 'hi '
    item_.bark = types.MethodType(_bark, item_)

modify(boby)
x = boby.bark() # WoOoOoF!!
x
boby.a
2 == (3 or 5)
