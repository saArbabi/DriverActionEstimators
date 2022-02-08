import sys
sys.path.insert(0, './src')
from viewer import Viewer
from envs.merge import EnvMerge
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('./src/envs/config.json', 'rb') as handle:
        config = json.load(handle)
    env = EnvMerge(config)
    # episode_id = 504
    # episode_id = 506
    episode_id = 23
    env.initialize_env(episode_id)

    viewer = Viewer(config)
    while True:
        if env.time_step >= 0:
            user_input = input()
            if user_input == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(user_input)
            except:
                pass
            viewer.render(env.vehicles)
            print(env.time_step)

        env.step()

if __name__=='__main__':
    main()
