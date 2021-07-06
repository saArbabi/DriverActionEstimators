from highway import Env
from viewer import Viewer

def main():
    config = {'lanes_n':6,
            'lane_width':3.7, # m
            'lane_length':400 # m
            }
    env = Env(config)
    viewer = Viewer(config)
    # for i in range(100):
    while True:
        # if env.time_step > 200:
        # if env.time_step > 640:
        if env.time_step > 1740:
            decision = input()
            if decision == 'n':
                sys.exit()
            try:
                viewer.focus_on_this_vehicle = int(decision)
            except:
                pass
            viewer.render(env.vehicles)
            print(env.time_step)

        env.step()


if __name__=='__main__':
    main()
