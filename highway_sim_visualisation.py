from highway import Env
from viewer import Viewer

def main():
    config = {'lanes_n':4,
            'lane_width':3.7, # m
            'lane_length':1200 # m
            }
    env = Env(config)
    viewer = Viewer(config)
    # for i in range(100):
    while True:
        decision = input()
        if decision == 'n':
            sys.exit()
        try:
            viewer.focus_on_this_vehicle = int(decision)
        except:
            pass

        env.step()
        viewer.render(env.vehicles)
        # print(env.elapsed_time)


if __name__=='__main__':
    main()
