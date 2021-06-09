from factory.sdv_vehicle import  SDVehicle
from factory.environment import Env
import sys

env = Env()
sdv = SDVehicle(id='sdv', lane_id=2, x=70, v=20)
env.vehicles = [sdv]
env.sdv = sdv

env.render('model_type')



def main():
    for i in range(100):
        env.render()
        # attention_logic(env)

        # if env.env_clock > 0 and  round(env.env_clock, 1) % 1 == 0:
        answer = input()
        if answer == 'n':
            sys.exit()

        # decision = int(input('Decision?'))


        if sdv.time_budget > 0:
            sdv.step(sdv_action)

            # for vehicle in self.vehicles:#
            #     if vehicle.id == 'sdv':
            #         vehicle.step(sdv_action)
            #     else:
            #         action = vehicle.act()
            #         vehicle.step(action)

            sdv.time_budget -= 0.1
        else:
            decision = input('Decision? (2, 5, 8)')
            while decision not in ['2', '5', '8']:
                decision = input('Choose again!')

            decision = int(decision)
            sdv_action = sdv.act(decision, None)

        env.step(sdv_action)
        print(sdv.x)

if __name__=='__main__':
    main()
