from factory.sdv_vehicle import  SDVehicle
from factory.environment import Env
import sys
from factory.vehicles import  *

env = Env()
sdv = SDVehicle(id='sdv', lane_id=2, x=70, v=20)
follower_IDM = IDMVehicle(id='follower', lane_id=1, x=20, v=20, driver_type='normal_idm')
leader = LeadVehicle(id='leader', lane_id=1, x=100, v=20)
follower_IDM.lead_vehicle = leader
env.vehicles = {}
vehicles = [sdv, follower_IDM, leader]
# env.sdv = sdv
for vehicle in vehicles:
    env.vehicles[vehicle.id] = vehicle
env.vehicle_ids = list(env.vehicles.keys())

env.render('model_type')
# env.vehicles['sdv'].update_obs_history(o_t)

OPTIONS = {
            0: ['LK', 'UP'],
            1: ['LK', 'DOWN'],
            2: ['LK', 'IDLE'],
            3: ['LCL', 'UP'],
            4: ['LCL', 'DOWN'],
            5: ['LCL', 'IDLE'],
            6: ['LCR', 'UP'],
            7: ['LCR', 'DOWN'],
            8: ['LCR', 'IDLE']
            }
OPTION_keys = list(OPTIONS.keys())
OPTION_keys = [str(key) for key in OPTION_keys]

def main():
    o_t = env.step()
    env.vehicles['sdv'].update_obs_history(o_t)
    # env.viewer.
    while True:
        # if env.env_clock > 0 and  round(env.env_clock, 1) % 1 == 0:
        answer = input()
        if answer == 'n':
            sys.exit()


        if sdv.time_budget > 0:
            sdv.time_budget -= 1
        else:
            decision = input(OPTIONS)
            while decision not in OPTION_keys:
                decision = input('Choose again!')
                if decision == 'n':
                    sys.exit()

            decision = int(decision)
            sdv_action = sdv.act(decision, None)

            action_plan = env.vehicles['sdv'].get_action_plan(sdv_action[-1])
            obs_history = env.vehicles['sdv'].obs_history
            prior_param = env.vehicles['sdv'].get_belief(obs_history, action_plan)
            sampled_att_z, sampled_idm_z = env.vehicles['sdv'].belief_net.sample_z(prior_param)
            att_scores =  env.vehicles['sdv'].arbiter(sampled_att_z).numpy().tolist()
            trace_i = 0
            print(att_scores)
            if env.viewer.attention_values:
                for trace in att_scores:
                    env.viewer.attention_values[trace_i].extend(att_scores[trace_i])
                    trace_i += 1
            else:
                env.viewer.attention_values = att_scores


        o_t = env.step(sdv_action)
        env.vehicles['sdv'].update_obs_history(o_t)
        env.vehicles['sdv'].update_action_history(sdv_action[-1])
        print(env.vehicles['sdv'].v)
        print(env.vehicles['leader'].v)
        env.render()


if __name__=='__main__':
    main()
