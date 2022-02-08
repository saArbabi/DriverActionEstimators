import sys
import os
import json
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVAL
mc_run_name = 'rwse_test'
config_name = 'rwse_test'

eval_config_dir = './src/evaluation/config_files/'+ config_name +'.json'

def read_eval_config(config_name):
    with open(eval_config_dir, 'rb') as handle:
        eval_config = json.load(handle)
    return eval_config

def main():
    exp_dir = './src/evaluation/mc_collections/'+ mc_run_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    eval_config = read_eval_config(config_name)
    eval_obj = MCEVAL(eval_config)
    eval_obj.mc_run_name = mc_run_name
    eval_obj.eval_config_dir = eval_config_dir
    eval_obj.run()

if __name__=='__main__':
    main()
