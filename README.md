# Paper intro

This repository is the official implementation of DriverActionEstimation
<!-- # [I](https://arxiv.org/abs/2030.12345). -->

## Training

To train the model(s) in the paper, go to the models/model_trainers folder and
run the the script for the any of the models presented in the paper.

## Evaluation
Evaluations are done by collecting bunch of metrics from a set number of simulation episodes. The config for the
setup is in the evaluation folder. To initiate a MC run, run the eval_run.py script (e.g., from powershell).

eval_obj.py allows you to start monte carlo simulations from any episode. Simply
put in the config file the number of episodes you want, and then run the eval_run.py
script. New MC logs are continually stored in experiments folder.

## Data Preprocessing
