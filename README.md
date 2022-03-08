# Neural Intelligent Driver Model (NIDM)

This repository is the official implementation of NIDM. Using a sequence of past traffic states, NIDM produces distributions over the sequence of future vehicle actions.
<!-- # [I](https://arxiv.org/abs/2030.12345). -->

## Training

To train the model(s), go to the models/model_trainers folder and
run the script for any of the models.

## Evaluation
Evaluations are done by collecting bunch of metrics from a set number of simulation episodes. The config for the
setup is in the evaluation folder. To initiate a MC run, run the eval_run.py script (e.g., from powershell).

eval_obj.py allows you to start monte carlo simulations from any episode. Simply
put in the config file the number of episodes you want, and then run the eval_run.py
script. New MC logs are continually stored in experiments folder.

## Data Preprocessing
The synthetic dataset we use for training contains 500 driving episodes which amount to roughly three hours of driving by 1500 drivers each with unique IDM parameters.
The following set of features are chosen to represent the local traffic context from ego car's viewpoint:
