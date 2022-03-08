# Neural Intelligent Driver Model (NIDM)

This repository is the implementation of NIDM. Using a sequence of past traffic states, NIDM produces distributions over the sequence of future vehicle actions.
<!-- # [I](https://arxiv.org/abs/2030.12345). -->

## Training

To train the model(s), go to the models/model_trainers folder and
run the train.py script for any of the models.

We can visualise the latent encoding while NIDM is being trained:

![alt text](https://github.com/saArbabi/DriverActionEstimators/blob/main/Figures/latent_evolution.gif)

We train the model for 20 epochs (which amounts to 3220 training iterations when using a mini-batch size of 512 samples).

## Evaluation
Evaluations are done by collecting several metrics from a set 2100 policy rollouts (i.e., Monte Carlo simulations). The configuration for the evaluation setup is located in the evaluation folder. To initiate the evaluation experiments, run the eval_run.py script (e.g., from powershell).

Note: eval_obj.py allows you to start Monte Carlo simulations from any driving episode. Simply
put in the config file the number of episodes you want, and then run the eval_run.py
script.

## Data Preprocessing
The synthetic dataset we use for training contains 500 driving episodes which amount to roughly three hours of driving by 1500 drivers each with unique IDM parameters.
The following set of features are used to represent the local traffic context from ego car's viewpoint:
![alt text](https://github.com/saArbabi/DriverActionEstimators/blob/main/Figures/features.PNG)
