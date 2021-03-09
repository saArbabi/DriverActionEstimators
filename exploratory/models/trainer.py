import matplotlib.pyplot as plt
from exploratory.models.neural import Encoder
from importlib import reload
import tensorflow as tf
import pickle

config = {
 "model_config": {
     "learning_rate": 1e-3,
    "batch_size": 50,
    },
    "exp_id": "NA",
    "Note": ""
}
# from exploratory.models import neural
# reload(neural)
# from exploratory.models.neural import  Encoder
from exploratory.models import idm_neural
reload(idm_neural)
from exploratory.models.idm_neural import  Encoder
# (1) Load model and setup checkpoints
# model = Encoder(config)
model = Encoder(config, model_use='training')

# for more on checkpointing model see: https://www.tensorflow.org/guide/checkpoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model) # no need for optimizer for now
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
exp_dir = './experiments/idm_neural/model_dir'
manager = tf.train.CheckpointManager(ckpt, exp_dir, max_to_keep=None)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# (2) Load data
with open("./exploratory/y_vals", "rb") as fp:
    ys = pickle.load(fp)

with open("./exploratory/x_vals", "rb") as fp:
    xs = pickle.load(fp)
data_objs = [xs, ys]

# (3) Run experiment
write_graph = 'True'
for epoch in range(5):
    model.train_loop([xs, ys])
    # model.test_loop([xs, ys], epoch)

    ckpt.step.assign_add(1)
    if int(ckpt.step) % 5 == 0:
        save_path = manager.save()
