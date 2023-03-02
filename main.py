from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time
import keras
import warnings
import collections

import numpy as np
from utility import *
from test_data import *
import tensorflow as tf
from qinfo.qinfo import partial_trace2
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Input
from datetime import datetime
from keras import optimizers
from scipy.special import comb
from keras.initializers import RandomUniform, Identity
from keras.layers import Dense, ReLU
from keras import activations
from keras.callbacks import LearningRateScheduler, TensorBoard
from ucell.ucell import UParamLayer, RevDense, ReNormaliseLayer, ULayer


config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=12, 
                        inter_op_parallelism_threads=12, 
                        allow_soft_placement=True,
                        device_count = {'CPU': 24})

session = tf.compat.v1.Session(config=config)
#np.random.seed(1234)


# ------------------------------------------------------------
# Section 0: Program parameter definitions
# ------------------------------------------------------------
# define target modes of memristors
targets = [[1,2,3],[4,5,6]]
# define temporal depth
temporal_num = 1
# define total mode number
modes = 6
# define total number of single photons
photons = 3
# number of layers in neural network
neural_layers = 1
# number of logits in each layer
layer_units = 30
# number of epochs to train over
epochs = 15
# initial learning rate
init_lr = 5e-2
# batch size for learning
batch_size = 5
# learning rate polynomial
lr_pow = 0
# whether or not to train the network or just compile it
train = True
# whether to save resevoir mapping channel or recompute from scratch
save = True
# location and name of save file
file_name = "MNIST_MAP.npz"#"\\\\FREENAS\\Organon\\Research\\Papers\\QMemristor\\Code\\Archive\\MNIST_MAP.npz"
# location and name of model save
modelfile_name = ""#"\\\\FREENAS\\Organon\\Research\\Archive\\ML\\models\\checkpoint"
# task toggle ("witness" or "mnist")
task = "mnist" 
# ------------------------------------------------------------
# Section 1: define program constants
# ------------------------------------------------------------
# define spatial depth 
spatial_num = len(targets)
# define a single instance of memristor element as example desciption
pdesc = {'theta_init':0.1,'MZI_target': [1,2,2], "tlen":10}
# compute dimension of input network
dim = comb(modes+photons-1, photons, exact=True) 
# initialiser for dense network
init = RandomUniform(minval=-1, maxval=1, seed=None)
# ------------------------------------------------------------
# Section 2: Apply resevoir compute layer to quantum data encoding 
# ------------------------------------------------------------

if task == "witness":
	# generate random entangled and seperable states
	data_train, y_train, data_test, y_test = entanglement_gen(dim=10, num=5000, partition=0.5, embed_dim=dim)

	data_train = reservoir_map(data_train, modes, photons, pdesc, targets, temporal_num)
	data_test = reservoir_map(data_test, modes, photons, pdesc, targets, temporal_num)

else:

	# check if data has already been saved, else we will need to regenerate 
	if os.path.isfile(file_name):
		# load saved data
		save_data = np.load(file_name)
		# extract all data
		data_train = save_data["data_train"]
		data_test = save_data["data_test"]
		y_train = save_data["y_train"]
		y_test = save_data["y_test"]

	else:

		# load MNIST data
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

		x_train, y_train = filter_36(x_train, y_train)
		x_test, y_test = filter_36(x_test, y_test)

		# add channel axis (tensorflow being lame)
		x_train = x_train[...,tf.newaxis]
		x_test = x_test[...,tf.newaxis]
		# downsample for easy initial training
		data_train = np.squeeze(tf.image.resize(x_train, (14,14)).numpy())[:,1:-1,2:-2]
		data_test = np.squeeze(tf.image.resize(x_test, (14,14)).numpy())[:,1:-1,2:-2]
		#data_train[:,-1,:] = data_test[:,-1,:] = 1.0

		# remove conflicting training items
		data_train, y_train = remove_contradicting(data_train, y_train)
		data_test, y_test = remove_contradicting(data_test, y_test)

		# apply encoding of classical data to quantum state space
		encoder = QEncoder(modes=modes, photons=photons, density=True)
		data_train = encoder.encode(data=data_train, method="amplitude", normalise=True)
		data_test = encoder.encode(data=data_test, method="amplitude", normalise=True)

		#data_train = eigen_encode_map(data_train, modes, photons)
		#data_test = eigen_encode_map(data_test, modes, photons)

		# pass through resevoir
		data_train = reservoir_map(data_train, modes, photons, pdesc, targets, temporal_num)
		data_test = reservoir_map(data_test, modes, photons, pdesc, targets, temporal_num)

		# save this mapped data so we don't have to recompute
		if save:
			np.savez(file_name, data_train=data_train, 
								data_test=data_test, 
								y_train=y_train, 
								y_test=y_test)


	data_train = data_train[:1000]
	data_test = data_test[:1000]
	y_train = y_train[:1000]
	y_test = y_test[:1000]







# ------------------------------------------------------------
# Section 3: Define network topology
# ------------------------------------------------------------

# define input layer of network, no longer a time sequence to be considered
input_state = Input(batch_shape=[None, dim, dim], dtype=tf.complex64, name="state_input")


# extract classical output state of resevoir
#output = ULayer(modes, photons, force=True)(input_state)
output = MeasureLayer(modes, photons, force=True)(input_state)
# feed this into a small feedforward network

output = Dense(units=20)(output)
#output = ReLU(negative_slope=0.1, threshold=0.0)(output)
output = Dense(units=20)(output)
#output = ReLU(negative_slope=0.01, threshold=0.0)(output)
output = Dense(units=3, activation="softmax",  use_bias=True)(output)

# define standard optimiser
opt = optimizers.Adam(learning_rate=init_lr)

# define loss function
loss = tf.keras.losses.CategoricalCrossentropy(
				    from_logits=False,
				    label_smoothing=0,
				    reduction="auto",
				    name="categorical_crossentropy")

# define the model
model = Model(inputs=input_state, outputs=output, name="Optical_Resevoir_Compute_Network")

# ------------------------------------------------------------
# Section 4: Model compilation and training
# ------------------------------------------------------------

# compile it using probability fidelity
model.compile(optimizer=opt,
			  loss=loss,
			  metrics=["accuracy"])

# setup callbacks
#name = datetime.now().strftime("%Y%m%d_%H%M%S")
#logdir = "C:\\Users\\Joshua\\Projects\\Research\\Archive\\Logs\\fit\\"
#tensorboard_callback = TensorBoard(log_dir=logdir+name, write_graph=True)
schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=init_lr, power=lr_pow)
lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
							save_weights_only=True,
						    filepath=modelfile_name,
						    save_freq="epoch",
						    monitor='val_accuracy',
						    mode='max',
						    save_best_only=True)

# map integer labels to 0,1,2 instead of 0,3,8
y_train = integer_remap(y_train, [[3,1],[8,2]])
y_test = integer_remap(y_test, [[3,1],[8,2]])

# output model summary for visual checks
model.summary()
label_train = keras.utils.to_categorical(y_train, 3)
label_test = keras.utils.to_categorical(y_test, 3)

# train model if flag is set
if train:
	model.fit(x=data_train,
              y=label_train,
              epochs=epochs,
              steps_per_epoch=len(data_train)//batch_size,
              verbose=1,
              validation_data=(data_test, label_test),
              validation_steps=1,
              callbacks=[lr_callback])#model_checkpoint_callback,

	cnn_results = model.evaluate(data_test, label_test)

print(y_test[:10])
