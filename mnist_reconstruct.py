from keras.datasets import mnist
import numpy
from generic_utils import *
from models import Model
from layers import WrapperLayer, pixelConv
import theano
import theano.tensor as T
import lasagne
import random

from plot_images import plot_20_figure

from sys import argv

DIM = 128
GRAD_CLIP = 1.
# Q_LEVELS = 2
BATCH_SIZE = 16
PRINT_EVERY = 100
EPOCH = 1000

OUT_DIR = '/Tmp/kumarkun/mnist_new'
create_folder_if_not_there(OUT_DIR)

model = Model(name = "MNIST.pixelCNN")

X = T.tensor3('X') # shape: (batchsize, height, width)

input_layer = WrapperLayer(X.dimshuffle(0,1,2,'x')) # input reshaped to (batchsize, height, width,1)

pixel_CNN = pixelConv(
	input_layer, 
	1, 
	DIM, 
	3, 
	name = model.name + ".pxCNN",
	num_layers = 13
	)

model.add_layer(pixel_CNN)

output_probab = T.nnet.sigmoid(pixel_CNN.output())

model.print_params()


generate_routine = theano.function([X], output_probab)

# print "Loading weigths from : {}".format(argv[1])
# model.load_params(argv[1])

# cost = T.nnet.binary_crossentropy(output_probab.flatten(), X.flatten()).mean()


# # reporting NLL in bits
# cost = cost * floatX(1.44269504089)


(X_train, _), (X_test, _) = mnist.load_data()

X_train = downscale_images(X_train, 256)
X_test = downscale_images(X_test, 256)

inp = stochastic_binarize(X_train[0:20])

output = stochastic_binarize(generate_routine(inp)[:,:,:,0])
save(inp,'input.pkl')
save(output, 'input_reconstructed.pkl')

plot_20_figure(inp, 'input.jpg')

plot_20_figure(output, 'output_reconstructed.jpg')



