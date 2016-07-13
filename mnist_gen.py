from keras.datasets import mnist
import numpy
from generic_utils import *
from models import Model
from layers import WrapperLayer, pixelConv, Softmax
import theano
import theano.tensor as T
import lasagne

from plot_images import plot_20_figure

DIM = 32
GRAD_CLIP = 1.
# Q_LEVELS = 2
BATCH_SIZE = 100
PRINT_EVERY = 100
EPOCH = 1000

OUT_DIR = '/Tmp/kumarkun/mnist'
create_folder_if_not_there(OUT_DIR)

model = Model(name = "MNIST.pixelCNN")

X = T.tensor3('X') # shape: (batchsize, height, width)

input_layer = WrapperLayer(X.dimshuffle(0,1,2,'x')) # input reshaped to (batchsize, height, width,1)

pixel_CNN = pixelConv(
	input_layer, 
	1, 
	DIM, 
	(3, 3), 
	name = model.name + ".pxCNN1"
	num_layers = 13
	)

model.add_layer(pixel_CNN)

output_probab = T.nnet.sigmoid(pixel_CNN.output())

cost = T.nnet.binary_crossentropy(output_probab.flatten(), X.flatten()).mean()


# reporting NLL in bits
cost = cost * floatX(1.44269504089)

model.print_params()

params = model.get_params()

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function([X], cost, updates=updates)

valid_fn = theano.function([X], cost)

generate_routine = theano.function([X], output_probab)

def generate_fn(generate_routine, HEIGHT, WIDTH, num):
	X = floatX(numpy.zeros((num, HEIGHT, WIDTH)))

	for i in range(HEIGHT):
		for j in range(WIDTH):
			samples = generate_routine(X)
			X[:,i,j] = floatX(stochastic_binarize(samples)[:,i,j,0])

	return X

(X_train, _), (X_test, _) = mnist.load_data()

X_train = downscale_images(X_train, 256)
X_test = downscale_images(X_test, 256)


print "Training"
for i in range(EPOCH):
	"""Training"""
	costs = []
	num_batch_train = len(X_train)//BATCH_SIZE
	for j in range(num_batch_train):
		X_curr = stochastic_binarize(X_train[j*BATCH_SIZE: (j+1)*BATCH_SIZE])

		cost = train_fn(X_curr)

		costs.append(cost)
		if j % PRINT_EVERY == 0:
			print ("Training: epoch {}, iter {}, cost {}".format(i,j,numpy.mean(costs)))

	print("Training cost for epoch {}: {}".format(i+1, numpy.mean(costs)))

	costs = []
	num_batch_valid = len(X_test)//BATCH_SIZE
	for j in range(num_batch_valid):
		X_curr = stochastic_binarize(X_test[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		cost = valid_fn(X_curr)
		costs.append(cost)
		if j % PRINT_EVERY == 0:
			print ("Validation: epoch {}, iter {}, cost {}".format(i,j,numpy.mean(costs)))

	model.save_params('{}/epoch_{}_val_error_{}.pkl'.format(OUT_DIR,i, numpy.mean(costs)))

	X = generate_fn(generate_routine, 28, 28, 20)
	save(X, '{}/epoch_{}_val_error_{}_gen_images.pkl'.format(OUT_DIR, i, numpy.mean(costs)))

	print("Validation cost after epoch {}: {}".format(i+1, numpy.mean(costs)))











