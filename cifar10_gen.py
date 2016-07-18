from keras.datasets import cifar10
import numpy
from generic_utils import *
from models import Model
from layers import WrapperLayer, pixelConv, Softmax
import theano
import theano.tensor as T
import lasagne
import random
from plot_images import plot_25_figure

DIM = 32
GRAD_CLIP = 1.
Q_LEVELS = 256
BATCH_SIZE = 20
PRINT_EVERY = 250
EPOCH = 100

OUT_DIR = '/Tmp/kumarkun/cifar10'
create_folder_if_not_there(OUT_DIR)

model = Model(name = "CIFAR10.pixelCNN")


is_train = T.scalar()
X = T.tensor4('X') # shape: (batchsize, channels, height, width)
X_r = T.itensor4('X_r')

X_transformed = X_r.dimshuffle(0,2,3,1)
input_layer = WrapperLayer(X.dimshuffle(0,2,3,1)) # input reshaped to (batchsize, height, width,3)

pixel_CNN = pixelConv(
	input_layer, 
	3, 
	DIM,
	Q_LEVELS = Q_LEVELS,
	name = model.name + ".pxCNN",
	num_layers = 12,
	)

model.add_layer(pixel_CNN)

output_probab = Softmax(pixel_CNN).output()

cost = T.nnet.categorical_crossentropy(
	output_probab.reshape((-1,output_probab.shape[output_probab.ndim - 1])),
	X_r.flatten()
	).mean()
# in nats
output_image = sample_from_softmax(output_probab)

model.print_params()

params = model.get_params()

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

# learning_rate = T.scalar('learning_rate')

updates = lasagne.updates.adam(grads, params, learning_rate = 1e-3)

train_fn = theano.function([X, X_r], cost, updates = updates)

valid_fn = theano.function([X, X_r], cost)

generate_routine = theano.function([X], output_image)

def generate_fn(generate_routine, HEIGHT, WIDTH, num):
	X = floatX(numpy.zeros((num, 3, HEIGHT, WIDTH)))
	out = numpy.zeros((num,HEIGHT, WIDTH, 3))

	for i in range(HEIGHT):
		for j in range(WIDTH):
			samples = generate_routine(X)
			out[:,i,j] = samples[:,i,j]
			X[:,:,i,j] = downscale_images(samples[:,i,j,:], Q_LEVELS - 1)

	return out

(X_train_r, _), (X_test_r, _) = cifar10.load_data()

X_train_r = upscale_images(downscale_images(X_train_r, 256), Q_LEVELS) 
X_test_r = upscale_images(downscale_images(X_test_r, 256), Q_LEVELS)

X_train = downscale_images(X_train_r, Q_LEVELS - 1)
X_test = downscale_images(X_test_r, Q_LEVELS - 1)

errors = {'training' : [], 'validation' : []}

num_iters = 0
# init_learning_rate = floatX(0.001)

print "Training"
for i in range(EPOCH):
	"""Training"""
	costs = []
	num_batch_train = len(X_train)//BATCH_SIZE
	for j in range(num_batch_train):

		cost = train_fn(
			X_train[j*BATCH_SIZE: (j+1)*BATCH_SIZE],
			X_train_r[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
		)

		costs.append(cost)

		num_iters += 1

		if (j+1) % PRINT_EVERY == 0:
			print ("Training: epoch {}, iter {}, cost {}".format(i,j+1,numpy.mean(costs)))

	print("Training cost for epoch {}: {}".format(i+1, numpy.mean(costs)))
	errors['training'].append(numpy.mean(costs))

	costs = []
	num_batch_valid = len(X_test)//BATCH_SIZE

	for j in range(num_batch_valid):
		cost = valid_fn(
			X_test[j*BATCH_SIZE: (j+1)*BATCH_SIZE],
			X_test_r[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
			)
		costs.append(cost)

		if (j+1) % PRINT_EVERY == 0:
			print ("Validation: epoch {}, iter {}, cost {}".format(i,j+1,numpy.mean(costs)))

	model.save_params('{}/epoch_{}_val_error_{}.pkl'.format(OUT_DIR,i, numpy.mean(costs)))

	X = generate_fn(generate_routine, 32, 32, 25)

	reconstruction = generate_routine(X_test[:25])

	plot_25_figure(X, '{}/epoch_{}_val_error_{}_gen_images.jpg'.format(OUT_DIR, i, numpy.mean(costs)), num_channels = 3)
	plot_25_figure(reconstruction, '{}/epoch_{}_reconstructed.jpg'.format(OUT_DIR, i), num_channels = 3)

	print("Validation cost after epoch {}: {}".format(i+1, numpy.mean(costs)))
	errors['validation'].append(numpy.mean(costs))

	if i % 2 == 0:
		save(errors, '{}/epoch_{}_NLL.pkl'.format(OUT_DIR, i))





