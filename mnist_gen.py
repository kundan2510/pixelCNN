from keras.datasets import mnist  #for loading the dataset
import numpy
from generic_utils import *  #has some utility functions
from models import Model  # class for collecting parameters fom layers and other stuffs
from layers import WrapperLayer, pixelConv, Softmax # module that implements various layers
import theano
import theano.tensor as T
import lasagne
import random
from plot_images import plot_25_figure, plot_100_figure # visualiztion of images
from sys import argv

DIM = 32 # feature map length for convolution layers
GRAD_CLIP = 1.
Q_LEVELS = 4 #level of quantization for channels, it has to be between 2 and 256
TRAIN_BATCH_SIZE = 100 
VALIDATE_BATCH_SIZE = 200
PRINT_EVERY = 100 #number of iterations after which stats are printed
EPOCH = 1000

PRETRAINED = True # whether to use a pre-trained weigths. if True, then argv[1] is assumed to be pickled weights of pre-trained model
GENERATE_ONLY = True # whether to only generate samples. Useful when you want to generate from a pre-trained model

OUT_DIR = '/Tmp/kumarkun/mnist_new_samples' # output folder for storing weights, generated and reconstructed samples
create_folder_if_not_there(OUT_DIR)


# Creating model.

model = Model(name = "MNIST.pixelCNN")


X = T.tensor3('X') # shape: (batchsize, height, width)
X_r = T.itensor3('X_r') #shape: (batchsize, height, width)

input_layer = WrapperLayer(X.dimshuffle(0,1,2,'x')) # input reshaped to (batchsize, height, width,1)

pixel_CNN = pixelConv(
	input_layer, 
	1, 
	DIM,
	name = model.name + ".pxCNN",
	num_layers = 12,
	Q_LEVELS = Q_LEVELS
	)

model.add_layer(pixel_CNN)

output_probab = Softmax(pixel_CNN).output()

# in nats
cost = T.nnet.categorical_crossentropy(
	output_probab.reshape((-1,output_probab.shape[output_probab.ndim - 1])),
	X_r.flatten()
	).mean()

output_image = sample_from_softmax(output_probab)



model.print_params() 

params = model.get_params()

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]



updates = lasagne.updates.adam(grads, params, learning_rate = 1e-3)

train_fn = theano.function([X, X_r], cost, updates = updates)

valid_fn = theano.function([X, X_r], cost)

generate_routine = theano.function([X], output_image)

def generate_fn(generate_routine, HEIGHT, WIDTH, num):
	X = floatX(numpy.zeros((num, HEIGHT, WIDTH)))
	for i in range(HEIGHT):
		for j in range(WIDTH):
			samples = generate_routine(X)
			X[:,i,j] = downscale_images(samples[:,i,j,0], Q_LEVELS-1)

	return X


if PRETRAINED:
	model.load_params(argv[1])


(X_train_r, _), (X_test_r, _) = mnist.load_data()

'''
First, downscale images from 0-255 to [0,1), then upscale to 0-(Q_LEVELS-1).
This quantized image becomes the target.
Targets are again downscaled to [0,1] to get the inputs.
'''

X_train_r = upscale_images(downscale_images(X_train_r, 256), Q_LEVELS) 
X_test_r = upscale_images(downscale_images(X_test_r, 256), Q_LEVELS)

X_train = downscale_images(X_train_r, Q_LEVELS - 1)
X_test = downscale_images(X_test_r, Q_LEVELS - 1)


if GENERATE_ONLY:
	X = generate_fn(generate_routine, 28, 28, 100)
	plot_100_figure(X, '{}/generated_only_images.jpg'.format(OUT_DIR))
	j = int(random.random() * (len(X_train) - 101))
	plot_100_figure(X_train[j : j + 100], '{}/training_images.jpg'.format(OUT_DIR))
	exit()



errors = {'training' : [], 'validation' : []}

num_iters = 0

def validate():
	costs = []
	BATCH_SIZE = VALIDATE_BATCH_SIZE
	num_batch_valid = len(X_test)//BATCH_SIZE

	for j in range(num_batch_valid):
		cost = valid_fn(X_test[j*BATCH_SIZE: (j+1)*BATCH_SIZE], X_test_r[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		costs.append(cost)

	return numpy.mean(costs)


print "Training"
for i in range(EPOCH):
	"""Training"""
	costs = []
	BATCH_SIZE = TRAIN_BATCH_SIZE
	num_batch_train = len(X_train)//BATCH_SIZE
	for j in range(num_batch_train):

		cost = train_fn(X_train[j*BATCH_SIZE: (j+1)*BATCH_SIZE], X_train_r[j*BATCH_SIZE: (j+1)*BATCH_SIZE])

		costs.append(cost)

		num_iters += 1

		if (j+1) % PRINT_EVERY == 0:
			print ("Training: epoch {}, iter {}, cost {}".format(i,j+1,numpy.mean(costs)))

	print("Training cost for epoch {}: {}".format(i+1, numpy.mean(costs)))
	errors['training'].append(numpy.mean(costs))

	val_error = validate()	
	errors['validation'].append(val_error)

	model.save_params('{}/epoch_{}_val_error_{}.pkl'.format(OUT_DIR,i, val_error)) #parameters are saved after every epoch

	X = generate_fn(generate_routine, 28, 28, 25) # 25 images are generated after every epoch

	reconstruction = generate_routine(X_test[:25])[:,:,:,0]

	plot_25_figure(X, '{}/epoch_{}_val_error_{}_gen_images.jpg'.format(OUT_DIR, i, val_error))
	plot_25_figure(reconstruction, '{}/epoch_{}_reconstructed.jpg'.format(OUT_DIR, i))

	print("Validation cost after epoch {}: {}".format(i+1, val_error))

	if i % 2 == 0:
		save(errors, '{}/epoch_{}_NLL.pkl'.format(OUT_DIR, i)) #NLL upto ith epoch stored after every 2 epochs. Too much redundancy here. 





