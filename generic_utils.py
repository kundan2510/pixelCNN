import theano
import theano.tensor as T
import os
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pickle
import json
import numpy

srng = RandomStreams(seed=4884)
def create_folder_if_not_there(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
		print "Created folder {}".format(folder)

def floatX(num):
	if theano.config.floatX == 'float32':
		return numpy.float32(num)
	else:
		raise Exception("{} type not supported".format(theano.config.floatX))


def downscale_images(X, LEVEL):
	X = floatX(X)/floatX(LEVEL)
	return X

def upscale_images(X, LEVEL):
	X = numpy.uint8(X*LEVEL)
	return X

def stochastic_binarize(X):
	return (numpy.random.uniform(size=X.shape) < X).astype('float32')

def sample_from_softmax(softmax_var):
	#softmax_var assumed to be of shape (batch_size, num_classes)
	old_shape = softmax_var.shape

	softmax_var_reshaped = softmax_var.reshape((-1,softmax_var.shape[softmax_var.ndim-1]))

	return T.argmax(
		T.cast(
	        srng.multinomial(pvals=softmax_var_reshaped),
	        theano.config.floatX
	        ).reshape(old_shape),
		axis = softmax_var.ndim-1
		)


#<Ishaan's code>
def Skew(inputs, WIDTH, HEIGHT):
    """
    input.shape: (batch size, HEIGHT, WIDTH, num_channels)
    """
    buf = T.zeros(
        (inputs.shape[0], inputs.shape[1], 2*inputs.shape[2] - 1, inputs.shape[3]),
        theano.config.floatX
    )

    for i in xrange(HEIGHT):
        buf = T.inc_subtensor(buf[:, i, i:i+WIDTH, :], inputs[:,i,:,:])
    
    return buf

def Unskew(padded, WIDTH, HEIGHT):
    """
    input.shape: (batch size, HEIGHT, 2*WIDTH - 1, num_channels)
    """
    return T.stack([padded[:, i, i:i+WIDTH, :] for i in xrange(HEIGHT)], axis=1)

def new_learning_time_decay(learning_rate, iter_num, k):
	return floatX(learning_rate/(1.0+ iter_num*k))

#</Ishaan's code>

def load(file_name):
	open_file = open(file_name, 'rb')
	if ".json" in file_name:
		obj = json.load(open_file)
	elif ".pkl" in file_name:
		obj = pickle.load(open_file)
	open_file.close()
	return obj

def save(obj, file_name):
	open_file = open(file_name, 'wb')
	if ".json" in file_name:
		json.dump(obj,open_file)
	elif ".pkl" in file_name:
		pickle.dump(obj, open_file)
	open_file.close()

