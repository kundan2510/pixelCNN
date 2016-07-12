import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv
from generic_utils import *

def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)

def bias_weights(length, initialization='zeros', param_list = None, name = ""):
	"theano shared variable for bias unit, given length and initialization"
	if initialization == 'zeros':
		bias_initialization = numpy.zeros(length).astype(theano.config.floatX)
	else:
		raise Exception("Not Implemented Error: {} initialization not implemented".format(initialization))

	bias =  theano.shared(
			bias_initialization,
			name=name
			)

	if param_list is not None:
		param_list.append(bias)

	return bias

def get_conv_2d_filter(filter_shape, subsample = (1,1), param_list = None, masktype = None, name = "", initialization='glorot'):
	if initialization == 'glorot':

		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(subsample))
		w_std = numpy.sqrt(2 / (fan_in + fan_out))

		filter_init = uniform(w_std, filter_shape)

		assert(filter_shape[2] % 2 == 1), "Only filters with odd dimesnions are allowed."
		assert(filter_shape[3] % 2 == 1), "Only filters with odd dimesnions are allowed."

		if masktype is not None:
			filter_init *= floatX(0.5)

		conv_filter = theano.shared(filter_init, name = name)
		param_list.append(conv_filter)

		if masktype is not None:
			mask = numpy.zeros(
				filter_shape,
				dtype=theano.config.floatX
				)

			if filter_shape[3] == 1 and filter_shape[2] == 1:
				raise Exception("Masking not allowed for (1,1) filter shape")
			elif filter_shape[3] == 1:
				mask[:,:,:filter_shape[2]//2,:] = floatX(1.)
				if masktype == 'f':
					mask[:,:,filter_shape[2]//2,:] = floatX(1.)
			elif filter_shape[2] == 1:
				mask[:,:,:,:filter_shape[3]//2] = floatX(1.)
				if masktype == 'f':
					mask[:,:,:,:filter_shape[3]//2] = floatX(1.)
			else:
				center_row = filter_shape[2]//2
				centre_col = filter_shape[3]//2
				if masktype == 'f':
					mask[:,:,:center_row,:] = floatX(1.)
					mask[:,:,center_row,:centre_col+1] = floatX(1.)
				elif masktype == 'b':
					mask[:,:,:center_row,:] = floatX(1.)
					mask[:,:,center_row,:centre_col] = floatX(1.)
				elif masktype == 'p':
					mask[:,:,:center_row,:] = floatX(1.)

			conv_filter = conv_filter*mask

		return conv_filter
	else:
		raise Exception('Not Implemented Error')

class Layer:
	'''Generic Layer Template which all layers should inherit'''
	def __init__(name = ""):
		self.name = name
		self.params = []

	def get_params():
		return self.params


class Conv2D(Layer):
	"""
	input_shape: (batch_size, input_channels, height, width)
	filter_size: int or (row, column)
	"""
	def __init__(self, input_layer, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='half', masktype = None, name = ""):
		self.X = input_layer.output()
		self.name = name
		self.subsample = subsample
		self.border_mode = border_mode

		self.params = []

		if isinstance(filter_size, tuple):
			self.filter_shape = (output_channels, input_channels, filter_size[0], filter_size[1])
		else:
			self.filter_shape = (output_channels, input_channels, filter_size, filter_size)

		self.filter = get_conv_2d_filter(self.filter_shape, param_list = self.params, initialization = 'glorot', masktype = masktype, name=name+'.filter')

		self.bias = bias_weights((output_channels,), param_list = self.params, name = name+'.b')

	def output(self):
		# conv_out = dnn_conv( self.X, self.filter, border_mode = self.border_mode, conv_mode='cross', subsample=self.subsample)
		conv_out = T.nnet.conv2d(self.X, self.filter, border_mode = self.border_mode, filter_flip=False)
		return conv_out + self.bias[None,:,None,None]


class pixelConv(Layer):
	"""
	input_shape: (batch_size, height, width, input_dim)
	output_shape: (batch_size, height, width, output_dim)
	"""
	def __init__(self, input_layer, input_dim, output_dim, filter_size, masktype='p', activation='relu', name=""):
		self.X = input_layer.output().dimshuffle(0,3,1,2)
		vertical_stack = Conv2D(WrapperLayer(self.X), input_dim, output_dim, filter_size, masktype=masktype, border_mode='half', name=name+".conv2d")

		self.params = vertical_stack.params
		self.Y = vertical_stack.output().dimshuffle(0,2,3,1)

		if activation is not None:
			if activation == 'relu':
				self.Y = T.nnet.relu(self.Y)
			elif activation == 'tanh':
				self.Y = T.tanh(self.Y)
			else:
				raise Exception("Not Implemented Error: {} activation not allowed".format(activation))

	def output(self):
		return self.Y

class WrapperLayer(Layer):
	def __init__(self, X, name=""):
		self.params = []
		self.name = name
		self.X = X

	def output(self):
		return self.X

class Softmax(Layer):
	def __init__(self, input_layer,  name=""):
		self.input_layer = input_layer
		self.name = name
		self.params = []
		self.X = self.input_layer.output()
		self.input_shape = self.X.shape

	def output(self):
		return T.nnet.softmax(self.X.reshape((-1,self.input_shape[self.X.ndim-1]))).reshape(self.input_shape)

