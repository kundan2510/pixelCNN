import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv
from generic_utils import *

srng = RandomStreams(seed=3732)
T.nnet.relu = lambda x: T.switch(x > floatX(0.), x, floatX(0.00001)*x)


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

def get_conv_2d_filter(filter_shape, param_list = None, masktype = None, name = ""):
	fan_in = numpy.prod(filter_shape[1:])
	fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
	w_std = numpy.sqrt(2.0 / (fan_in + fan_out))

	filter_init = uniform(w_std, filter_shape)

	# assert(filter_shape[2] % 2 == 1), "Only filters with odd dimesnions are allowed."
	# assert(filter_shape[3] % 2 == 1), "Only filters with odd dimesnions are allowed."

	if masktype is not None:
		filter_init *= floatX(numpy.sqrt(2.))

	conv_filter = theano.shared(filter_init, name = name)
	param_list.append(conv_filter)

	if masktype is not None:
		mask = numpy.ones(
			filter_shape,
			dtype=theano.config.floatX
			)

		for i in range(filter_shape[2]):
			for j in range(filter_shape[3]):
				if i > filter_shape[2]//2:
					mask[:,:,i,j] = floatX(0.0)

				if i == filter_shape[2]//2 and j > filter_shape[3]//2:
					mask[:,:,i,j] = floatX(0.0)

		if masktype == 'a':
			mask[:,:,filter_shape[2]//2,filter_shape[3]//2] = floatX(0.0)

		conv_filter = conv_filter*mask

	return conv_filter

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
	def __init__(self, input_layer, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='half', masktype = None, activation = None, name = ""):
		self.X = input_layer.output()
		self.name = name
		self.subsample = subsample
		self.border_mode = border_mode

		self.params = []

		if isinstance(filter_size, tuple):
			self.filter_shape = (output_channels, input_channels, filter_size[0], filter_size[1])
		else:
			self.filter_shape = (output_channels, input_channels, filter_size, filter_size)

		self.filter = get_conv_2d_filter(self.filter_shape, param_list = self.params, masktype = masktype, name=name+'.filter')

		self.bias = bias_weights((output_channels,), param_list = self.params, name = name+'.b')

		self.activation = activation


		conv_out = T.nnet.conv2d(self.X, self.filter, border_mode = self.border_mode, filter_flip=False)
		self.Y = conv_out + self.bias[None,:,None,None]
		if self.activation is not None:
			if self.activation == 'relu':
				self.Y = T.nnet.relu(self.Y)
			elif self.activation == 'tanh':
				self.Y = T.tanh(self.Y)
			else:
				raise Exception("Not Implemented Error: {} activation not allowed".format(activation))


		# conv_out = dnn_conv( self.X, self.filter, border_mode = self.border_mode, conv_mode='cross', subsample=self.subsample)


	def output(self):
		# conv_out = dnn_conv( self.X, self.filter, border_mode = self.border_mode, conv_mode='cross', subsample=self.subsample)
		return self.Y


class pixelConv(Layer):
	"""
	input_shape: (batch_size, height, width, 1)
	output_shape: (batch_size, height, width, 1)
	"""
	def __init__(self, input_layer, input_dim, DIM, Q_LEVELS = None, num_layers = 6, activation='relu', name=""):

		if activation is None:
			apply_act = lambda r: r
		elif activation == 'relu':
			apply_act = T.nnet.relu
		elif activation == 'tanh':
			apply_act = T.tanh
		else:
			raise Exception("{} activation not implemented!!".format(activation))


		self.X = input_layer.output().dimshuffle(0,3,1,2)

		filter_size = 7 # for first layer

		vertical_stack = Conv2D(
			WrapperLayer(self.X), 
			input_dim,
			DIM, 
			((filter_size // 2) + 1, filter_size), 
			masktype=None, 
			border_mode=(filter_size // 2 + 1, filter_size // 2), 
			name= name + ".vstack1",
			activation = None
			)

		out_v = vertical_stack.output()

		vertical_and_input_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], self.X], axis=1)

		horizontal_stack = Conv2D(
			WrapperLayer(vertical_and_input_stack), 
			input_dim+DIM, DIM, 
			(1,filter_size), 
			border_mode = (0,filter_size//2), 
			masktype='a', 
			name = name + ".hstack1",
			activation = None
			)

		self.params = vertical_stack.params + horizontal_stack.params

		X_h = horizontal_stack.output()
		X_v = out_v[:,:,1:-(filter_size//2) - 1,:]

		filter_size = 3 #all layers beyond first

		for i in range(num_layers - 2):
			# TODO: operations on integrating horizontal and vertical stacks
			vertical_stack = Conv2D(
				WrapperLayer(X_v), 
				DIM, 
				DIM, 
				((filter_size // 2) + 1, filter_size), 
				masktype = None, 
				border_mode = (filter_size // 2 + 1, filter_size // 2), 
				name= name + ".vstack{}".format(i+1),
				activation = None
				)
			out_v = vertical_stack.output()
			vertical_and_prev_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], X_h], axis=1)

			horizontal_stack = Conv2D(
				WrapperLayer(vertical_and_prev_stack),
				DIM*2, 
				DIM,
				(1, (filter_size // 2) + 1), 
				border_mode = (0, filter_size // 2), 
				masktype = None, 
				name = name + ".hstack{}".format(i+1),
				activation = activation
				)

			self.params += (vertical_stack.params + horizontal_stack.params)

			X_v = apply_act(out_v[:,:,1:-(filter_size//2) - 1,:])
			X_h = horizontal_stack.output()[:,:,:,:-(filter_size//2)] + X_h #residual connection added

		combined_stack1 = Conv2D(
				WrapperLayer(X_h), 
				DIM, 
				DIM, 
				(1, 1), 
				masktype = None, 
				border_mode = 'valid', 
				name=name+".combined_stack1",
				activation = activation
				)

		if Q_LEVELS is None:
			out_dim = input_dim
		else:
			out_dim = Q_LEVELS

		combined_stack2 = Conv2D(
				combined_stack1, 
				DIM, 
				out_dim, 
				(1, 1), 
				masktype = None, 
				border_mode = 'valid', 
				name=name+".combined_stack2",
				activation = None
				)

		self.params += (combined_stack1.params + combined_stack2.params)
		self.Y = combined_stack2.output().dimshuffle(0,2,3,1)

	def output(self):
		return self.Y

class WrapperLayer(Layer):
	def __init__(self, X, name=""):
		self.params = []
		self.name = name
		self.X = X

	def output(self):
		return self.X


