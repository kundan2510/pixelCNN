import pickle
import numpy

class Model:
	def __init__(self, name=""):
		self.name = name
		self.layers = []
		self.params = []
	
	def add_layer(self,layer):
		self.layers.append(layer)
		for p in layer.params:
			self.params.append(p)

	def print_layers(self):
		for layer in self.layers:
			print layer.name

	def get_params(self):
		return self.params

	def print_params(self):
		total_params = 0
		for p in self.params:
			curr_params = numpy.prod(numpy.shape(p.get_value()))
			total_params += curr_params
			print "{} ({})".format(p.name, curr_params)
		print ("total number of parameters: {}".format(total_params))
		print ("Note: Effective number of parameters might be less due to masking!!")

	def save_params(self, file_name):
		params = {}
		for p in self.params:
			params[p.name] = p.get_value()
		pickle.dump(params, open(file_name, 'wb'))

	def load_params(self, file_name):
		params = pickle.load(open(file_name, 'rb'))
		for p in self.params:
			p.set_value(params[p.name])
