import pickle
import scipy.misc
import numpy as np
from sys import argv

def plot_25_figure(images, output_name, num_channels = 1):
	HEIGHT, WIDTH = images.shape[1], images.shape[2]
	if num_channels == 1:
		images = images.reshape((5,5,HEIGHT,WIDTH))
		# rowx, rowy, height, width -> rowy, height, rowx, width
		images = images.transpose(1,2,0,3)
		images = images.reshape((5*28, 5*28))
		scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(output_name)
	elif num_channels == 3:
		images = images.reshape((5,5,HEIGHT,WIDTH,3))
		images = images.transpose(1,2,0,3,4)
		images = images.reshape((5*HEIGHT, 5*WIDTH, 3))
		scipy.misc.toimage(images).save(output_name)
	else:
		raise Exception("You should not be here!! Only 1 or 3 channels allowed for images!!")


def plot_100_figure(images, output_name, num_channels = 1):
	HEIGHT, WIDTH = images.shape[1], images.shape[2]
	if num_channels == 1:
		images = images.reshape((10,10,HEIGHT,WIDTH))
		# rowx, rowy, height, width -> rowy, height, rowx, width
		images = images.transpose(1,2,0,3)
		images = images.reshape((10*28, 10*28))
		scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(output_name)
	elif num_channels == 3:
		images = images.reshape((10,10,HEIGHT,WIDTH,3))
		images = images.transpose(1,2,0,3,4)
		images = images.reshape((10*HEIGHT, 10*WIDTH, 3))
		scipy.misc.toimage(images).save(output_name)
	else:
		raise Exception("You should not be here!! Only 1 or 3 channels allowed for images!!")


if __name__ == "__main__":
	X = pickle.load(open(argv[1],'rb'))
	output_name = argv[1].split('/')[-1].split('.')[0] + '.jpg'
	plot_25_figure(X, output_name)

