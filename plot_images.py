import pickle
import scipy.misc
import numpy as np
from sys import argv

def plot_25_figure(images, output_name):
	images = images.reshape((5,5,28,28))
	# rowx, rowy, height, width -> rowy, height, rowx, width
	images = images.transpose(1,2,0,3)
	images = images.reshape((5*28, 5*28))
	scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(output_name)

if __name__ == "__main__":
	X = pickle.load(open(argv[1],'rb'))
	output_name = argv[1].split('/')[-1].split('.')[0] + '.jpg'
	plot_25_figure(X, output_name)

