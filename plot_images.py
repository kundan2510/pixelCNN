import pickle
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sys import argv

def plot_20_figure(X, output_name):
	fig = plt.figure()
	for x in range(5):
		for y in range(4):
			ax = fig.add_subplot(5, 4, 5*y+x)
			ax.matshow(X[5*y+x], cmap = matplotlib.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
	plt.savefig(output_name)

if __name__ == "__main__":
	X = pickle.load(open(argv[1],'rb'))

	output_name = argv[1].split('/')[-1].split('.')[0] + '.jpg'
	plot_20_figure(X, output_name)

