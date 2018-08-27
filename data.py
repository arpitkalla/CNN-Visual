import helper
import numpy as np
import tensorflow as tf 

class MNIST(object):
	def __init__(self):
		mnist = tf.keras.datasets.mnist
		(x_train, y_train),(x_test, y_test) = mnist.load_data()

		tr_data = x_train
		ts_data = x_test
		tr_labels = []
		ts_labels = []
		for ytr in y_train:
		    tr = np.zeros([10])
		    tr[ytr] = 1.0
		    tr_labels.append(tr)
		for yts in y_test:
		    ts = np.zeros([10])
		    ts[yts] = 1.0
		    ts_labels.append(ts)

		self.tr_data = np.array(tr_data,dtype=np.float32)
		self.ts_data = np.array(ts_data,dtype=np.float32)
		self.tr_labels = np.array(tr_labels,dtype=np.float32)
		self.ts_labels = np.array(ts_labels,dtype=np.float32)
		self.shape = self.tr_data.shape

class CIFAR10(object):
	def __init__(self):
		cifar = tf.keras.datasets.cifar10
		(x_train, y_train),(x_test, y_test) = cifar.load_data()

		tr_data = np.squeeze(helper.grayscale(x_train))
		ts_data = np.squeeze(helper.grayscale(x_test))
		tr_labels = []
		ts_labels = []
		for ytr in y_train:
		    tr = np.zeros([10])
		    tr[ytr] = 1.0
		    tr_labels.append(tr)
		for yts in y_test:
		    ts = np.zeros([10])
		    ts[yts] = 1.0
		    ts_labels.append(ts)

		self.tr_data = np.array(tr_data,dtype=np.float32)
		self.ts_data = np.array(ts_data,dtype=np.float32)
		self.tr_labels = np.array(tr_labels,dtype=np.float32)
		self.ts_labels = np.array(ts_labels,dtype=np.float32)
		self.shape = self.tr_data.shape


__factory = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

