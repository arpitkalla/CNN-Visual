import argparse
import numpy as np
import tensorflow as tf

from model import ANN
import data


parser = argparse.ArgumentParser(description='Visualize ANN')
parser.add_argument('-d', '--dataset', type=str, default='mnist',
                    choices=data.get_names())
parser.add_argument('--num_iter', type=int, default=5000)
args = parser.parse_args()


dataset = data.init_dataset(name=args.dataset)

model = ANN(dataset.shape)

model.train(dataset.tr_data, dataset.tr_labels, num_iter=args.num_iter)