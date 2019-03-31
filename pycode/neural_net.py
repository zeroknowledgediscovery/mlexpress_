from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import os
import sys
import re
import networkx as nx
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prefix = 'cchfl'
seq_len = 8084
cutoff = 75
input_dir = '../hivdata'
input_dir = os.path.abspath(input_dir)
model_output_path = '../models'

types = [
    'SPhiv',
    'RPhiv',
    'Phiv',
    'LTNPhiv',
]

model_output_path = '../models/{}'.format('_'.join(types))
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)

f_pat = r'{}_[0-9]+-[0-9]+_{}\.dot'.format(prefix, cutoff)
dotpattern = r'P([0-9]+) -> P([0-9]+)'
pklpattern = r'{}_{}_(P[0-9]+)\.pkl'.format(cutoff, prefix)

nucleotides = 'ACTG'

onehot_dict = {}

for i, x in enumerate(nucleotides):
    vec = [0] * len(nucleotides)
    vec[i] = 1
    onehot_dict[x] = vec

def seq_to_vec(seq):
    vec = [onehot_dict[x] for x in seq]
    return [x for subvec in vec for x in subvec]

def load_data_file(path):
    with open(path, 'r') as fh:
        results = fh.read().strip().split('\n')[1:]
    results = [x.split(',') for x in results]
    results = map(lambda l: [x[:1] for x in l], results)
    results = [seq_to_vec(x) for x in results]
    return results

encoder = LabelEncoder()

train_items = []
train_labels = []
for type in types:
    fpath = os.path.join(input_dir, '{}_train.dat'.format(type))
    train_items.append(load_data_file(fpath))
    train_labels.append([type] * len(train_items[-1]))

length = min([len(x) for x in train_items])
for i in range(len(train_items)):
    train_items[i] = train_items[i][:length]
    train_labels[i] = train_labels[i][:length]

train_items = np.array([x for sub in train_items for x in sub])
train_labels = [x for sub in train_labels for x in sub]
train_labels = encoder.fit_transform(train_labels)
train_labels = np.array(np_utils.to_categorical(train_labels))

test_items = []
test_labels = []
for type in types:
    fpath = os.path.join(input_dir, '{}_test.dat'.format(type))
    test_items.append(load_data_file(fpath))
    test_labels.append([type] * len(test_items[-1]))

length = min([len(x) for x in test_items])
for i in range(len(test_items)):
    test_items[i] = test_items[i][:length]
    test_labels[i] = test_labels[i][:length]

test_items = np.array([x for sub in test_items for x in sub])
test_labels = [x for sub in test_labels for x in sub]
test_labels = encoder.transform(test_labels)
test_labels = np.array(np_utils.to_categorical(test_labels))

feature_dict = defaultdict(list)

edges = set()
nodes = set()

eseq_len = seq_len * len(nucleotides)

orig_wt_matrix = np.zeros((eseq_len, eseq_len))

# for fname in files:
#     match = re.match(pklpattern, fname)
#     if match:
#         with open(os.path.join(root, fname), 'rb') as fh:
#             TR = pickle.load(fh)
#             # print(TR)
#             # print(TR.nodes)
#             # print("Feature")
#             # print(TR.feature)
#             # print(TR.feature[1])
#             # print(TR.TREE_LEAF)
#             # print(TR.children_right)
#             # print(TR.children_left)
#             # print(TR.edge_cond_)
#             # print(type(TR.edge_cond_))
#             # print(TR.ACC_)
#             # print(TR.ACCx_)
#             result = [int(TR.feature[x][1:]) for x in TR.feature if not TR.TREE_LEAF[x]]
#             for x in result:
#                 orig_wt_matrix[int(match.group(1)[1:])][x] = 1

activations = ["relu", "sigmoid"]
layer_sizes = [256, 512, 768, 1024]
num_hiddens = [1,2,3,4,5]
num_epochs = [2, 5, 7, 10]
# layer_sizes = [256]
# num_hiddens = [2]
# num_epochs = [5]

results = []

class_labels = encoder.classes_

for epochs in num_epochs:
    for activation_func in activations:
        for layer_size in layer_sizes:
            for num_hidden in num_hiddens:
                model = Sequential()
                model.add(Dense(layer_size, input_dim=eseq_len, activation=activation_func))
                for i in range(num_hidden):
                    model.add(Dense(layer_size, activation=activation_func))
                model.add(Dense(len(class_labels)))
                model.add(Activation('softmax'))

                sgd = SGD(lr=0.01)
                model.compile(loss="binary_crossentropy", optimizer=sgd,
                	metrics=["accuracy"])

                model.fit(np.array(train_items), np.array(train_labels), verbose=0, shuffle=True, epochs=epochs)


                loss, accuracy = model.evaluate(test_items, test_labels, verbose=0)

                results.append(
                    (
                        model,
                        accuracy,
                        activation_func,
                        layer_size,
                        num_hidden,
                        epochs,
                    )
                )


results = sorted(results, key = lambda x: x[1], reverse = True)

cmap = plt.cm.Blues

for i in range(5):
    model, accuracy, activation_func, layer_size, num_hidden, epochs = results[i]
    figname = os.path.join(model_output_path, '{}.png'.format(i))
    print("{}: {}/{} units/{} layers/{} epochs".format(accuracy, activation_func, layer_size, num_hidden, epochs))
    out_fname = os.path.join(model_output_path, 'model_{}.pkl'.format(i))

    y_pred = model.predict(test_items)

    y_pred = np.argmax(y_pred, axis=1)

    title = 'Normalized confusion matrix\n{} hidden layers of size {}, {} epochs, {} activation\n{:.2f} % test accuracy'.format(
        num_hidden,
        layer_size,
        epochs,
        activation_func,
        accuracy * 100,
    )
    y_true = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = class_labels[unique_labels(y_true, y_pred)]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


    plt.savefig(figname)
    plt.close()

    with open(out_fname, 'wb') as fh:
        pickle.dump(results[i], fh)
