from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
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
import gc

all_types = [
    'SPhiv',
    'RPhiv',
    'Phiv',
    'LTNPhiv',
]

prefix = 'cchfl'
seq_len = 8084
cutoff = 75
input_dir = '../hivdata'
input_dir = os.path.abspath(input_dir)
model_output_path = '../models'

f_pat = r'{}_[0-9]+-[0-9]+_{}\.dot'.format(prefix, cutoff)
dotpattern = r'P([0-9]+) -> P([0-9]+)'
pklpattern = r'{}_{}_(P[0-9]+)\.pkl'.format(cutoff, prefix)

all_types = [
    'SPhiv',
    'RPhiv',
    'Phiv',
    'LTNPhiv',
]

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


for i in range(2, len(all_types) + 1):

    types = all_types[:i]
    print("Classification: {}".format('/'.join(types)))
    
    model_output_path = '../rf_models/{}'.format('_'.join(types))
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

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

    train_items = [x for sub in train_items for x in sub]
    train_labels = [x for sub in train_labels for x in sub]
    train_labels = encoder.fit_transform(train_labels)

    test_items = []
    test_labels = []
    for phenotype in types:
        fpath = os.path.join(input_dir, '{}_test.dat'.format(phenotype))
        test_items.append(load_data_file(fpath))
        test_labels.append([phenotype] * len(test_items[-1]))

    length = min([len(x) for x in test_items])
    for i in range(len(test_items)):
        test_items[i] = test_items[i][:length]
        test_labels[i] = test_labels[i][:length]

    test_items = [x for sub in test_items for x in sub]
    test_labels = [x for sub in test_labels for x in sub]
    test_labels = encoder.transform(test_labels)

    feature_dict = defaultdict(list)

    edges = set()
    nodes = set()

    results = []
    n_estimators = [10, 25, 50, 100, 250, 500]
    for estimators in n_estimators:
        clf = RandomForestClassifier(n_estimators=estimators, n_jobs=32)
        clf.fit(train_items, train_labels)
        preds = clf.predict(test_items)
        accuracy = accuracy_score(test_labels, preds)
        results.append((clf, accuracy, estimators))
    class_labels = encoder.classes_

    cmap = plt.cm.Blues
    results = list(sorted(results, key=lambda x: x[1], reverse=True))
    for idx in range(5):
        clf, accuracy, num_estimators = results[idx]
        figname = os.path.join(model_output_path, '{}.png'.format(idx))
        print("Random Forest {:.3f}%: {} estimators".format(accuracy * 100, num_estimators))
        out_fname = os.path.join(model_output_path, 'rf_model_{}.pkl'.format(idx))
        y_pred = clf.predict(test_items)

        title = 'Normalized confusion matrix\nRandom Forest, {} estimators, acc {:.2f}%'.format(
            num_estimators,
            accuracy * 100,
        )
        y_true = test_labels
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
            pickle.dump(results[idx], fh)
