import numpy as np
import os
import mlx as ml
import pickle
import sys
import re
import networkx as nx
from pprint import pprint
from collections import deque
from copy import deepcopy
import random

seq_path = sys.argv[1]

perturbation_path = sys.argv[2]

with open(seq_path, 'r') as fh:
    sequence = fh.read().strip().split(',')

with open(perturbation_path, 'r') as fh:
    primary_perturbations, secondary_perturbations = pickle.load(fh)

def update_sequence(sequence, perturbation_list):
    sequence = list(sequence)
    perturbation_dicts = [x[0] for x in perturbation_list]
    probabilities = [x[1] for x in perturbation_list]
    selected_perturbation = np.random.choice(perturbation_dicts, 1, probabilities)[0]
    for index in selected_perturbation:
        sequence[index] = selected_perturbation[index]
    return sequence

with open('../perturbation_example/perturbed_sequences.csv', 'w') as fh:
    for i in range(50):
        new_sequence = update_sequence(sequence, primary_perturbations)
        for perturbation in secondary_perturbations:
            new_sequence = update_sequence(new_sequence, perturbation)
        fh.write(','.join(new_sequence))
        fh.write('\n')
