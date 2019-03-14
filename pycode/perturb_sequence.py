#!/usr/bin/python

"""
Note that mlx is in Python 2, and as such, any file that imports it (including)
this one) will also be in Python 2.
"""

import numpy as np
import os
import mlx as ml
import cPickle as pickle
import sys
import re
import networkx as nx
from pprint import pprint
from collections import deque
from copy import deepcopy
import random
import multiprocessing as mp
import gc
import argparse

parser = argparse.ArgumentParser(description="""Arguments for generating a pickle
of probability dictionaries""")
parser.add_argument('tree_path', type=str, help="Path to folder with graph trees")
parser.add_argument('--cutoffs', dest='CUTOFFS', nargs = '+', type=int)
parser.add_argument('--prefixes', dest='PREFIXES', nargs='+', type=str)
parser.add_argument('--input_sequence', dest='INPUT_SEQUENCE', type=str,
help="Path to file containing a single sequence to be mutated")
parser.add_argument('--target_location', dest='TARGET_LOCATION', type=int,
help="Index of location we wish to mutate")
parser.add_argument('--target_amino_acid', dest='TARGET_AA', type=str,
help="Amino acid we wish to change at this point")
parser.add_argument('--end_subtype', dest='END_SUBTYPE', type=str,
help="Subtype to change sequence into")
parser.add_argument('--end_cutoff', dest='END_CUTOFF', type=int,
help="Subtype confidence interval to verify")
parser.add_argument('--primary_only', dest='PRIMARY_ONLY', action='store_true',
default = False, help="""Will only store perturbation results for selected component""")
parser.add_argument('--max_confidence', dest='USE_MAX_CONFIDENCE',
action='store_true', default=False, help="""Generated a limited subset of the
output space by choosing most likely trees""")
parser.add_argument('--output_file', dest='OUTPUT_FILE', action='store',
type=str, help="Pickle of resulting probability dictionaries being stored")


results = parser.parse_args()

TREE_PATH = results.tree_path
CUTOFFS = results.CUTOFFS
INPUT_SEQUENCE = results.INPUT_SEQUENCE
END_SUBTYPE = results.END_SUBTYPE
END_CUTOFF = results.END_CUTOFF
PRIMARY_ONLY = results.PRIMARY_ONLY
USE_MAX_CONFIDENCE = results.USE_MAX_CONFIDENCE
TARGET_LOCATION = results.TARGET_LOCATION
TARGET_AA = results.TARGET_AA
OUTPUT_FILE = results.OUTPUT_FILE

PREFIXES = results.PREFIXES

fpattern = r'(.*?)_.*?_([0-9]+)\.dot'
datfpattern = r'(.*?)_.*?_([0-9]+)\.dat'
datpattern = r'"\(\'(.*?)\', \'(.*?)\'\)",([0-9]+\.[0-9]+)'
dotpattern = r'P([0-9]+) -> P([0-9]+)'

# cutoff, prefix, index
tstr = '{}_{}_P{}.pkl'

# CSV of nucleotide sequences
with open(INPUT_SEQUENCE, 'r') as fh:
    sequence = fh.read().strip().split(',')

root, fname, files = next(os.walk(TREE_PATH))

def load_trees(prefix, cutoff):
    trees = {}
    index = 0
    for index in range(len(sequence)):
        fname = os.path.join(TREE_PATH, tstr.format(
            cutoff, prefix, index
        ))
        if os.path.exists(fname):
            with open(fname, 'r') as fh:
                trees[index] = pickle.load(fh)
    return trees

def find_relevant_component(graph, part):
    for component in component_list:
        if part in component:
            return component
    else:
        return None

MASTER_TREE_DICT = {}

GRAPH_DICT = {}

for prefix in PREFIXES:
    for cutoff in CUTOFFS:
        # print("{} at {}".format(prefix, cutoff))
        trees = load_trees(prefix, cutoff)
        # print("{} trees loaded".format(len(trees)))
        edges = set()
        nodes = set()
        for fname in files:
            match = re.match(fpattern, fname)
            if match:
                if match.group(1) == prefix and match.group(2) == str(cutoff):
                    with open(os.path.join(root, fname), 'r') as fh:
                        text = fh.read()
                        for match in re.finditer(dotpattern, text):
                            x, y = int(match.group(1)), int(match.group(2))
                            edges.add((x, y))
                            nodes.add(x)
                            nodes.add(y)
        if len(edges) == 0 or len(nodes) == 0:
            continue
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        MASTER_TREE_DICT[(prefix, cutoff)] = trees
        GRAPH_DICT[(prefix, cutoff)] = graph

# our goal here is to construct in essence a set of trees of outcomes

print("Loaded sample RPhiv sequence")
print("{} nucleotides long".format(len(sequence)))

end = (END_SUBTYPE, END_CUTOFF)
# print(end)
# print("SEEN VALUES:")
# for item in MASTER_TREE_DICT:
#     print(item)

TREE_DICT = MASTER_TREE_DICT[end]
GRAPH = GRAPH_DICT[end]

# when porting to python 3, add the 'as_view = True' kwarg to this call
ud = GRAPH.to_undirected()

GRAPH_COMPONENTS = []

# necessary because this is an iterator
for component in nx.connected_components(ud):
    GRAPH_COMPONENTS.append(component)


print("{} components".format(len(GRAPH_COMPONENTS)))

# pprint(GRAPH_COMPONENTS)
# print(GRAPH.successors(335))

# print(sequence[112])
# print(GRAPH.successors(112))
# print(GRAPH.predecessors(112))
# pprint(TREE_DICT[9].__dict__)

def create_parent_dictionary(tree):
    left_child_dict = tree.children_left
    right_child_dict = tree.children_right
    parent_dict = {}
    for x in left_child_dict:
        parent_dict[left_child_dict[x]] = x
    for x in right_child_dict:
        parent_dict[right_child_dict[x]] = x
    return parent_dict

def get_prob_from_dict(prob_dict, class_name):
    if class_name not in prob_dict:
        return 0
    return prob_dict[class_name]/sum(prob_dict.values())

def normalize_probability_dict(prob_dict):
    prob_dict = deepcopy(prob_dict)
    total = float(sum(prob_dict.values()))
    for x in prob_dict:
        prob_dict[x] /= total
    return prob_dict

def get_leaf_probabilities(tree, target_class):
    leaf_prob_dict = {}
    for node in tree.TREE_LEAF:
        if tree.TREE_LEAF[node] is True:
            num_samples = tree.n_node_samples[node]
            classes = tree.class_pred_[node]
            leaf_prob_dict[node] = tree.n_node_samples[node] \
                * tree.class_pred_[node][target_class]
    return normalize_probability_dict(leaf_prob_dict)

def get_successors_and_predecessors(location):
    """
    Returns a list of children and parents of a particular node in the current
    graph. Resolves conflicts between two nodes that have edges going both ways
    by setting the node that predicts the other with more accuracy to be the
    parent.
    """
    successors = list(GRAPH.successors(location))
    predecessors = list(GRAPH.predecessors(location))

    conflicts = [x for x in successors if x in predecessors]

    successors = [x for x in successors if x not in conflicts]
    predecessors = [x for x in predecessors if x not in conflicts]

    for x in conflicts:
        # if A pred B and B pred A, choose the one that's more accurate
        # if our current location predicts our neighbor better
        # then we treat ourselves as the parent
        if TREE_DICT[location].ACCx_ > TREE_DICT[x].ACCx_:
            successors.append(x)
        else:
            predecessors.append(x)
    return successors, predecessors

def location_to_node_index(tree, location):
    target = 'P{}'.format(location)
    for feature in tree.feature:
        if tree.feature[feature] == target:
            return feature
    print("Feature {} not found in tree {}".format(
        target,
        tree.response_var_
    ))
    return None

def leaf_to_root_path(tree, leaf_index):
    path = []
    parent_dict = create_parent_dictionary(tree)
    current_node = leaf_index
    while current_node in parent_dict:
        path.append(current_node)
        current_node = parent_dict[current_node]
    path.append(current_node)
    return list(reversed(path))

def get_best_node(tree, target_class):
    prob_dict = get_leaf_probabilities(tree, target_class)
    max_prob = 0
    best_leaf = None
    for x in prob_dict:
        if prob_dict[x] > max_prob:
            best_leaf = x
            max_prob = prob_dict[x]
    return best_leaf

def get_child(tree, index, decision):
    """
    Given a node and a decision, evaluate it at that node and return
    the index of the next node

    Returns None if we can't evaluate it
    """
    for edge in tree.edge_cond_:
        if edge[0] == index:
            if decision in tree.edge_cond_[edge]:
                return edge[1]
    return None

def is_leaf(tree, index):
    """
    Given a tree and the index of a node within the tree, return whether
    or not it is a leaf
    """
    return tree.TREE_LEAF[index]

def tree_str_to_index(tree_str):
    """
    The tree stores labels as 'P[label]', e.g. index 456 is stored as 'P456'.
    This function returns an int version of the string
    """
    return int(tree_str[1:])

def classify_prob_dict(prob_dict):
    """
    Given a probability dictionary mapping outcomes to probabilities, choose
    the most likely outcome and return it
    """
    best_outcome = None
    best_prob = 0
    for item in prob_dict:
        if prob_dict[item] > best_prob:
            best_prob = prob_dict[item]
        best_outcome = item
    return best_outcome

def mutate_sequence(sequence, location, new_value):
    """
    This function has two main phases
    The first is resolving the dependency graph of the component containing
    our targeted node.

    The second is resolving other structural issues/dependencies, which can
    be done at random.

    We return a list of probability dictionaries, where each dictionary
    corresponds to a set of outcomes from which we must choose, along with
    the probability of each outcome.

    Each dictionary is a discrete probability distribution over the possible
    set of outcomes for that component.
    """
    new_sequence = list(sequence)
    new_sequence[location] = new_value
    primary_outcomes = [
        (
            {location: new_value},
            1.0
        )
    ]

    if location not in TREE_DICT:
        return primary_outcomes

    visited = set()

    node_queue = deque()

    probability_list = []

    successors, predecessors = get_successors_and_predecessors(location)

    for x in successors:
        node_queue.append((x, location, 's'))
    for x in predecessors:
        node_queue.append((x, location, 'p'))



    visited.add(location)

    while len(node_queue) > 0:
        node, orig_node, node_type = node_queue.popleft()
        # print("Next node queue item: {}, {}, {}".format(node, orig_node, node_type))
        if node in visited:
            continue
        visited.add(node)
        if node not in TREE_DICT:
            continue
        # print("Outcome length: {}".format(len(primary_outcomes)))
        # for i, x in enumerate(primary_outcomes):
        #     for j, y in enumerate(primary_outcomes):
        #         if i != j:
        #             if x[0] == y[0]:
        #                 print("MATCH")
        #                 print(x)
        #                 print(y)

        if node_type == 'p':
            # if this is a predecessor tree, then node predicts orig_node
            # we expect node to be in the tree for orig_node
            current_tree = TREE_DICT[orig_node]
            target_leaf = get_best_node(current_tree, sequence[orig_node])
            path = leaf_to_root_path(current_tree, target_leaf)
            for i in range(len(path)-1, 0, -1):
                options = current_tree.edge_cond_[
                    (path[i-1], path[i])
                ]
                # print("OPTIONS")
                # print(options)
                # print(current_tree.feature[path[i-1]])
                current_tree_node = int(current_tree.feature[path[i-1]][1:])
                prob_update = 1.0/len(options)
                # print("PROBABILITY UPDATE: {}".format(prob_update))
                new_outcomes = []
                for outcome in primary_outcomes:
                    # make a copy
                    for nucleotide in options:
                        new_outcome = list(deepcopy(outcome))
                        new_outcome[0][current_tree_node] = nucleotide
                        new_outcome[1] *= prob_update
                        new_outcomes.append(tuple(new_outcome))
                primary_outcomes = new_outcomes

        if node_type == 's':
            # if this is a successor tree, then this means that orig_node
            # predicts node
            # we need to evaluate the node using current positions
            # and then decide
            new_outcomes = []
            current_tree = TREE_DICT[node]
            current_tree_node = 1
            target_pos = tree_str_to_index(current_tree.response_var_)
            for outcome in primary_outcomes:
                while not is_leaf(current_tree, current_tree_node):
                    current_pos = tree_str_to_index(current_tree.feature[current_tree_node])
                    if current_pos in outcome[0]:
                        target = outcome[0][current_pos]
                    else:
                        target = sequence[current_pos]
                    current_tree_node = get_child(
                        current_tree,
                        current_tree_node,
                        target,
                )
                normalized_pdict = normalize_probability_dict(current_tree.class_pred_[current_tree_node])
                for target in normalized_pdict:
                    if normalized_pdict[target] == 0:
                        continue
                    new_outcome = list(deepcopy(outcome))
                    new_outcome[0][target_pos] = target
                    orig = new_outcome[1]
                    new_outcome[1] *= normalized_pdict[target]
                    new_outcomes.append(tuple(new_outcome))
            primary_outcomes = new_outcomes

        gc.collect()
        if len(primary_outcomes) > 100000:
            return primary_outcomes
        successors, predecessors = get_successors_and_predecessors(node)

        for predecessor in predecessors:
            if predecessor not in visited and predecessor not in node_queue:
                node_queue.append((
                    predecessor,
                    node,
                    'p'
                ))
        for successor in successors:
            if successor not in visited and successor not in node_queue:
                node_queue.append((
                    successor,
                    node,
                    's'
                ))
    return primary_outcomes

def split_list(orig_list, chunk_size):
    for i in range(0, len(orig_list), chunk_size):
        yield orig_list[i:i+chunk_size]

def map_mutate_sequence(tup):
    sequence, selected_location, update_val = tup
    x = mutate_sequence(sequence, selected_location, update_val)
    return x

def perturb_sequence(sequence, location, replacement_val):
    new_sequence = list(sequence)
    primary_outcomes = mutate_sequence(sequence, location, replacement_val)

    secondary_outcomes_list = []

    tups = []

    num_cores = mp.cpu_count()

    if not PRIMARY_ONLY:
        for component in GRAPH_COMPONENTS:
            if location not in component:
                selected_location = random.choice(list(component))
                tups.append(
                    (
                        sequence,
                        selected_location,
                        sequence[selected_location]
                    )
                )
        print("Resolving {} components".format(len(tups)))
        chunks = [x for x in split_list(tups, num_cores * 5)]
        initial_results = []
        pool = mp.Pool(num_cores)
        secondary_outcomes_list = list(map(map_mutate_sequence, tups))
        pool.close()
        pool.join()
    return primary_outcomes, secondary_outcomes_list

primary_perturbations, secondary_perturbations = perturb_sequence(sequence, TARGET_LOCATION, TARGET_AA)

with open(OUTPUT_FILE, 'w') as fh:
    pickle.dump((primary_perturbations, secondary_perturbations), fh)
