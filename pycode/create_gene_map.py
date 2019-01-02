import sys
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

path = sys.argv[1]

root, folders, files = next(os.walk(path))

low_tier_map = {
    "5' LTR": [(1, 634)],
    'p17': [(1790, 1186)],
    'p24': [(1186, 1879)],
    'p2': [(1879, 1921)],
    'p7': [(1921, 2086)],
    'p1': [(2086, 2135)],
    'p6': [(2134, 2292)],
    'prot': [(2253, 2550)],
    'p51_RT': [(2550, 3870)],
    'p15_RNAse': [(3870, 4230)],
    'p31_int': [(4230, 5096)],
    'vif': [(5041, 5619)],
    'vpr': [(5559, 5850)],
    'tat': [(5831, 6045), (8379, 8469)],
    'rev': [(5850, 6050), (8379, 8653)],
    'gp120': [(6045, 7758)],
    'gp41': [(7758, 8795)],
    'nef': [(8797, 9417)],
    '3_LTR': [(9086, 9719)],
}

high_tier_map = {
    "5' LTR": [(1, 634)],
    'gag': [(790, 2292)],
    'pol': [(2085, 5096)],
    'vif': [(5041, 5619)],
    'env': [(6045, 8795)],
    'tat': [(5831, 6045), (8379, 8469)],
    'rev': [(5850, 6050), (8379, 8653)],
    'nef': [(8797, 9417)],
    '3_LTR': [(9086, 9719)],
}

def get_labels(value, label_map):
    labels = []
    for item in label_map:
        for interval in label_map[item]:
            if value in range(*interval):
                labels.append(item)
                break
    return labels

def get_tuple_labels(tup, label_map):
    x, y = tup
    return (
        get_labels(x, label_map),
        get_labels(y, label_map),
    )

def get_sorted_counts(graph, label_map):
    labels = []
    for edge in nx.edges(graph):
        xs, ys = get_tuple_labels(edge, label_map)
        if not xs or not ys:
            continue
        for x in xs:
            for y in ys:
                labels.append((x, y))
    return Counter(labels).most_common(20)

def strfmt(tup):
    return '{} -> {}'.format(tup[0], tup[1])

def pred_bar_plot(count_list, prefix, cutoff):
    fig = plt.figure(figsize=(16, 10))
    plt.style.use('seaborn')
    fig.tight_layout()
    labels = list(reversed([strfmt(x[0]) for x in count_list]))
    ys = pd.Series(list(reversed([x[1] for x in count_list])))
    ax = ys.plot(kind='barh')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Location predictions')
    ax.set_title('{}, cutoff {}'.format(prefix, cutoff))
    ax.set_yticklabels(labels)
    plt.savefig('../pred_bar_plot/{}_{}.png'.format(prefix, cutoff))
    plt.close()

path = sys.argv[1]

root, folders, files = next(os.walk(path))

dims = {
    'SPhiv': (12, 12),
    'LTNPhiv': (50, 50),
    'Phiv': (200, 200),
    'RPhiv': (200, 200),
}

prefixes = [
    'SPhiv',
    'Phiv',
    'LTNPhiv',
    'ELITEhiv',
    'RPhiv',
]

fpattern = r'(.*?)_.*?_([0-9]+)\.dot'
datfpattern = r'(.*?)_.*?_([0-9]+)\.dat'
datpattern = r'"\(\'(.*?)\', \'(.*?)\'\)",([0-9]+\.[0-9]+)'
dotpattern = r'P([0-9]+) -> P([0-9]+)'

cutoffs = set()

for fname in files:
    match = re.match(fpattern, fname)
    if match:
        cutoffs.add(int(match.group(2)))

params = []

for prefix in prefixes:
    for cutoff in cutoffs:
        params.append((prefix, cutoff))

for prefix in prefixes:
    for cutoff in cutoffs:
        print("{} at {}".format(prefix, cutoff))
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
        if prefix in dims:
            dim = dims[prefix]
        else:
            dim = (50, 50)
        if len(edges) == 0 or len(nodes) == 0:
            continue
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        low_counts = get_sorted_counts(graph, low_tier_map)
        spec = prefix + '_low_level'
        pred_bar_plot(low_counts, spec, cutoff)
        high_counts = get_sorted_counts(graph, high_tier_map)
        spec = prefix + '_high_level'
        pred_bar_plot(high_counts, spec, cutoff)
