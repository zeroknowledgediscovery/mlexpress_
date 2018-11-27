import sys
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import multiprocessing as mp

path = sys.argv[1]

root, folders, files = next(os.walk(path))

prefixes = [
    'SPhiv',
    'Phiv',
    'LTNPhiv',
    'ELITEhiv',
    'RPhiv',
]

dims = {
    'SPhiv': (12, 12),
    'LTNPhiv': (50, 50),
    'Phiv': (200, 200),
    'RPhiv': (200, 200),
}

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

def plot(dict, prefix, subtype, cutoff, max_val = 8084):
    fig = plt.figure(figsize=(16, 10))
    fig.tight_layout()
    x = [0] * (max_val + 1)
    for item in dict:
        x[item] = dict[item]
    y = x
    x = range(len(y))
    plt.scatter(x, y)
    plt.savefig('graph_metadata/{}_{}_{}.png'.format(prefix, subtype, cutoff))
    plt.close()

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
        deg_centrality = nx.degree_centrality(graph)
        average_neighbor_degree = nx.average_neighbor_degree(graph)

        # plot(deg_centrality, 'deg_centrality', prefix, cutoff)
        # plot(average_neighbor_degree, 'avg_neighbor_deg', prefix, cutoff)

def barplot(info_dict, graph_name, graph_title, cutoff):
    xs = range(len(info_dict))
    labels = [x for x in info_dict]
    ys = [info_dict[y] for y in info_dict]
    fig = plt.figure(figsize=(16, 10))
    fig.tight_layout()
    plt.bar(xs, ys, align='center', tick_label = labels)
    plt.title(graph_title)
    plt.savefig('graph_metadata/{}_{}.png'.format(graph_name, cutoff))
    plt.close()

for cutoff in cutoffs:
    average_dict = {}
    count_dict = {}
    max_dict = {}
    for prefix in prefixes:
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
        ud = graph.to_undirected(as_view = True)

        counts = []
        for component in nx.connected_components(ud):
            counts.append(len(component))
        average_dict[prefix] = sum(counts)/len(counts)
        count_dict[prefix] = len(counts)
        max_dict[prefix] = max(counts)
    xs = len(average_dict)

    barplot(average_dict, 'average_chain_length', 'Average component length vs Subtype', cutoff)
    barplot(count_dict, 'num_chains', 'Number of components vs Subtype', cutoff)
    barplot(max_dict, 'max_length', 'Maximum chain length vs Subtype', cutoff)
