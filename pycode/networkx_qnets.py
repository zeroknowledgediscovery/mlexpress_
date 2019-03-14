import sys
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('graph_dot_path', help='Path to folder containing edge (.dot) files generated by qNet.py')
parser.add_argument('--output_dir', help='Folder to store resulting pngs in', default='.')
parser.add_argument('--prefixes', nargs='+', help='Subtype prefixes (e.g. SPhiv, Phiv, cchfl)', default=['cchfl'])
results = parser.parse_args()

prefixes = results.prefixes
path = results.graph_dot_path
output_dir =results.output_dir

root, folders, files = next(os.walk(path))

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

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for prefix in prefixes:
    for cutoff in cutoffs:
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
        fig = plt.figure(figsize=dim)
        fig.tight_layout()
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(graph)
        nx.draw(
            graph,
            pos=pos,
            with_labels = True,
            node_size = 1000,
            node_color = 'white',
        )
        ax = plt.gca() # to get the current axis
        ax.collections[0].set_edgecolor("#000000")
        out_name = os.path.join(
            output_dir,
            '{}_{}.png'.format(prefix, cutoff),
        )
        print("Saving to {}".format(out_name))
        plt.savefig(
            out_name,
            bbox_inches = 'tight',
        )
        plt.close()
