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
    # 'Phiv',
    # 'LTNPhiv',
    # 'ELITEhiv',
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
        plt.savefig(
            '{}_{}.png'.format(prefix, cutoff),
            bbox_inches = 'tight',
        )
        plt.close()
        print(prefix, cutoff)
