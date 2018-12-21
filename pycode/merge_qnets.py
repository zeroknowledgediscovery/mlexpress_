import sys
import os
import re

path = sys.argv[1]

root, folders, files = next(os.walk(path))

prefixes = [
    'SPhiv',
    # 'RPhiv',
    'LTNPhiv',
    'ELITEhiv',
    'Phiv',
]

fpattern = r'(.*?)_.*?_([0-9]+)\.dot'
datfpattern = r'(.*?)_.*?_([0-9]+)\.dat'
datpattern = r'"\(\'(.*?)\', \'(.*?)\'\)",([0-9]+\.[0-9]+)'
dotpattern = r'(P[0-9]+) -> (P[0-9]+)'

template = """digraph {{
{}
}}
"""

cutoffs = set()

for fname in files:
    match = re.match(fpattern, fname)
    if match:
        cutoffs.add(int(match.group(2)))

for prefix in prefixes:
    for cutoff in cutoffs:
        edges = set()
        for fname in files:
            match = re.match(fpattern, fname)
            if match:
                if match.group(1) == prefix and match.group(2) == str(cutoff):
                    with open(os.path.join(root, fname), 'r') as fh:
                        text = fh.read()
                        for match in re.finditer(dotpattern, text):
                            x, y = match.group(1), match.group(2)
                            edges.add((x, y))
        edges = list(edges)
        edges = ['{} -> {}'.format(x[0], x[1]) for x in edges]
        if len(edges) == 0:
            continue
        with open('../QNET_COMPILED/{}_{}.dot'.format(prefix, cutoff), 'w') as fh:
            fh.write(template.format(',\n'.join(edges)))

cutoff = 75
sub_cutoffs = [0.9, 0.925, 0.95, 0.975, 0.98, 0.99, 0.995]
for sub_cutoff in sub_cutoffs:
    edges = set()
    for fname in files:
        match = re.match(datfpattern, fname)
        if match:
            if match.group(1) == 'RPhiv' and match.group(2) == str(cutoff):
                with open(os.path.join(root, fname), 'r') as fh:
                    text = fh.read()
                    for match in re.finditer(datpattern, text):
                        if float(match.group(3)) >= sub_cutoff:
                            edges.add((match.group(1), match.group(2)))
    edges = list(edges)
    edges = ['{} -> {}'.format(x[0], x[1]) for x in edges]
    if len(edges) == 0:
        continue
    with open('../QNET_COMPILED/RPhiv_{}.dot'.format(sub_cutoff), 'w') as fh:
        fh.write(template.format(',\n'.join(edges)))
