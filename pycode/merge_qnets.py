import sys
import os
import re

path = sys.argv[1]

root, folders, files = next(os.walk(path))

OUTPUT_DIR = 'QNETNEW_COMPILED'

prefixes = [
    'SPhiv',
    'RPhiv',
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
    print(fname)
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
        with open('../{}/{}_{}.dot'.format(OUTPUT_DIR, prefix, cutoff), 'w') as fh:
            fh.write(template.format(',\n'.join(edges)))
