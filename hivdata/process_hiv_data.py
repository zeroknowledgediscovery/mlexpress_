import os
import re
import math

root, folders, files = next(os.walk('.'))

counts = []

for fname in files:
    match = re.match(r'(.*?)_\.dat', fname)
    if match:
        new_name = match.group(1) + '.dat'
        with open(fname, 'r') as fh:
            rows = fh.read().strip().split('\n')

            for i, x in enumerate(rows):
                rows[i] = x.strip().split(' ')
                counts.append(len(rows[i]))

reference_count = min(counts)
index_row = list(map(str, list(range(reference_count))))

for fname in files:
    match = re.match(r'(.*?)_\.dat', fname)
    if match:
        prefix = match.group(1)
        print(fname)
        with open(fname, 'r') as fh:
            rows = fh.read().strip().split('\n')
            for i, x in enumerate(rows):
                rows[i] = x.strip().split(' ')
            train_rows = int(math.ceil(float(len(rows))/2))
            test_rows = len(rows)-train_rows
        with open(prefix + '_train.dat', 'w') as fh:
            fh.write(','.join(index_row))
            fh.write('\n')
            for i in range(train_rows):
                fh.write(','.join(rows[i][:reference_count]))
                fh.write('\n')
        with open(prefix + '_test.dat', 'w') as fh:
            fh.write(','.join(index_row))
            fh.write('\n')
            for i in range(test_rows, len(rows)):
                fh.write(','.join(rows[i][:reference_count]))
                fh.write('\n')
        with open('hivdata/' + prefix + '.dat', 'w') as fh:
            for i in range(len(rows)):
                fh.write(' '.join(rows[i][:reference_count]))
                fh.write('\n')
