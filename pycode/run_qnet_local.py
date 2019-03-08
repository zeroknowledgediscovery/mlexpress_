import subprocess
import multiprocessing as mp
import os

prefixes = [
    'cchfl',
]

cutoffs = [
    0.75,
]

num_features = [
    400,
]

data_path = '../'
output_directory = '../test_output'

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

command = """python qNet.py --file {} --filex {} --varimp True --response {} --importance_threshold {} --edgefile {}.dat --dotfile {}.dot --tree_prefix {}"""

num_cores = mp.cpu_count()

for i, prefix in enumerate(prefixes):
    features = list(range(num_features[i]))
    feature_groups = [features[i:i + num_cores] for i in range(0, len(features), num_cores)]
    train_name = '{}_train.csv'.format(prefix)
    test_name = '{}_test.csv'.format(prefix)
    train_file = os.path.join(data_path, train_name)
    test_file = os.path.join(data_path, test_name)
    for cutoff in cutoffs:
        for group in feature_groups:
            current_command = command.format(
                train_file,
                test_file,
                ' '.join([str(x) for x in group]),
                cutoff,
                os.path.join(output_directory, '{}_{}-{}_{}'.format(prefix, group[0], group[-1], int(cutoff * 100))),
                os.path.join(output_directory, '{}_{}-{}_{}'.format(prefix, group[0], group[-1], int(cutoff * 100))),
                '{}_{}'.format(int(cutoff * 100), prefix),
            )
            print(current_command)
            subprocess.call([current_command])
