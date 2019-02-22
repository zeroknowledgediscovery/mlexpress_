import re
import os

# prefix = """#!/bin/bash
#
# #!/bin/bash
# #SBATCH --job-name={}
# #SBATCH --output={}.out
# #SBATCH --error={}.err
# #SBATCH --partition=broadwl
# #SBATCH --exclusive
# #SBATCH --time=12:00:00
# #SBATCH --account=pi-ishanu
#
# module load midway2;
# module load python/2.7.12-nompi
# module load R/3.3
# module load mkl/11.3
# export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_rt.so
# """

prefix = """"""

fnames = [
    'cchfl',
    # 'SPhiv',
    # 'ELITEhiv',
    # 'LTNPhiv',
    # 'Phiv',
]

"""
VALUES TO UPDATE
"""
python_path = os.path.abspath('../pycode/qNet.py')
data_path = '~/demo/mlexpress_/'
num_locations = 400
num_cores = 12
thresholds = [0.75]
"""
END OF VALUES TO UPDATE
"""

command = """
python {} --file {} --filex {}  --varimp True --response {}  --importance_threshold {} --edgefile {}.dat --dotfile {}.dot --tree_prefix {}
"""

orange = list(range(num_locations))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

range_chunks = []

for chunk in chunks(orange, num_cores):
    range_chunks.append(chunk)

commands = []

for threshold in thresholds:
    for fname in fnames:
        for chunk in range_chunks:
            commands.append((threshold, fname, tuple(chunk)))

x = 0
i = 0

sh = open('{}.sh'.format(x), 'w')
sbatch = open('{}.sbatch'.format(i), 'w')
fnames = [0]
sbatch.write(prefix.format(
    'qnet_{}'.format(i),
    'qnet_{}'.format(i),
    'qnet_{}'.format(i),
))
while i < len(commands):
    if (i % 500) == 0 and i != 0:
        x += 1
        sbatch.close()
        sbatch = open('{}.sbatch'.format(x), 'w')
    if (i % 20) == 0 and i != 0:
        sbatch.close()
        sbatch = open('{}.sbatch'.format(i), 'w')
        fnames.append(i)
        sbatch.write(prefix.format(
            'qnet_{}'.format(i),
            'qnet_{}'.format(i),
            'qnet_{}'.format(i),
        ))
    threshold, fname, chunk = commands[i]
    params = ' '.join([str(x) for x in list(chunk)])
    train_name = os.path.join(data_path, '{}_train.csv'.format(fname))
    test_name = os.path.join(data_path, '{}_test.csv'.format(fname))
    sbatch.write(command.format(
        python_path,
        train_name,
        test_name,
        params,
        threshold,
        '{}_{}-{}_{}'.format(fname, chunk[0], chunk[-1], int(threshold * 100)),
        '{}_{}-{}_{}'.format(fname, chunk[0], chunk[-1], int(threshold * 100)),
        '{}_{}'.format(int(threshold * 100), fname),
    ))
    i += 1

for i in fnames:
    sh.write('sbatch ./{}.sbatch\n'.format(i))
