# gen_scripts.py

This script is a modified version of the script that runs on midway to generate sbatch files for jobs. The script in question has been modified for local use.

## Usage

Open `gen_scripts.py` and update the following variables:

1. Update the `data_path` variable to be the path to where your CSVs are located
2. Update the `fnames` variable (list of prefixes). Note that we expect each file to be of the format `data_path/prefix_train.csv` and `data_path/prefix_test.csv`.
3. Update the `num_locations` variable to be in line with the number of expected values
4. Update the `num_cores` variable to be the number of cores you expect each machine to have.


## Design reasoning

1. Memory constraints: It's not computationally possible to run all ~8500 locations of HIV in parallel, at once. We need to distribute number of locations across multiple runs, write into a file corresponding to the ranges of each run, and then update accordingly.
2. Parallelization: We may wish to distribute runs across an arbitrary number of nodes.

So in exchange for relatively minimal local/manual overhead, we deal with both of these requirements.
