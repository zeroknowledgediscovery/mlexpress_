#!/bin/bash

# run_qnet_local will iteratively call qNet in a subprocess to generate all of the decision trees and edge files
# those files are stored in test_output
python run_qnet_local.py ../ ../test_output --prefixes cchfl --cutoffs 75 --num_features 400
# output generated in ../test_output

# networkx_qnets will run on the .dot files generated in the first step and create a PNG of the qnet
# the default resolution is sufficient for the cchfl example
python3 networkx_qnets.py ../test_output --output_dir ../test_output --prefixes cchfl
# output also stored there
