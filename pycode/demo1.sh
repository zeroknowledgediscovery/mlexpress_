#!/bin/bash

python run_qnet_local.py ../ ../test_output --prefixes cchfl --cutoffs 75 --num_features 400
# output generated in ../test_output

python3 networkx_qnets.py ../test_output --output_dir ../test_output --prefixes cchfl
# output also stored there
