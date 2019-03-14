#!/bin/bash

python perturb_sequence.py ../test_output --prefixes cchfl ../test_output --target_location 135 --target_amino_acid A --cutoffs 75 --input_sequence ../cchfl_sample_seq.txt --end_subtype cchfl --end_cutoff 75 --output_file ../test_output/CCHFL_DEMO.pkl

python generate_perturbed_sequences.py ../cchfl_sample_seq.txt ../test_output/CCHFL_DEMO.pkl ../test_output/CCHFL_GENERATED.csv
