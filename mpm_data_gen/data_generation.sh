#!/bin/bash

for run_number in {1..300}
do
    python3 data_generation.py --run_number $run_number 
    echo $run_number
done
