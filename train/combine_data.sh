#!/bin/bash

#!/bin/bash
#
#SBATCH --job-name=data_concat
#
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --partition=rondror,deissero

eval "$(micromamba shell hook --shell bash)"

micromamba activate rhomax2

python combine_data2.py
