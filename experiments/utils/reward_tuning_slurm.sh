#!/bin/bash
  
#SBATCH --nodes=1                   # do not change
#SBATCH --gres=gpu:0                # id of gpu to use
#SBATCH --ntasks-per-node=10        # num of cores to allocate (cpu * (1/gpu) in configs)
#SBATCH --time=01-01:01             # Time limit (D-HH:MM)
#SBATCH -o %j.out                   # Standard output log file
#SBATCH -e %j.err                   # Standard err log file

which python
conda list | grep torch
conda list | grep cuda

python reward_tuning.py