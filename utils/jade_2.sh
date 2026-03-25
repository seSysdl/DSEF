#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=
#SBATCH --output=ClusteringMemory_CIFAR10toMNIST_DynamicExpansion_39_32.log            # Standard output and error log

# run the application
python ClusteringMemory_CIFAR10toMNIST_DynamicExpansion_39_32.py