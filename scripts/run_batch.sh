#!/bin/bash
# 
# Author: Kevin Ingles
# File: run.sbatch
# Description: Command file to run GReX on UIUC cluster

# Embbeded options
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2048
#SBATCH --partition=qgp
#SBATCH --job-name=kingles_grex 
#SBATCH --output=slurm_output/bmm.o%j \
#SBATCH --error=slurm_output/bmm.e%j \
#SBATCH --mail-user=kingles@illinois.edu 
#SBATCH --mail-type=END

module load gcc/7.2.0
module load python/3
module load openblas/0.3.12_sandybridge

source /home/kingles/.bashrc

export TQDM_DISABLE=1
srun python3 RunBMMMCMC.py

