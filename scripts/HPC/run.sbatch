#!/bin/bash

# options for sbatch
#SBATCH --job-name=trainTsuML # Job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --time=0-01:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --mem=16G # memory pool for all cores
#SBATCH --output ./log/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./log/error-%j.txt        # Standard err goes to this file

# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# setting up variables
region=$1
size=$2

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/HPC
python 01_preprocess.py $region $size


# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds