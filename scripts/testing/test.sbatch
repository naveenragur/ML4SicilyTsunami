#!/bin/bash

#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=Test # Job name
#SBATCH --time=0-09:00 # time limit (D-HH:MM)
#SBATCH --partition=cpuq
#SBATCH --output ./log/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./log/error-%j.txt        # Standard err goes to this file


# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# setting up variables if provided
region=$1
size=$2
mode=$3
masksize=$4
split=$5
#TODO: add channels, zdim, and other hyper parameters  like batch size, learning rate, epoch of pretrian etc

# echo 'Running on' $region 'with' $size 'events' $mode 'mode' $masksize 'masksize'

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/testing

python checkgpu.py


# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
