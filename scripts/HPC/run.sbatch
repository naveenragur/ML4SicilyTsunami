#!/bin/bash

# options for sbatch
#SBATCH --job-name=trainTsuML # Job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=2 # number of cores per task
#SBATCH --time=0-06:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --mem=16G # memory pool for all cores
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

echo 'Running on' $region 'with' $size 'events' $mode 'mode' $masksize 'masksize'

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/HPC

# echo 'Running 00_filecheck.py'
# python 00_filecheck.py $region $size $mode $masksize

# echo 'Running 01_splitevents.py'
# python 01_splitevents.py $region $size $mode $masksize

# echo 'Running 02_preprocess.py'
# python 02_preprocess.py $region $size $mode $masksize

# echo 'Running 03_pretrain_offshore.py'
# python 03_pretrain_offshore.py $region $size $mode $masksize

# echo 'Running 04_pretrain_onshore.py'
# python 04_pretrain_onshore.py $region $size $mode $masksize

# echo 'Running 05_pretrain_deformation.py'
# python 05_pretrain_deformation.py $region $size $mode $masksize

echo 'Running 06_finetune_autoencoder.py'
python 06_finetune_autoencoder.py $region $size $mode $masksize

# echo 'Running 02_preprocess.py used to preprocess test events for evaluation'
# python 02_preprocess.py $region $size $mode $masksize 

# echo 'Running 07_evaluate_autoencoder.py'
# python 07_evaluate_autoencoder.py $region $size $mode $masksize

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
