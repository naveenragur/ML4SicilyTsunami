#!/bin/bash

# options for sbatch
#SBATCH --job-name=trainTsuML # Job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --time=0-02:00 # time limit (D-HH:MM)
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
trainsize=$2
mode=$3
testsize=$4
#TODO: add channels, zdim, and other hyper parameters  like batch size, learning rate, epoch of pretrian etc

echo 'Running on' $region 'with' $size 'events'

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/HPC
# echo 'Running 01_splitevents.py'
# python 01_splitevents.py $region $size

# echo 'Running 02_preprocess.py'
# python 02_preprocess.py $region $size $mode

# echo 'Running 03_pretrain_offshore.py'
# python 03_pretrain_offshore.py $region $size

# echo 'Running 04_pretrain_onshore.py'
# python 04_pretrain_onshore.py $region $size

# echo 'Running 05_finetune_autoencoder.py'
# python 05_finetune_autoencoder.py $region $size

# echo 'Running 02_preprocess.py'
# python 02_preprocess.py $region $size $mode

echo 'Running 06_evaluate_autoencoder.py'
python 06_evaluate_autoencoder.py $region $trainsize $mode $testsize

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds