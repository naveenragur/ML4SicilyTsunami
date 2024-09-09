#!/bin/bash

#PBS -N trainTsuML
#PBS -o log
#PBS -j oe
#PBS -l walltime=02:00:00
##PBS -l mem=16G
#PBS -q mercalli_gpu
#PBS -l select=1:ncpus=32:mem=16G

source /soft/centos7/anaconda/2021.05/etc/profile.d/conda.sh
conda activate my_environment

cd $PBS_O_WORKDIR



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
