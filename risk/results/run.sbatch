#!/bin/bash
#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=PML_test # Job name
#SBATCH --time=0-24:00 # time limit (D-HH:MM)
#SBATCH --ntasks=1 # total number of tasks (jobs) across all nodes (3 nodes x 10 jobs each)
#SBATCH --cpus-per-task=1 # number of CPU cores per task
#SBATCH --partition=cpuq
#SBATCH --nodelist=cn02
# #SBATCH -p gpuq # partition
# #SBATCH --gres=gpu:1 # number of GPUs
# #SBATCH --nodelist=gp02
#SBATCH --output ./log/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./log/error-%j.txt      # Standard err goes to this file

# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# setting up variables if provided
# region=$1
# size=$2
# mode=$3
# masksize=$4
# split=$5

# echo 'Running on' $region 'with' $size 'events' $mode 'mode' $masksize 'masksize'

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
# conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/risk/results/

#quick commands
# cat /proc/cpuinfo 
# export CUDA_VISIBLE_DEVICES=3
# # python checkgpu.py
# nvidia-smi

#For comparison plots
# conda deactivate
conda activate rioxarray_env
python calc_rates.py

# python calc_pml_aal.py CT asset true 50
# python calc_pml_aal.py CT asset 892 50
# python calc_pml_aal.py CT asset 1658 50
# python calc_pml_aal.py CT asset 3454 50
# python calc_pml_aal.py CT asset 7071 50
# python calc_pml_aal.py CT foglio true 50
# python calc_pml_aal.py CT foglio 892 50
# python calc_pml_aal.py CT foglio 1658 50
# python calc_pml_aal.py CT foglio 3454 50
# python calc_pml_aal.py CT foglio 7071 50

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds