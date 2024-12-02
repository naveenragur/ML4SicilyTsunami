#!/bin/bash
#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=pred_depth # Job name
#SBATCH --time=0-24:00 # time limit (D-HH:MM)
#SBATCH --ntasks=1 # total number of tasks (jobs) across all nodes (3 nodes x 10 jobs each)
#SBATCH --mem=256G # memory limit
#SBATCH --cpus-per-task=1 # number of CPU cores per task
#SBATCH --partition=cpuq
#SBATCH --nodelist=cn08
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

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi

# running commands
cd /mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard

#quick commands
# cat /proc/cpuinfo 
# export CUDA_VISIBLE_DEVICES=3
# # python checkgpu.py
# nvidia-smi

conda activate rioxarray_env
python prep_depth_true.py
python prep_depth_pred.py 892
python prep_depth_pred.py 1658
python prep_depth_pred.py 3454
python prep_depth_pred.py 7071

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds