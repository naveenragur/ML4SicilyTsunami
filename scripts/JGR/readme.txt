#Readme for the codebase used for the 2025 JGR:ML and Computations submission of the ML4SicilyTsunami Emulator 
#Date: 06/05/2025
#Author: Naveen Ragu Ramalingam (naveen.ragu@ngi.no)

#Main Workflow to be followed for tsunami emulation, we use slurm  for job scheduling using sbatch run.sbatchCT and run.sbatchSR for the two regions of interest (CT and SR).

1. Scripts 00_calcFlowdepth.py, 01_calcWaveheight.py and 02_calcWaveperiod.py are run to prepare some useful event statistics and summaries for selecting events for training.

2. Notebooks 03_sample_eve_CT.ipynb and 03_sample_eve_SR.ipynb are used to sample events from the INGV database. The events are selected based on the event statistics calculated in the previous step. The events list are then saved in /data/events/ as txt files.

3. Script 04_preprocess.py is run to prepare the data for machine learning emulation and saved in /data/processed/, a mask size derived from a set of sample events is used for maintaining homogenity in the locations of prediction.

4. Script train.py and test.py are used to train and test the machine learning models. The models are trained on the data prepared in the previous step. The models and predictions are saved in /models/<region>/<task>.

5. 

#Other scripts used in the codebase but not run directly by the user

checkgpu.py #This script checks if a GPU is available and prints the name of the GPU if it is available.
experiment.py #This script is used to run the machine learning experiments. It contains all the functions need for data loading training, prediction and testing.
