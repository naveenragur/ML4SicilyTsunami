#Prepare the code for the JGR submission  
#Date: 06/05/2025
#Author: Naveen Ragu Ramalingam (naveen.ragu@ngi.no)

#Script to run in order to prepare the data for machine learning emulation

1. Python script 00_calcFlowdpeth.py is run for each region (CT or SR) to create _flowdepth.nc, _deformation.nc and _height.nc files. Additionally, the script also writes the summary statistics of inundation across the eventset ('id','count','dmax','logsum','mean','sd','dzmin','dzmax','hmax') to a csv file.

2. Python script 01_check_Offshore.py is run to check the offshore data. The script checks the offshore data and creates a txt file per event, with the overall summary statistics of the offshore data for the whole eventset ('id','count','dmax','logsum','mean','sd','dzmin','dzmax','hmax').

3. Python script 00