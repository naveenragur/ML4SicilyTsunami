#Description: Checks offshore and onshore nc files and create summary statistics
#run: python 00_filecheck.py $region $size $mode $masksize #only reg used in this script
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import model_utils as utils

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#file path for offshore time series
TSpath = SimDir + '/{:s}/grid0_ts.nc'
Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc'
dZpath = SimDir + '/{:s}/{:s}_deformation.nc'
Zpath = MLDir + '/data/processed/{:s}_defbathy.nc' #hardcoded for now as I dont have defbathy.nc for each event

#output destination for summary statistics
Opath = SimDir + '/{:s}/CDepth_{:s}.nc.onshore.txt'
OnshorePath = MLDir + '/data/info/CDepth_{:s}_alleve{:s}.onshore.txt'
OffshorePath = MLDir + '/data/info/grid0_allpts{:s}_alleveWP{:s}.offshore.txt'

#gauge variables: time,eta,depth,velocity
features_name = ['id','count','max','logsum','mean','sd','dzmin','dzmax']
all_eve_df = pd.DataFrame(columns = features_name)
offshore_wp = pd.DataFrame()

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str') 

#loop over events to calculate max offshore time series at 87 points
#and max flow depth at CT and SR regions
for i, event in enumerate(event_list): 
    if i%100==0:
        print(f'Event {i} of {len(event_list)}') 
    #offshore
    wp = utils.process_ts(TSpath.format(event)) #process time series stats per event for all points
    TS_pts = xr.open_dataset(TSpath.format(event)) #ts file of offshore points
    offshore_wp = pd.concat([offshore_wp, wp], axis=1)
    #rename column to event id
    offshore_wp = offshore_wp.rename(columns={'period':event})


# write summary statistics to file - offshore
offshore_wp = offshore_wp.T
offshore_wp.index.name = 'id'
offshore_wp.to_csv(OffshorePath.format(str(offshore_wp.shape[1]),str(offshore_wp.shape[0])),index=True,sep='\t')

