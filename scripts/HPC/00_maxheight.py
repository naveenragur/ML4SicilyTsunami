#Description: Checks offshore and onshore nc files and create summary statistics
#run: python 00_filecheck.py $region $size $mode $masksize #only reg used in this script
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#file path for offshore time series
Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc'
dZpath = SimDir + '/{:s}/{:s}_deformation.nc'
Zpath = MLDir + '/data/processed/{:s}_defbathy.nc' #hardcoded for now as I dont have defbathy.nc for each event

#output destination for summary statistics
Opath = SimDir + '/{:s}/CHeight_{:s}.nc.onshore.txt'
OnshorePath = MLDir + '/data/info/CHeight_{:s}_alleve{:s}.onshore.txt'

#gauge variables: time,eta,depth,velocity
features_name = ['id','count','max','logsum','mean','sd','dzmin','dzmax','hmax']
all_eve_df = pd.DataFrame(columns = features_name)

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str')

#loop over events to calculate max offshore time series at 87 points
#and max flow depth at CT and SR regions
for i, event in enumerate(event_list):  
    if i%100==0:
        print(f'Event {i} of {len(event_list)}') 
    #onshore
    eve_df = pd.DataFrame(columns = features_name)
    D_grids = xr.open_dataset(Dpath.format(event,reg)) #grid file of depth
    Z_grids = xr.open_dataset(Zpath.format(reg)) #grid file of bathymetry
    dZ_grids = xr.open_dataset(dZpath.format(event,reg)) #grid file of deformation

    D = D_grids.z
    dZ = dZ_grids.deformation
       
    Zval = Z_grids.z.values
    Dval = D_grids.z.values
    Dcount=D.count().values.astype(int)
    
    #on land values only used
    D = D.where(Zval > 0)
    dZ = dZ.where(Zval > 0)

    #set nan values to 0 before calculating height
    Zval[np.isnan(Zval)] = 0 #elevation
    Dval[np.isnan(Dval)] = 0 #depth
    
    Hval = Zval + Dval #height=depth+elevation
    Hval = np.where(Dval > 0, Hval, 0) 
    Hval = np.where(Zval > 0, Hval, 0) 
    
    df=[event,Dcount,
        D.max().values.astype(float).round(3),
        np.log(D.sum().values.astype(float)+1).round(3),
        D.mean().values.astype(float).round(3),
        D.std().values.astype(float).round(3),
        dZ.min().values.astype(float).round(3),
        dZ.max().values.astype(float).round(3),
        np.nanmax(Hval).round(3)]
    
    df = pd.DataFrame(df).T
    df.columns = features_name

    eve_df = pd.concat([eve_df,df])
    eve_df.to_csv(Opath.format(event,reg),index=False,sep='\t')
    eve_df = pd.DataFrame(columns = features_name)
    all_eve_df = pd.concat([all_eve_df,df])
         
#write summary statistics to file - onshore
all_eve_df.to_csv(OnshorePath.format(reg,str(all_eve_df.shape[0])),index=False,sep='\t')
