#Description: Checks offshore and onshore nc files and create summary statistics
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import model_utils as utils

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
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
OffshorePath = MLDir + '/data/info/grid0_allpts{:s}_alleve{:s}.offshore.txt'

#gauge variables: time,eta,depth,velocity
features_name = ['id','count','max','logsum','mean','sd','dzmin','dzmax']
all_eve_df = pd.DataFrame(columns = features_name)
offshore_maxh = pd.DataFrame()

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{size}.txt', dtype='str')

#loop over events to calculate max offshore time series at 87 points
#and max flow depth at CT and SR regions
for i, event in enumerate(event_list):  
    #offshore
    utils.process_ts(TSpath.format(event)) #process time series stats per event for all points
    TS_pts = xr.open_dataset(TSpath.format(event)) #ts file of offshore points
    offshore_maxh[event] = pd.DataFrame(TS_pts.eta.max(dim='time').values.astype(float).round(2),
                            columns=[event])
    max_values = pd.DataFrame(TS_pts.eta.max(dim='time').values.astype(float).round(2), columns=[event])
    offshore_maxh = pd.concat([offshore_maxh, max_values], axis=1)

    #onshore
    eve_df = pd.DataFrame(columns = features_name)
    D_grids = xr.open_dataset(Dpath.format(event,reg)) #grid file of depth
    Z_grids = xr.open_dataset(Zpath.format(reg)) #grid file of bathymetry
    dZ_grids = xr.open_dataset(dZpath.format(event,reg)) #grid file of deformation

    D = D_grids.z
    dZ = dZ_grids.deformation
    Z = Z_grids.z.values
    D = D.where(Z > 0)
    dZ = dZ.where(Z > 0)

    
    df=[event,D.count().values.astype(int),
        D.max().values.astype(float).round(3),
        np.log(D.sum().values.astype(float)+1).round(3),
        D.mean().values.astype(float).round(3),
        D.std().values.astype(float).round(3),
        dZ.min().values.astype(float).round(3),
        dZ.max().values.astype(float).round(3)]
    
    df = pd.DataFrame(df).T
    df.columns = features_name

    eve_df = pd.concat([eve_df,df])
    eve_df.to_csv(Opath.format(event,reg),index=False,sep='\t')
    eve_df = pd.DataFrame(columns = features_name)
    all_eve_df = pd.concat([all_eve_df,df])
         
#write summary statistics to file - onshore
all_eve_df.to_csv(OnshorePath.format(reg,size),index=False,sep='\t')

#write summary statistics to file - offshore
offshore_maxh = offshore_maxh.T
offshore_maxh.index.name = 'id'
offshore_maxh.to_csv(OffshorePath.format(str(offshore_maxh.shape[1]),str(offshore_maxh.shape[0])),index=True,sep='\t')

