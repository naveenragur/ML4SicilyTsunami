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
ncpath = SimDir + '/{:s}/C_{:s}.nc' #contains max height and deformation 
Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc' #to write out max flow depth
dZpath = SimDir + '/{:s}/{:s}_deformation.nc' #to write out max deformation
Hpath = SimDir + '/{:s}/{:s}_height.nc' #to write out max height
Zpath = MLDir + '/data/processed/{:s}_defbathy.nc' #hardcoded for now as I dont have defbathy.nc for each event

#output destination for summary statistics
Opath = SimDir + '/{:s}/CHeight_{:s}.nc.onshore.txt'
OnshorePath = MLDir + '/data/info/CHeight_{:s}_alleve{:s}.onshore.txt'

#gauge variables: time,eta,depth,velocity
features_name = ['id','count','dmax','logsum','mean','sd','dzmin','dzmax','hmax']
all_eve_df = pd.DataFrame(columns = features_name)

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str')

#loop over events to calculate max offshore time series at 87 points
#and max flow depth at CT and SR regions
for i, event in enumerate(event_list):  
    if i%100==0:
        print(f'Event {i} of {len(event_list)}') 
    
    #onshore stats
    eve_df = pd.DataFrame(columns = features_name)
    
    #read nc files
    nc_grids = xr.open_dataset(ncpath.format(event,reg)) #grid file of depth
    Z_grids = xr.open_dataset(Zpath.format(reg)) #grid file of topobathymetry elevation
    dZ_grids = xr.open_dataset(dZpath.format(event,reg)) #grid file of deformation 

    #deformed topobathy-elevation
    deformed_ele = Z_grids.z.values - nc_grids.deformation.values

    #max height
    height = nc_grids.max_height.values #max height
    height = xr.DataArray(height, dims=nc_grids.max_height.dims, coords=nc_grids.max_height.coords,name='z') #create xr dataset
    height = height.where(Z_grids.z.values>0)  #mask to predeformed land mask
    height.to_netcdf(Hpath.format(event,reg),mode='w') #save as nc file

    #max flow depth
    depthval = nc_grids.max_height.values - deformed_ele #max flow depth
    depth = xr.DataArray(depthval, dims=nc_grids.max_height.dims, coords=nc_grids.max_height.coords,name='z') #create xr dataset
    depth = depth.where(Z_grids.z.values>0)  #mask to predeformed land mask
    depth.to_netcdf(Dpath.format(event,reg),mode='w') #save as nc file

    #calculate summary statistics
    D = depth
    dZ = dZ_grids.deformation   

    Zval = Z_grids.z.values
    Dval = depth.values
    Hval = height.values

    Dcount=D.count().values.astype(int)
    
    #stats are calculated on predeformed land values only
    dZ = dZ.where(Zval > 0)

    #set nan values to 0 before calculating height
    Zval[np.isnan(Zval)] = 0 #elevation

    #features_name = ['id','count','dmax','logsum','mean','sd','dzmin','dzmax','hmax']   
    df=[event,
        Dcount,
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
print(all_eve_df)
