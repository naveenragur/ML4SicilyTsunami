#Description: Checks offshore nc files and create event specific file and then overall summary statistics
#Usage: python 01_calcWaveheightarrival.py
import os
import numpy as np
import pandas as pd
import xarray as xr

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')

except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

ARRIVAL_THRESHOLD_M = 0.10
MISSING_ARRIVAL_TIME = -9999.0

def elapsed_seconds(time_values):
    if np.issubdtype(time_values.dtype, np.datetime64):
        return (time_values - time_values[0]) / np.timedelta64(1, 's')
    return time_values.astype(float) - float(time_values[0])

def calc_arrival_times(ts, time_values, threshold=ARRIVAL_THRESHOLD_M):
    time_seconds = elapsed_seconds(time_values)
    above_threshold = ts >= threshold
    has_arrival = above_threshold.any(axis=0)
    first_arrival_idx = np.argmax(above_threshold, axis=0)
    arrival_times = np.full(ts.shape[1], MISSING_ARRIVAL_TIME)
    arrival_times[has_arrival] = time_seconds[first_arrival_idx[has_arrival]]
    return np.round(arrival_times, 2)

def process_ts(file):
    #read data
    with xr.open_dataset(file) as data:
        ts = data['eta'].values
        maxTS = ts.max(axis=0).astype(float).round(2)
        arrival_times = calc_arrival_times(ts, data['time'].values)
        longitude = data['longitude'].values
        latitude = data['latitude'].values
        depth = data['deformed_bathy'].values

    #compile to dataframe #ID lon lat depth max_ssh arrival_time_10cm_sec
    df = pd.DataFrame({'ID':np.arange(len(maxTS)),
                        'lon':longitude,
                        'lat':latitude,
                        'depth':depth,
                        'max_ssh':maxTS,
                        'arrival_time_10cm_sec':arrival_times})

    #save to csv
    df.to_csv(file.replace('grid0_ts.nc','grid0_ts.nc.offshore.txt'),index=False,sep='\t')
 
    return maxTS, arrival_times

#file path for offshore time series
TSpath = SimDir + '/{:s}/grid0_ts.nc'

#output destination for summary statistics
OffshorePath = MLDir + '/data/info/grid0_allpts{:s}_alleveWH{:s}.offshore.txt'
OffshorePath2 = MLDir + '/data/info/grid0_allpts{:s}_alleveAT{:s}.offshore.txt'

#gauge variables: 
offshore_maxh = None
offshore_arrival = None

#Read event list from file
event_list = np.atleast_1d(np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str'))

#loop over events to calculate max offshore time series at 87 points
for i, event in enumerate(event_list):  
    #offshore
    max_values, arrival_times = process_ts(TSpath.format(event)) #process time series stats per event for all points

    if offshore_maxh is None:
        offshore_maxh = np.empty((len(event_list), len(max_values)))
        offshore_arrival = np.empty((len(event_list), len(arrival_times)))

    offshore_maxh[i, :] = max_values
    offshore_arrival[i, :] = arrival_times

#write summary statistics to file - offshore
offshore_maxh = pd.DataFrame(offshore_maxh, index=event_list)
offshore_maxh.index.name = 'id'
offshore_maxh.to_csv(OffshorePath.format(str(offshore_maxh.shape[1]),str(offshore_maxh.shape[0])),index=True,sep='\t')

offshore_arrival = pd.DataFrame(offshore_arrival, index=event_list)
offshore_arrival.index.name = 'id'
offshore_arrival.to_csv(OffshorePath2.format(str(offshore_arrival.shape[1]),str(offshore_arrival.shape[0])),index=True,sep='\t')
