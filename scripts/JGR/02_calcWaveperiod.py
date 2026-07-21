#Description: Checks offshore netcdf files and create summary statistics of the wave period and dominant period
#Usage: python 02_calcWaveperiod.py <region>
import os
import numpy as np
import pandas as pd
import xarray as xr

import scipy.signal
from scipy.fft import fft, fftfreq

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

def process_ts(file):
    #read data
    data = xr.open_dataset(file)
    ts = data['eta'].values
    maxTS = ts.max(axis=0)
    minTS = ts.min(axis=0)
    gperiod = []
    dperiod = []
    gpolarity = []
    greturn_code = []

    for g in range(87):
        #find peaks(positive and negative)
        ppeaks, _ = scipy.signal.find_peaks(ts[:,g], height=0.05,distance=10)
        npeaks, _ = scipy.signal.find_peaks(-ts[:,g], height=0.05,distance=10)

        #find polarity of wave based on positive and negative peaks indices
        if len(ppeaks)==0 and len(npeaks)==0:
            polarity = '0'
        elif len(ppeaks)==0:
            polarity = '-1'
        elif len(npeaks)==0:
            polarity = '+1'
        elif ppeaks[0]<npeaks[0]:
            polarity = '+1'
        elif ppeaks[0]>npeaks[0]:
            polarity = '-1'
        else:
            polarity = '0'

        #find waveperiod
        if polarity == '0':
            waveperiod = 0
        elif polarity == '+1':
            if len(ppeaks)==1:
                waveperiod = 0
            else:
                waveperiod = (ppeaks[1]-ppeaks[0])
        elif polarity == '-1':
            if len(npeaks)==1:
                waveperiod = 0
            else:
                waveperiod = (npeaks[1]-npeaks[0])

        #return code
        if polarity == '0':
            return_code = 1
        elif polarity == '+1' or polarity == '-1':
            return_code = 3

        gperiod.append(waveperiod)
        gpolarity.append(polarity)
        greturn_code.append(return_code)

        # Define the time step for the dataset in minutes
        time_step_minutes = 0.5

        # Perform Fourier analysis on the tsunami data using scipy.fft.
        tsunami_data = ts[1:,g] - np.mean(ts[1:,g])
        fft_tsunami_data = fft(tsunami_data)
        freqs = fftfreq(len(tsunami_data), d=time_step_minutes)
        psd = np.abs(fft_tsunami_data) ** 2

        positive_freqs = freqs > 0
        if np.any(psd[positive_freqs] > 0):
            max_psd_idx = np.argmax(psd[positive_freqs])
            max_wave_period = 1/freqs[positive_freqs][max_psd_idx]
        else:
            max_wave_period = 0

        #round to 1 decimal place
        dperiod.append(round(max_wave_period,1))

    gperiod = np.array(gperiod)
    dperiod = np.array(dperiod)
    gpolarity = np.array(gpolarity)
    greturn_code = np.array(greturn_code)

    #compile to dataframe #ID lon lat depth max_ssh min_ssh period polarity return_code
    df = pd.DataFrame({'ID':np.arange(87),
                        'lon':data['longitude'].values,
                        'lat':data['latitude'].values,
                        'depth':data['deformed_bathy'].values,
                        'max_ssh':maxTS,
                        'min_ssh':minTS,
                        'period':gperiod,
                        'dperiod':dperiod, #dperiod is 'dominant period
                        'polarity':gpolarity,
                        'return_code':greturn_code})

    #save to csv
    df.to_csv(file.replace('grid0_ts.nc','grid0_ts.nc.offshore.txt'),index=False,sep='\t')

    return df['period'],df['dperiod']

#file path for offshore time series
TSpath = SimDir + '/{:s}/grid0_ts.nc'

#output destination for summary statistics
OffshorePath = MLDir + '/data/info/grid0_allpts{:s}_alleveWP{:s}.offshore.txt'
OffshorePath2 = MLDir + '/data/info/grid0_allpts{:s}_alleveDP{:s}.offshore.txt'

#gauge variables: time,eta,depth,velocity
features_name = ['id','count','max','logsum','mean','sd','dzmin','dzmax']
all_eve_df = pd.DataFrame(columns = features_name)
offshore_wp = pd.DataFrame()
offshore_dp = pd.DataFrame()

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str') 

#loop over events to calculate max offshore time series at 87 points
#and max flow depth at CT and SR regions
for i, event in enumerate(event_list): 
    if i%100==0:
        print(f'Event {i} of {len(event_list)}') 
    #offshore
    wp,dp = process_ts(TSpath.format(event)) #process time series stats per event for all points
    TS_pts = xr.open_dataset(TSpath.format(event)) #ts file of offshore points
    
    offshore_wp = pd.concat([offshore_wp, wp], axis=1)
    offshore_dp = pd.concat([offshore_dp, dp], axis=1)
    #rename column period columns to event id
    offshore_wp = offshore_wp.rename(columns={'period':event})
    offshore_dp = offshore_dp.rename(columns={'dperiod':event})

# write summary statistics to file - offshore(Wave period and dominant period)
offshore_wp = offshore_wp.T  #transpose
offshore_wp.index.name = 'id' #rename index
offshore_wp.to_csv(OffshorePath.format(str(offshore_wp.shape[1]),str(offshore_wp.shape[0])),index=True,sep='\t')

offshore_dp = offshore_dp.T  #transpose
offshore_dp.index.name = 'id' #rename index
offshore_dp.to_csv(OffshorePath2.format(str(offshore_dp.shape[1]),str(offshore_dp.shape[0])),index=True,sep='\t')
