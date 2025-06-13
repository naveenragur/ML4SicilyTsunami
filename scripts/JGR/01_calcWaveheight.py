#Description: Checks offshore nc files and create event specific file and then overall summary statistics
#Usage: python 01_calcWaveheight.py <region>
import os
import sys
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

        # Define the time range for the entire dataset (600 minutes)
        total_time_minutes = 240
        time_step_minutes = 0.5
        time = np.arange(0, total_time_minutes, time_step_minutes)

        # Perform Fourier analysis on the tsunami data using scipy.fft
        fft_tsunami_data = fft(ts[1:,g])
        freqs = fftfreq(len(time), d=time_step_minutes)
        psd = np.abs(fft_tsunami_data) ** 2       
        #wave period with max spectral density
        max_psd_idx = np.argmax(psd)
        max_wave_period = 1/freqs[max_psd_idx]
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
OffshorePath = MLDir + '/data/info/grid0_allpts{:s}_alleve{:s}.offshore.txt'

#gauge variables: 
offshore_maxh = pd.DataFrame()

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_extreme371.txt', dtype='str') 

#loop over events to calculate max offshore time series at 87 points
for i, event in enumerate(event_list):  
    #offshore
    process_ts(TSpath.format(event)) #process time series stats per event for all points
    TS_pts = xr.open_dataset(TSpath.format(event)) #ts file of offshore points
    max_values = pd.DataFrame(TS_pts.eta.max(dim='time').values.astype(float).round(2), columns=[event])
    offshore_maxh = pd.concat([offshore_maxh, max_values], axis=1)

#write summary statistics to file - offshore
offshore_maxh = offshore_maxh.T
offshore_maxh.index.name = 'id'
offshore_maxh.to_csv(OffshorePath.format(str(offshore_maxh.shape[1]),str(offshore_maxh.shape[0])),index=True,sep='\t')

