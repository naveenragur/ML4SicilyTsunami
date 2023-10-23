#Description: Preprocess the data offshore and onshore data for training and testing
import os
import sys

import numpy as np
import xarray as xr

import model_utils as utils

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size used for training or test
    mode = sys.argv[3] #train or test
    mask_size = sys.argv[4] #eventset size for testing
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

# file_list = glob.glob('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/PS_manning003/**/*SR_defbathy.nc')
# for file in file_list:
#     print(file)
#     # print(os.path.join(file.rsplit('/',1)[-2],'SR_defbathy.nc'))
#     os.rename(file,os.path.join(file.rsplit('/',1)[-2],'SR_defbathy.nc')) 

if mode == 'train':
    #read event list from file
    event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', dtype='str')
elif mode == 'test':
    #or whole directory for final evaluation
    # event_list = os.listdir(f'{SimDir}')
    # event_list = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt', dtype='str')
    # np.random.shuffle(event_list)
    # event_list = event_list[:int(size)]
    # #save event list
    # np.savetxt(f'{MLDir}/data/events/shuffled_events_{mode}_{size}.txt', event_list, fmt='%s')
    event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_test_{reg}_{size}.txt', dtype='str')

#note size is the size of selection but not the number of events in the actual event list which can be less 
#than the size of selection if there are events that dont satisfy threshold criterias

#string template for file filepath
Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc'
Hpath = SimDir + '/{:s}/{:s}_hmax.nc'
# Zpath = SimDir + '/{:s}/{:s}_defbathy.nc' #original
Zpath = MLDir + '/data/processed/{:s}_defbathy.nc' #hardcoded for now as I dont have defbathy.nc for each event
dZpath = SimDir + '/{:s}/{:s}_deformation.nc'
TSpath = SimDir + '/{:s}/grid0_ts.nc'

#TODO: could be read from a parameter file instead of hardcoding here
#dimensions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #gauges time series
    pts_dim = 480 #time steps
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo)
    pts_dim = 480

#empty arrays
d_array = np.zeros((len(event_list), y_dim, x_dim))  #say for SR: no of events x 948(y) x 1300(x) x 
dZ_array = np.zeros((len(event_list), y_dim, x_dim))  #say for SR: no of events x 948(y) x 1300(x) x
t_array = np.zeros((len(event_list), pts_dim, ts_dim))  #say for SR: no of events x 480(time steps) x 5(gauges)

#loop over events 
#TODO: if needed parallelize? the event list are split into chunks,process each chunk in parallel and merge the results
for i, event in enumerate(event_list):
    #read in data
    Dfile = Dpath.format(event, reg)
    # Hfile = Hpath.format(event, reg)
    # Zfile = Zpath.format(event, reg) orginal
    Zfile = Zpath.format(reg)
    dZfile = dZpath.format(event, reg)
    TSfile = TSpath.format(event)

    D = xr.open_dataset(Dfile).z
    # H = xr.open_dataset(Hfile).z
    Z = xr.open_dataset(Zfile).z #land mask
    dZ = xr.open_dataset(dZfile).deformation
    TS = xr.open_dataset(TSfile).eta

    #extract data
    d_land = D.where(Z.values > 0).values #max flow depth on land
    dZ_land = dZ.where(Z.values > 0).values #max deformation on land
    t = TS[1:,GaugeNo].values #time series of gauge nos

    d_land[np.isnan(d_land)] = 0
    dZ_land[np.isnan(dZ_land)] = 0
    t[np.isnan(t)] = 0    

    #store data
    d_array[i, :, :] = d_land
    dZ_array[i, :, :] = dZ_land
    t_array[i, :, :] = t

    #close files
    D.close()
    # H.close()
    # Z.close()
    dZ.close()
    TS.close()

#arrange dimensions as prefered by pytorch
#(no of events ie batch size, (9) channels, 480 timesteps)
t_array = np.transpose(t_array, (0,2,1))

if mode == 'train':
    # find elements in d_array which are always zero across all the events
    # zero_indices = np.where(np.all(d_array == 0, axis=0))
    # zero_mask = np.all(d_array == 0, axis=0) #non flooded
    # print(f'Number of zero count: {np.count_nonzero(zero_mask)}')
    # non_zero_mask = ~zero_mask #flooded 
    # print(f'Number of non zero count: {np.count_nonzero(non_zero_mask)}')
    # #savemask
    # np.save(f'{MLDir}/data/processed/zero_mask_{reg}_{size}.npy', zero_mask)
    #load mask instead of calculating it
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    non_zero_mask = ~zero_mask
    print(f'Number of zero count: {np.count_nonzero(zero_mask)}')
    print(f'Number of non zero count: {np.count_nonzero(non_zero_mask)}')
elif mode == 'test':
    #load mask
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    non_zero_mask = ~zero_mask
    print(f'Number of zero count: {np.count_nonzero(zero_mask)}')
    print(f'Number of non zero count: {np.count_nonzero(non_zero_mask)}')

#remove elements which are always zero across all the events ir reduced d_array
red_d_array = d_array[:, ~zero_mask] #note: its a 2d array: events X (948*1300) ie events X 70k/123k locations
red_dZ_array = dZ_array[:, ~zero_mask] #note: its a 2d array: events X (948*1300) ie events X 70k/123k locations

#save as numpy binary files
onshore_map = np.memmap(f'{MLDir}/data/processed/d_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(d_array.shape[0], d_array.shape[1], d_array.shape[2]))
onshore_map[:] = d_array[:]

dZonshore_map = np.memmap(f'{MLDir}/data/processed/dZ_{reg}_{size}.dat',
                            mode='w+',
                            dtype=float,
                            shape=(dZ_array.shape[0], dZ_array.shape[1], dZ_array.shape[2]))
dZonshore_map[:] = dZ_array[:]

onshore_map2 = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(red_d_array.shape[0], red_d_array.shape[1]))
onshore_map2[:] = red_d_array[:]

dZonshore_map2 = np.memmap(f'{MLDir}/data/processed/dZflat_{reg}_{size}.dat',
                            mode='w+',  
                            dtype=float,
                            shape=(red_dZ_array.shape[0], red_dZ_array.shape[1]))
dZonshore_map2[:] = red_dZ_array[:]

offshore_map = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(t_array.shape[0], t_array.shape[1], t_array.shape[2]))
offshore_map[:] = t_array[:]
