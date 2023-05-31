import os
import sys
# import glob

import numpy as np
import xarray as xr


try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

# file_list = glob.glob('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/PS_manning003/**/*SR_defbathy.nc')
# for file in file_list:
#     print(file)
#     # print(os.path.join(file.rsplit('/',1)[-2],'SR_defbathy.nc'))
#     os.rename(file,os.path.join(file.rsplit('/',1)[-2],'SR_defbathy.nc')) 


#read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', dtype='str')
#not size is the size of selection but not the number of events in the actual event list which can be less 
#than the size of selection if there are events that dont satisfy threshold criterias

#string template for file filepath
Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc'
Hpath = SimDir + '/{:s}/{:s}_hmax.nc'
Zpath = SimDir + '/{:s}/{:s}_defbathy.nc'
dZpath = SimDir + '/{:s}/{:s}__deformation.nc'
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
t_array = np.zeros((len(event_list), pts_dim, ts_dim))  #say for SR: no of events x 480(time steps) x 5(gauges)

#loop over events 
#TODO: if needed parallelize? the event list are split into chunks,process each chunk in parallel and merge the results
for i, event in enumerate(event_list):
    #read in data
    Dfile = Dpath.format(event, reg)
    # Hfile = Hpath.format(event, reg)
    Zfile = Zpath.format(event, reg)
    # dZfile = dZpath.format(event, reg)
    TSfile = TSpath.format(event)

    D = xr.open_dataset(Dfile).z
    # H = xr.open_dataset(Hfile).z
    Z = xr.open_dataset(Zfile).z #land mask
    # dZ = xr.open_dataset(dZfile).deformation
    TS = xr.open_dataset(TSfile).eta

    #extract data
    d_land = D.where(Z.values > 0) #max flow depth on land
    t = TS[1:,GaugeNo].values #time series of gauge nos

    #store data
    d_array[i, :, :] = d_land
    t_array[i, :, :] = t

    #close files
    D.close()
    # H.close()
    # Z.close()
    # dZ.close()
    TS.close()

#arrange dimensions as prefered by pytorch
#(no of events ie batch size, (9) channels, 480 timesteps)
t_array = np.transpose(t_array, (0,2,1))

# find elements in d_array which are always zero across all the events
zero_indices = np.where(np.all(d_array == 0, axis=0))
zero_mask = np.all(d_array == 0, axis=0) #non flooded
non_zero_mask = ~zero_mask #flooded 

#savemask
np.save(f'{MLDir}/data/processed/zero_mask_{reg}_{size}.npy', zero_mask)

#remove elements which are always zero across all the events ir reduced d_array
red_d_array = d_array[:, ~zero_mask] #note: its a 2d array: events X (948*1300) ie events X 70k/123k locations

#save as numpy binary files
onshore_map = np.memmap(f'{MLDir}/data/processed/d_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(d_array.shape[0], d_array.shape[1], d_array.shape[2]))
onshore_map[:] = d_array[:]

onshore_map2 = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(red_d_array.shape[0], red_d_array.shape[1]))
onshore_map2[:] = red_d_array[:]

offshore_map = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
                         mode='w+',
                         dtype=float,
                         shape=(t_array.shape[0], t_array.shape[1], t_array.shape[2]))
offshore_map[:] = t_array[:]
