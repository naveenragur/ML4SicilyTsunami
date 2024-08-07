#Description: Preprocess the data offshore and onshore data for training and testing
import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import label
from matplotlib.colors import ListedColormap

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    mode = sys.argv[2] #reprocess or post
    train_size = sys.argv[3] #eventset size used for training
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

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

if mode == 'reprocess':
    for test_size in ['0','1','2','3']:
        #read postprocessed data
        empty_model_out = np.load(f'{MLDir}/model/{reg}/out/postprocessed_trainsize{train_size}_testsize{test_size}.npy')

        #save as memap file like reduced onshore depths in preprocessing
        onshore_map = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{test_size}_{train_size}_prediction.dat',
                                mode='w+',
                                dtype=float,
                                shape=(empty_model_out.shape[0], empty_model_out.shape[1]))

        onshore_map[:] = empty_model_out[:]

        print(f'Onshore data for test size {test_size} saved with events {empty_model_out.shape[0]} and locations {empty_model_out.shape[1]}')
elif mode == 'reprocess_direct':
    for test_size in ['0','1','2','3']:
        #read postprocessed data
        empty_model_out = np.load(f'{MLDir}/model/{reg}/out/postprocessed_trainsize{train_size}_testsize{test_size}.npy')

        #save as memap file like reduced onshore depths in preprocessing
        onshore_map = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{test_size}_{train_size}_prediction.dat',
                                mode='w+',
                                dtype=float,
                                shape=(empty_model_out.shape[0], empty_model_out.shape[1]))

        onshore_map[:] = empty_model_out[:]

        print(f'Onshore data for test size {test_size} saved with events {empty_model_out.shape[0]} and locations {empty_model_out.shape[1]}')
else:
    print('Error: Invalid mode')




    

