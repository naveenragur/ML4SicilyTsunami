#Description: Preprocess the data offshore and onshore data for training and testing
import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import torch
import pandas as pd

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    task = sys.argv[2] #reprocess or post
    train_size = sys.argv[3] #eventset size used for training
    mask_size = sys.argv[4] #eventset size used for testing
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#dimensions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    columnname = str(54)
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #gauges time series
    pts_dim = 480 #time steps
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    columnname = str(38)
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo)
    pts_dim = 480


#check if PTHA directory exists in task folder
if not os.path.exists(f'{MLDir}/model/{reg}/{task}/PTHA'):
    os.makedirs(f'{MLDir}/model/{reg}/{task}/PTHA')

#load data
flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
nflood_grids = np.count_nonzero(flood_mask)
zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
index_map.columns = ['m','n','lat','lon'] #add column names

#save direct depths - mean predictiona and sigma
print('Processing direct depths')
for test_size in ['0','1','2','3']:
    print(f'Processing test size {test_size}')
    pred_d_array = np.load(f'{MLDir}/model/{reg}/{task}/out/pred_trainsize{train_size}_testsize{test_size}_direct.npy')
    sigma_d_array = np.load(f'{MLDir}/model/{reg}/{task}/out/residual_trainsize{train_size}_testsize{test_size}_direct.npy')
    tensor = torch.tensor(pred_d_array,dtype=torch.float16)
    tensor_sigma = torch.tensor(sigma_d_array,dtype=torch.float16)
    
    #append events along first axis
    if test_size == '0':
        d = tensor
        sigma_d = tensor_sigma
    else:
        d = torch.cat((d,tensor),0)
        sigma_d = torch.cat((sigma_d,tensor_sigma),0)
d = d*100
d = d.type(torch.int16)
sigma_d = sigma_d*100
sigma_d = sigma_d.type(torch.int16)

sigma_minus = d - (2*sigma_d)
sigma_plus = d + (2*sigma_d)
#save 
np.save(f'{MLDir}/model/{reg}/{task}/PTHA/pred_d_{train_size}_direct.npy',d)
np.save(f'{MLDir}/model/{reg}/{task}/PTHA/sigma_d_{train_size}_direct.npy',sigma_d)
np.save(f'{MLDir}/model/{reg}/{task}/PTHA/sigma_minus_{train_size}_direct.npy',sigma_minus)
np.save(f'{MLDir}/model/{reg}/{task}/PTHA/sigma_plus_{train_size}_direct.npy',sigma_plus)
del d,tensor,pred_d_array

