#Description: Preprocess the data offshore and onshore data for training and testing
import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
import contextily as cx
import torch
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    mode = sys.argv[2] #reprocess or post
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

def Gfit_one(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    return Gvalue

def ptha_err(obs, pred, rate): 
    obs = np.array(obs)
    pred = np.array(pred)
    rate = np.array(rate)
    ptha_err = np.sum(rate*(obs-pred))
    return ptha_err                    

def calc_scores(true,pred):
    mse_val = mean_squared_error(true,pred)
    r2_val = r2_score(true,pred)
    Gfit_val = Gfit_one(true,pred)
    return mse_val,r2_val,Gfit_val

if mode == 'Grid':
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    #load data
    eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/results/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    eve_dep = pd.read_csv(f'{MLDir}/model/{reg}/results/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_true_pred_er_combined.csv')
    eve_SIS = pd.read_csv(f'{MLDir}/data/events/Sample_events_CT_2000_AA.txt') 
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names

    #calculate eve_id column for eve_dep
    eve_dep['eve_id'] = eve_dep['id'].apply(lambda x: x.split('/')[1])

    #find index for rows in eve_dep matching eve_SIS using eve_id column as key
    eve_SIS.columns = ['eve_id']
    sample_idx = eve_dep['eve_id'].isin(eve_SIS['eve_id'])
    rate = eve_perf['mean_prob']

    #load true and pred depth tables for a train_size
    #n_eve x nflood_grids
    pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy')
    true_d = np.load(f'{MLDir}/model/{reg}/PTHA/true_d_53550.npy')

    #pick event index which are test events from eve_perf
    eve_test = eve_perf['split']== 'test'
    pred_d = pred_d[eve_test]
    true_d_test = true_d[eve_test]
    rate = rate[eve_test]

    eve_train = eve_perf['split']== 'train'
    true_d_train = true_d[eve_train]

    #check if the number match
    print(pred_d.shape)

    #calculate R2, G , MSE and PTHA error(MSE wt by mean prob) for each event at each grid point
    table = np.zeros((pred_d.shape[1],6)) #mse,r2,Gfit,ptha_err,count

    for i in range(pred_d.shape[1]):
        if i%1000==0:
            print(i)
        true = true_d_test[:,i] #per location all events
        pred = pred_d[:,i]
        true_train = true_d_train[:,i]
        mse_val,r2_val,Gfit_val = calc_scores(true,pred) #mse_val,r2_val,Gfit_val
        ptha_err_val = ptha_err(true,pred,rate)

        table[i,0] = mse_val
        table[i,1] = r2_val
        table[i,2] = Gfit_val
        table[i,3] = ptha_err_val
        true[true<10] = 0
        table[i,4] = np.count_nonzero(true)/true.shape[0] #count greater than 10cm (events x grids)
        table[i,5] = np.count_nonzero(true_train)/true_train.shape[0]

    #save the table
    np.save(f'{MLDir}/model/{reg}/PTHA/Grid_scores_{train_size}.npy',table)
    
    #add header to the table
    header = ['mse','r2','Gfit','ptha_err','count_test','count_train']
    table = pd.DataFrame(table,columns=header)
    table.to_csv(f'{MLDir}/model/{reg}/PTHA/Grid_scores_{train_size}.txt',index=False)