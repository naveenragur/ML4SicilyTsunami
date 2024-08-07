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

def calc_scores(true,pred,rate=None,threshold=20):
    #preprocess
    # idx = true>threshold
    # true = true[idx] #remove values below threshold
    # pred = pred[idx] #small inundation events
    pred[pred < 0] = 0 #sometime the model predicts negative values instead of zero
    #apply rate
    if rate is not None:
        # rate = rate[idx]
        pred = pred*rate
        true = true*rate
    #calculate scores
    if np.count_nonzero(true) <= 2:
        return np.nan,np.nan,np.nan
    else:    
        mse_val = mean_squared_error(true,pred)
        r2_val = r2_score(true,pred)
        Gfit_val = Gfit_one(true,pred)
        return mse_val,r2_val,Gfit_val

def ptha_err(true,pred,rate,threshold=20): 
    # true = true[true>threshold] #remove values below threshold
    # pred = pred[true>threshold] #small inundation events
    # rate = rate[true>threshold]
    pred[pred < 0] = 0 #sometime the model predicts negative values when close to zero
    if np.count_nonzero(true) == 0:
        return np.nan
    else:
        ptha_err = np.sum(rate*(true-pred))
        return ptha_err  

if mode == 'Grid':
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    #load data
    if reg == 'CT':
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
        eve_dep = pd.read_csv(f'{MLDir}/model/{reg}/out/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_true_pred_er_combined.csv')
    else:
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
        eve_dep = pd.read_csv(f'{MLDir}/model/{reg}/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_true_pred_er_combined.csv')

    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names
    rate = eve_perf['mean_prob']

    #calculate eve_id column for eve_dep file
    eve_dep['eve_id'] = eve_dep['id'].apply(lambda x: x.split('/')[1])

    # #filter a selection like from importance sampling
    # eve_SIS = pd.read_csv(f'{MLDir}/data/events/Sample_events_CT_2000_AA.txt') 
    # eve_SIS.columns = ['eve_id']
    # sample_idx = eve_dep['eve_id'].isin(eve_SIS['eve_id'])

    #n_eve x nflood_grids
    true_d = np.load(f'{MLDir}/model/{reg}/PTHA/true_d_53550.npy').astype(np.int32)
    if reg == 'CT':
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}_direct.npy').astype(np.int32)
    else:
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy').astype(np.int32)
    
    #pick event index which are test events from eve_perf 
    # train-test split
    eve_test = eve_perf['split']== 'test'
    pred_d_test = pred_d[eve_test]
    true_d_test = true_d[eve_test]
    rate = rate[eve_test]

    eve_train = eve_perf['split']== 'train'
    pred_d_train = pred_d[eve_train]
    true_d_train = true_d[eve_train]

    #test and additonal split
    eve_BS = (eve_perf['SR']== 'BS') & (eve_perf['split']== 'test')
    pred_d_BS = pred_d[eve_BS]
    true_d_BS = true_d[eve_BS]

    eve_PS = (eve_perf['SR']!= 'BS') & (eve_perf['split']== 'test')
    pred_d_PS = pred_d[eve_PS]
    true_d_PS = true_d[eve_PS]

    eve_def = (eve_perf['max_absdz']>= 0.1) & (eve_perf['split']== 'test')
    pred_d_def = pred_d[eve_def]
    true_d_def = true_d[eve_def]

    eve_nodef = (eve_perf['max_absdz']< 0.1) & (eve_perf['split']== 'test')
    pred_d_nodef = pred_d[eve_nodef]
    true_d_nodef = true_d[eve_nodef]

    #check if the number match
    print(pred_d.shape)

    #calculate R2, G , MSE and PTHA error(MSE wt by mean prob) for each event at each grid point
    table = np.zeros((pred_d.shape[1],22)) #mse,r2,Gfit,ptha_err,count_test,count_train, repeat mse,r2G for BS,PS, Deformation, No deformation

    for i in range(pred_d.shape[1]): #for each grid point
        if i%1000==0:
            print(i)
        #segregrate for each event classifcation
        true_test = true_d_test[:,i] #per location all events
        true_train = true_d_train[:,i]
        true_BS = true_d_BS[:,i]
        true_PS = true_d_PS[:,i]
        true_def = true_d_def[:,i]
        true_nodef = true_d_nodef[:,i]

        pred_test = pred_d_test[:,i]
        pred_train = pred_d_train[:,i]
        pred_BS = pred_d_BS[:,i]
        pred_PS = pred_d_PS[:,i]
        pred_def = pred_d_def[:,i]
        pred_nodef = pred_d_nodef[:,i]

        #calculate scores
        mse_val,r2_val,Gfit_val = calc_scores(true_test,pred_test) 
        ptha_err_val = ptha_err(true_test,pred_test,rate)
        table[i,0] = mse_val
        table[i,1] = r2_val
        table[i,2] = Gfit_val
        table[i,3] = ptha_err_val
        table[i,4] = np.count_nonzero(true_test) 
        table[i,5] = np.count_nonzero(true_train)

        mse_val,r2_val,Gfit_val = calc_scores(true_BS,pred_BS) 
        table[i,6] = mse_val
        table[i,7] = r2_val
        table[i,8] = Gfit_val
        table[i,9] = np.count_nonzero(true_BS)

        mse_val,r2_val,Gfit_val = calc_scores(true_PS,pred_PS) 
        table[i,10] = mse_val
        table[i,11] = r2_val
        table[i,12] = Gfit_val
        table[i,13] = np.count_nonzero(true_PS)

        mse_val,r2_val,Gfit_val = calc_scores(true_def,pred_def)
        table[i,14] = mse_val
        table[i,15] = r2_val
        table[i,16] = Gfit_val
        table[i,17] = np.count_nonzero(true_def)

        mse_val,r2_val,Gfit_val = calc_scores(true_nodef,pred_nodef) 
        table[i,18] = mse_val
        table[i,19] = r2_val
        table[i,20] = Gfit_val
        table[i,21] = np.count_nonzero(true_nodef)

    #save the table
    # np.save(f'{MLDir}/model/{reg}/PTHA/Grid_scores_{train_size}.npy',table)
    
    #add header to the table
    header = [
              'mse','r2','Gfit','ptha_err',
              'count_test','count_train',
              'mse_BS','r2_BS','Gfit_BS','count_test_BS',
              'mse_PS','r2_PS','Gfit_PS','count_test_PS',
              'mse_def','r2_def','Gfit_def','count_test_def',
              'mse_nodef','r2_nodef','Gfit_nodef','count_test_nodef',
              ]
    
    table = pd.DataFrame(table,columns=header)
    table['m'] = index_map['m']
    table['n'] = index_map['n']
    table['lat'] = index_map['lat']
    table['lon'] = index_map['lon']
    table.to_csv(f'{MLDir}/model/{reg}/PTHA/_Grid_scores_{train_size}.txt',index=False)

elif mode== 'testfile':
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    #load data
    if reg == 'CT':
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    else:    
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    eve_filter = pd.read_csv(f'{MLDir}/data/events/shuffled_events_CT_PSHalf.txt') 
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names
    rate = eve_perf['mean_prob']
    max_off = eve_perf['max_off']

    #n_eve x nflood_grids
    true_d = np.load(f'{MLDir}/model/{reg}/PTHA/true_d_53550.npy').astype(np.int32)
    if reg == 'CT':
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}_direct.npy').astype(np.int32)
    else:
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy').astype(np.int32)
    
    # train-test-PS split
    #pick events which are from eve_filter and mark as PS
    eve_filter.columns = ['id']
    eve_perf.loc[eve_perf['id'].isin(eve_filter['id']), 'split'] = 'PSHalf'
    #save this table to a file
    eve_perf.to_csv(f'{MLDir}/model/{reg}/PTHA/_eve_perf_{train_size}_PSHalf.txt',index=False)
        
    #train
    eve_train = eve_perf['split']== 'train'
    pred_d_train = pred_d[eve_train]
    true_d_train = true_d[eve_train]
    #test
    eve_test = eve_perf['split']!= 'train'
    pred_d_test = pred_d[eve_test]
    true_d_test = true_d[eve_test]
    rate = rate[eve_test]
    max_off = max_off[eve_test]
    #PS
    eve_PS = eve_perf['split']== 'PSHalf'
    pred_d_PS = pred_d[eve_PS]
    true_d_PS = true_d[eve_PS]

    #check if the number match
    print(pred_d.shape)

    #calculate R2, G , MSE and PTHA error(MSE wt by mean prob) for each event at each grid point
    table = np.zeros((pred_d.shape[1],12)) #mse,r2,Gfit,count_test,count_train
    for i in range(pred_d.shape[1]): #for each grid point
        if i%1000==0:
            print(i)
        #segregrate for each event classifcation
        true_test = true_d_test[:,i] #per location all events
        true_train = true_d_train[:,i]
        true_PS = true_d_PS[:,i]

        pred_test = pred_d_test[:,i]
        pred_train = pred_d_train[:,i]
        pred_PS = pred_d_PS[:,i]

        #calculate scores
        mse_val,r2_val,Gfit_val = calc_scores(true_test,pred_test,rate) 
        table[i,0] = mse_val
        table[i,1] = r2_val
        table[i,2] = Gfit_val

        mse_val,r2_val,Gfit_val = calc_scores(true_test,pred_test,rate*max_off*100) 
        table[i,3] = mse_val
        table[i,4] = r2_val
        table[i,5] = Gfit_val

        mse_val,r2_val,Gfit_val = calc_scores(true_PS,pred_PS) 
        table[i,6] = mse_val
        table[i,7] = r2_val
        table[i,8] = Gfit_val

        table[i,9] = np.count_nonzero(true_test) 
        table[i,10] = np.count_nonzero(true_PS)
        table[i,11] = np.count_nonzero(true_train)

    #add header to the table
    header = [
              'mse_test_wt','r2_test_wt','Gfit_test_wt',
              'mse_test_imp','r2_test_imp','Gfit_test_imp',
              'mse_PS','r2_PS','Gfit_PS',
              'count_test','count_PS','count_train',
              ]
    
    table = pd.DataFrame(table,columns=header)
    #append m,n,lat,lon to table
    table['m'] = index_map['m']
    table['n'] = index_map['n']
    table['lat'] = index_map['lat']
    table['lon'] = index_map['lon']
    table.to_csv(f'{MLDir}/model/{reg}/PTHA/_Grid_scores_{train_size}_PSHalf.txt',index=False)

elif mode== 'misfit':
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    #load data
    if reg == 'CT':
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    else:    
        eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/out/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names

    #n_eve x nflood_grids
    true_d = np.load(f'{MLDir}/model/{reg}/PTHA/true_d_53550.npy').astype(np.int32)
    if reg == 'CT':
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}_direct.npy').astype(np.int32)
    else:
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy').astype(np.int32)
          
    #test
    eve_test = eve_perf['split']!= 'train'
    pred_d_test = pred_d[eve_test]
    true_d_test = true_d[eve_test]

    #misfit
    err_test = np.abs(pred_d_test - true_d_test)
    
    #convert to integer16
    err_test = err_test.astype(np.int16) #neve x nflood_grids
    pred_d_test = pred_d_test.astype(np.int16) #neve x nflood_grids
    true_d_test = true_d_test.astype(np.int16) #neve x nflood_grids
    np.save(f'{MLDir}/model/{reg}/PTHA/_misfit_{train_size}.npy',err_test)
    #stack the 3 variable in the last dimension
    table_test = np.stack((pred_d_test,true_d_test,err_test),axis=-1) #neve x nflood_grids x 3
    
    np.save(f'{MLDir}/model/{reg}/PTHA/_TPE_table_{train_size}.npy',table_test)


    
