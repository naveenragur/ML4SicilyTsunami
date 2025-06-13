#Description: Calculates the PTHA hazard curves at all prediction points for maps and subsets of events
#Usage: python 08_calcPTHA.py CT  <mode> <train/test> <mask size> modes are: PTHA_rate/emulator_rate/emulator_sigma
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

def single_exceedance_depth(thresholds, eve_depths, eve_rates): #given threshold of return period as rates, predicts depth
    lambda_exc = np.zeros((len(thresholds))) 
    #sort events and rates by depth
    ix = np.argsort(eve_depths)
    eve_depths = eve_depths[ix]
    eve_rates = eve_rates[ix]
    #calculate cdf from highest to lowest depth
    eve_cdf = np.cumsum(eve_rates[::-1])
    for threshold in range(len(thresholds)):
        if eve_depths.sum() == 0:
            lambda_exc[threshold] = 0
        exceed_indices = np.where(eve_cdf > thresholds[threshold])[0]
        if len(exceed_indices) == 0:
            lambda_exc[threshold] = 0
        else:
            lambda_exc[threshold] = eve_depths[-exceed_indices[0]] #negative index to get the highest depth
    return lambda_exc #give the depth of exceedance for each thresholf retrun period

def single_exceedance_rate(thresholds, eve_depths, eve_rates): #given threshold of depth, predicts rate
    # Annual Mean Rate of threshold exceedance
    lambda_exc = np.zeros(len(thresholds))
    for threshold in range(len(thresholds)):
        ix = np.array(np.where(eve_depths > thresholds[threshold])).squeeze() #events where depth exceeds threshold
        if ix is None:
            lambda_exc[threshold] = 0 #no events exceed threshold
        else:
            lambda_exc[threshold] = eve_rates[ix].sum(axis=0) #sum of rates of events that exceed threshold
    return lambda_exc #give the rate of exceedance for each threshold

if mode == 'PTHA_rate':
    #check if PTHA directory exists
    if not os.path.exists(f'{MLDir}/model/{reg}/multifoldMC/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/multifoldMC/PTHA')
        
    RP_list = [1/100,1/1000,1/10000,1/100000,1/1000000]
    Depth_list = [20,100,300,500,1000]
    
    #load data
    eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/multifoldMC/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    eve_rate = pd.read_csv(f'{MLDir}/resources/processed/all_eventsBS_PS53550_perce_rates.txt')
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names
    
    PTHA_table_true = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_true_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_true_nodef = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_true = np.zeros((nflood_grids,len(RP_list)))
    PTHA_depth_true_def = np.zeros((nflood_grids,len(RP_list)))
    PTHA_depth_true_nodef = np.zeros((nflood_grids,len(RP_list)))

    eve_def = eve_perf['max_absdz']>= 0.1
    eve_nodef = eve_perf['max_absdz']< 0.1
    true_d = np.load(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_d_53550.npy')
    # rate = eve_perf['mean_prob'].to_numpy()
    rate = eve_rate['P84'].to_numpy()
    
    for select_pt in range(nflood_grids): #nflood_grids
        #compute exceedance rate for true
        if select_pt % 50000 == 0:
            print(f'Processing point {select_pt}')
        PTHA_table_true[select_pt,:] = single_exceedance_rate(Depth_list, true_d[:,select_pt], rate)
        PTHA_depth_true[select_pt,:] = single_exceedance_depth(RP_list, true_d[:,select_pt], rate)
        PTHA_table_true_def[select_pt,:] = single_exceedance_rate(Depth_list, true_d[eve_def,select_pt], rate[eve_def])
        PTHA_depth_true_def[select_pt,:] = single_exceedance_depth(RP_list, true_d[eve_def,select_pt], rate[eve_def])
        PTHA_table_true_nodef[select_pt,:] = single_exceedance_rate(Depth_list, true_d[eve_nodef,select_pt], rate[eve_nodef])
        PTHA_depth_true_nodef[select_pt,:] = single_exceedance_depth(RP_list, true_d[eve_nodef,select_pt], rate[eve_nodef])
       
    # save PTHA tables
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHArate_53550.npy',PTHA_table_true)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHAdepth_53550.npy',PTHA_depth_true)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHArate_def_53550.npy',PTHA_table_true_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHAdepth_def_53550.npy',PTHA_depth_true_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHArate_nodef_53550.npy',PTHA_table_true_nodef)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/true_PTHAdepth_nodef_53550.npy',PTHA_depth_true_nodef)

elif mode == 'emulator_rate':
    #check if PTHA directory exists
    if not os.path.exists(f'{MLDir}/model/{reg}/multifoldMC/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/multifoldMC/PTHA')

    # RP_list = [1/100,1/1000,1/10000,1/100000,1/1000000]
    # Depth_list = [20,100,300,500,1000]
    # RP_list = [1/2500, 1/10000, 1/100000]
    Depth_list = [20]
    
    #load data
    eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/multifoldMC/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names
    
    pred_d = np.load(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_d_{train_size}_direct.npy')
    pred_d[pred_d<0] = 0
    PTHA_table_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))
    
    eve_def = eve_perf['max_absdz']>= 0.1
    eve_nodef = eve_perf['max_absdz']< 0.1
    rate = eve_perf['mean_prob'].to_numpy()

    for select_pt in range(nflood_grids): #nflood_grids
        #compute exceedance rate for true
        if select_pt % 50000 == 0:
            print(f'Processing point {select_pt}')
        PTHA_table_pred[select_pt,:] = single_exceedance_rate(Depth_list, pred_d[:,select_pt], rate)
        # PTHA_depth_pred[select_pt,:] = single_exceedance_depth(RP_list, pred_d[:,select_pt], rate)
        # PTHA_table_pred_def[select_pt,:] = single_exceedance_rate(Depth_list, pred_d[eve_def,select_pt], rate[eve_def])
        # PTHA_depth_pred_def[select_pt,:] = single_exceedance_depth(RP_list, pred_d[eve_def,select_pt], rate[eve_def])
        # PTHA_table_pred_nodef[select_pt,:] = single_exceedance_rate(Depth_list, pred_d[eve_nodef,select_pt], rate[eve_nodef])
        # PTHA_depth_pred_nodef[select_pt,:] = single_exceedance_depth(RP_list, pred_d[eve_nodef,select_pt], rate[eve_nodef])
       
    # save PTHA tables
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHArate_{train_size}.npy',PTHA_table_pred)
    # np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHAdepth_{train_size}.npy',PTHA_depth_pred)
    # np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHArate_def_{train_size}.npy',PTHA_table_pred_def)
    # np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHAdepth_def_{train_size}.npy',PTHA_depth_pred_def)
    # np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHArate_nodef_{train_size}.npy',PTHA_table_pred_nodef)
    # np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_PTHAdepth_nodef_{train_size}.npy',PTHA_depth_pred_nodef)

elif mode == 'emulator_sigma':
    #check if PTHA directory exists
    if not os.path.exists(f'{MLDir}/model/{reg}/multifoldMC/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/multifoldMC/PTHA')

    RP_list = [1/100,1/1000,1/10000,1/100000,1/1000000]
    Depth_list = [20,100,300,500,1000]
    
    #load data
    eve_perf = pd.read_csv(f'{MLDir}/model/{reg}/multifoldMC/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt')
    index_map.columns = ['m','n','lat','lon'] #add column names
    
    pred_d = np.load(f'{MLDir}/model/{reg}/multifoldMC/PTHA/pred_d_{train_size}_direct.npy')
    sigma_d = np.load(f'{MLDir}/model/{reg}/multifoldMC/PTHA/sigma_d_{train_size}_direct.npy')
    minus_sigma_d = pred_d - sigma_d
    plus_sigma_d = pred_d + sigma_d
    
    pred_d[pred_d<0] = 0
    minus_sigma_d[pred_d<0] = 0
    plus_sigma_d[pred_d<0] = 0

    PTHA_table_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))
    
    eve_def = eve_perf['max_absdz']>= 0.1
    eve_nodef = eve_perf['max_absdz']< 0.1
    rate = eve_perf['mean_prob'].to_numpy()

    #minus sigma
    for select_pt in range(nflood_grids): #nflood_grids
        #compute exceedance rate for true
        if select_pt % 50000 == 0:
            print(f'Processing point {select_pt}')
        PTHA_table_pred[select_pt,:] = single_exceedance_rate(Depth_list, minus_sigma_d[:,select_pt], rate)
        PTHA_depth_pred[select_pt,:] = single_exceedance_depth(RP_list, minus_sigma_d[:,select_pt], rate)
        PTHA_table_pred_def[select_pt,:] = single_exceedance_rate(Depth_list, minus_sigma_d[eve_def,select_pt], rate[eve_def])
        PTHA_depth_pred_def[select_pt,:] = single_exceedance_depth(RP_list, minus_sigma_d[eve_def,select_pt], rate[eve_def])
        PTHA_table_pred_nodef[select_pt,:] = single_exceedance_rate(Depth_list, minus_sigma_d[eve_nodef,select_pt], rate[eve_nodef])
        PTHA_depth_pred_nodef[select_pt,:] = single_exceedance_depth(RP_list, minus_sigma_d[eve_nodef,select_pt], rate[eve_nodef])
       
    # save PTHA tables
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHArate_{train_size}.npy',PTHA_table_pred)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHAdepth_{train_size}.npy',PTHA_depth_pred)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHArate_def_{train_size}.npy',PTHA_table_pred_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHAdepth_def_{train_size}.npy',PTHA_depth_pred_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHArate_nodef_{train_size}.npy',PTHA_table_pred_nodef)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/minussigma_PTHAdepth_nodef_{train_size}.npy',PTHA_depth_pred_nodef)

    #plus sigma
    PTHA_table_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_pred_nodef = np.zeros((nflood_grids,len(Depth_list)))

    for select_pt in range(nflood_grids): #nflood_grids
        #compute exceedance rate for true
        if select_pt % 50000 == 0:
            print(f'Processing point {select_pt}')
        PTHA_table_pred[select_pt,:] = single_exceedance_rate(Depth_list, plus_sigma_d[:,select_pt], rate)
        PTHA_depth_pred[select_pt,:] = single_exceedance_depth(RP_list, plus_sigma_d[:,select_pt], rate)
        PTHA_table_pred_def[select_pt,:] = single_exceedance_rate(Depth_list, plus_sigma_d[eve_def,select_pt], rate[eve_def])
        PTHA_depth_pred_def[select_pt,:] = single_exceedance_depth(RP_list, plus_sigma_d[eve_def,select_pt], rate[eve_def])
        PTHA_table_pred_nodef[select_pt,:] = single_exceedance_rate(Depth_list, plus_sigma_d[eve_nodef,select_pt], rate[eve_nodef])
        PTHA_depth_pred_nodef[select_pt,:] = single_exceedance_depth(RP_list, plus_sigma_d[eve_nodef,select_pt], rate[eve_nodef])
       
    # save PTHA tables
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHArate_{train_size}.npy',PTHA_table_pred)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHAdepth_{train_size}.npy',PTHA_depth_pred)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHArate_def_{train_size}.npy',PTHA_table_pred_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHAdepth_def_{train_size}.npy',PTHA_depth_pred_def)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHArate_nodef_{train_size}.npy',PTHA_table_pred_nodef)
    np.save(f'{MLDir}/model/{reg}/multifoldMC/PTHA/plussigma_PTHAdepth_nodef_{train_size}.npy',PTHA_depth_pred_nodef)
    



else:
    print('Error: Invalid mode')

