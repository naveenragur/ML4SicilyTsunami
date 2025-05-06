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

# def exceedance_curve(thresholds, max_stage_point, rates): 
#    # Annual Mean Rate of threshold exceedance 
#    lambda_exc = np.zeros((len(thresholds), 1000)) 
#    for threshold in range(len(thresholds)): 
#        ix = np.array(np.where(max_stage_point > thresholds[threshold])).squeeze() 
#        for j in range(1000): 
#            #for index in ix: 
#            lambda_exc[threshold, j] = lambda_exc[threshold, j] +  rates[ix,j].sum(axis=0) 
#    #------------------------------------------------ 
#    return lambda_exc 
# def make_map(RP_table_true,RP_table_pred,RP_table_sis,RPi):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     for i, ax in enumerate(axs):
#         if i == 0:
#             RP_table = RP_table_true
#             title = 'True'
#         elif i == 1:
#             RP_table = RP_table_pred
#             title = 'Pred'
#         elif i == 2:
#             RP_table = RP_table_sis
#             title = 'SIS'
#         array_2d = np.zeros((y_dim,x_dim))
#         array_2d[~zero_mask] = RP_table[:,RPi]
#         im = ax.imshow(array_2d, cmap='viridis',vmin=0, vmax=200)
#         ax.invert_yaxis()
#         ax.set_title(title)
#         ax.axis('off')
#     fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal',label='Depth (cm)')
#     plt.suptitle(f'Return period depths at {RP_list[RPi]}')
#     plt.savefig(f'{MLDir}/model/{reg}/PTHA/RP{RPi+1}_map_{train_size}.png')
#     plt.close()

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
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    RP_list = [1/100,1/1000,1/10000,1/100000,1/1000000]
    Depth_list = [20,100,300,500,1000]
    
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
    
    PTHA_table_true = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_true_def = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_table_true_nodef = np.zeros((nflood_grids,len(Depth_list)))
    PTHA_depth_true = np.zeros((nflood_grids,len(RP_list)))
    PTHA_depth_true_def = np.zeros((nflood_grids,len(RP_list)))
    PTHA_depth_true_nodef = np.zeros((nflood_grids,len(RP_list)))

    eve_def = eve_perf['max_absdz']>= 0.1
    eve_nodef = eve_perf['max_absdz']< 0.1
    true_d = np.load(f'{MLDir}/model/{reg}/PTHA/true_d_53550.npy')
    rate = eve_perf['mean_prob'].to_numpy()
    
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
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHArate_53550.npy',PTHA_table_true)
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHAdepth_53550.npy',PTHA_depth_true)
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHArate_def_53550.npy',PTHA_table_true_def)
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHAdepth_def_53550.npy',PTHA_depth_true_def)
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHArate_nodef_53550.npy',PTHA_table_true_nodef)
    np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHAdepth_nodef_53550.npy',PTHA_depth_true_nodef)

elif mode == 'emulator_rate':
    #check if PTHA directory exists
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    RP_list = [1/100,1/1000,1/10000,1/100000,1/1000000]
    Depth_list = [20,100,300,500,1000]
    
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
    
    if reg == 'CT':
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}_direct.npy')
    else:
        pred_d = np.load(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy')
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
        PTHA_depth_pred[select_pt,:] = single_exceedance_depth(RP_list, pred_d[:,select_pt], rate)
        PTHA_table_pred_def[select_pt,:] = single_exceedance_rate(Depth_list, pred_d[eve_def,select_pt], rate[eve_def])
        PTHA_depth_pred_def[select_pt,:] = single_exceedance_depth(RP_list, pred_d[eve_def,select_pt], rate[eve_def])
        PTHA_table_pred_nodef[select_pt,:] = single_exceedance_rate(Depth_list, pred_d[eve_nodef,select_pt], rate[eve_nodef])
        PTHA_depth_pred_nodef[select_pt,:] = single_exceedance_depth(RP_list, pred_d[eve_nodef,select_pt], rate[eve_nodef])
       
    # save PTHA tables
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHArate_{train_size}.npy',PTHA_table_pred)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHAdepth_{train_size}.npy',PTHA_depth_pred)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHArate_def_{train_size}.npy',PTHA_table_pred_def)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHAdepth_def_{train_size}.npy',PTHA_depth_pred_def)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHArate_nodef_{train_size}.npy',PTHA_table_pred_nodef)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHAdepth_nodef_{train_size}.npy',PTHA_depth_pred_nodef)

else:
    print('Error: Invalid mode')

