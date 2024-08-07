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

def exceedance_curve(thresholds, max_stage_point, rates): 
   # Annual Mean Rate of threshold exceedance 
   lambda_exc = np.zeros((len(thresholds), 1000)) 
   for threshold in range(len(thresholds)): 
       ix = np.array(np.where(max_stage_point > thresholds[threshold])).squeeze() 
       for j in range(1000): 
           #for index in ix: 
           lambda_exc[threshold, j] = lambda_exc[threshold, j] +  rates[ix,j].sum(axis=0) 
   #------------------------------------------------ 
   return lambda_exc 

def single_exceedance_curve(thresholds, max_stage_point, rates): 
   # Annual Mean Rate of threshold exceedance 
   lambda_exc = np.zeros((len(thresholds), )) 
   for threshold in range(len(thresholds)): 
       ix = np.array(np.where(max_stage_point > thresholds[threshold])).squeeze() 
       #for index in ix: 
       lambda_exc[threshold] = lambda_exc[threshold] +  rates[ix].sum(axis=0) 
   #------------------------------------------------ 
   return lambda_exc

def make_map(RP_table_true,RP_table_pred,RP_table_sis,RPi):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axs):
        if i == 0:
            RP_table = RP_table_true
            title = 'True'
        elif i == 1:
            RP_table = RP_table_pred
            title = 'Pred'
        elif i == 2:
            RP_table = RP_table_sis
            title = 'SIS'
        array_2d = np.zeros((y_dim,x_dim))
        array_2d[~zero_mask] = RP_table[:,RPi]
        im = ax.imshow(array_2d, cmap='viridis',vmin=0, vmax=200)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.axis('off')
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal',label='Depth (cm)')
    plt.suptitle(f'Return period depths at {RP_list[RPi]}')
    plt.savefig(f'{MLDir}/model/{reg}/PTHA/RP{RPi+1}_map_{train_size}.png')
    plt.close()

if mode == 'PTHA':
    #check if PTHA directory exists
    if not os.path.exists(f'{MLDir}/model/{reg}/PTHA'):
        os.makedirs(f'{MLDir}/model/{reg}/PTHA')

    RP_list = [1/1000,1/10000,1/100000]
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

    for test_size in ['0','1','2','3']:
        print(f'Processing test size {test_size}')
        # load test events related parameters
        event_list_path = f'{MLDir}/data/events/shuffled_events_test_{reg}_{test_size}.txt'
        event_list = np.loadtxt(event_list_path, dtype='str')
        n_eve = len(event_list)    
        # true_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{test_size}.dat',
        #                         mode='r',
        #                         dtype=float,
        #                         shape=(n_eve, nflood_grids))
        pred_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{test_size}_{train_size}_prediction.dat',
                                mode='r',
                                dtype=float,
                                shape=(n_eve, nflood_grids))    

        # true = torch.tensor(true_d_array,dtype=torch.float16)
        pred = torch.tensor(pred_d_array,dtype=torch.float16)
        #append events along first axis
        if test_size == '0':
            # true_d = true
            pred_d = pred
        else:
            # true_d = torch.cat((true_d,true),0)
            pred_d = torch.cat((pred_d,pred),0)

    # true_d = true_d*100
    pred_d = pred_d*100

    # #change datatype to int
    # true_d = true_d.type(torch.int16)
    pred_d = pred_d.type(torch.int16)
    # sis_d = pred_d[sample_idx] #filter events in eve_dep that are in eve_SIS

    # #save depth tables in cm
    # print('Saving depth tables')
    # np.save(f'{MLDir}/model/{reg}/PTHA/true_d_53500.npy',true_d)
    np.save(f'{MLDir}/model/{reg}/PTHA/pred_d_{train_size}.npy',pred_d)
    # np.save(f'{MLDir}/model/{reg}/PTHA/sis_d_{train_size}.npy',sis_d)

    # #compute depth thresholds 
    # rate = eve_perf['mean_prob']
    # thresholds_depth = np.linspace(0, 200, 100)

    # PTHA_table_true = np.zeros((nflood_grids,len(thresholds_depth)))
    # PTHA_table_pred = np.zeros((nflood_grids,len(thresholds_depth)))
    # PTHA_table_sis = np.zeros((nflood_grids,len(thresholds_depth)))

    # RP_table_true = np.zeros((nflood_grids,len(RP_list)))
    # RP_table_pred = np.zeros((nflood_grids,len(RP_list)))
    # RP_table_sis = np.zeros((nflood_grids,len(RP_list)))

    # for select_pt in range(nflood_grids): #nflood_grids
    #     #compute exceedance curve for true
    #     lambda_exc_true = single_exceedance_curve(thresholds_depth, true_d[:,select_pt], rate)
    #     lambda_exc_pred = single_exceedance_curve(thresholds_depth, pred_d[:,select_pt], rate)
    #     lambda_exc_sis = single_exceedance_curve(thresholds_depth, sis_d[:,select_pt], rate)

    #     PTHA_table_true[select_pt] = lambda_exc_true
    #     PTHA_table_pred[select_pt] = lambda_exc_pred
    #     PTHA_table_sis[select_pt] = lambda_exc_sis

    #     #compute return period depths at 1000,10000,100000 years ie 1/1000,1/10000,1/100000
    #     RP_table_true[select_pt] = np.interp(RP_list, lambda_exc_true, thresholds_depth)
    #     RP_table_pred[select_pt] = np.interp(RP_list, lambda_exc_pred, thresholds_depth)    
    #     RP_table_sis[select_pt] = np.interp(RP_list, lambda_exc_sis, thresholds_depth)
        
    #     #plot exceedance curve
    #     if select_pt%1000 == 0:
    #         print(f'Plotting exceedance curve for gauge {select_pt}')
    #         plt.figure()
    #         plt.plot(thresholds_depth,lambda_exc_true, label='True')
    #         plt.plot(thresholds_depth,lambda_exc_pred, label='Pred')
    #         plt.plot(thresholds_depth,lambda_exc_sis, label='SIS')
    #         plt.xlabel('Depth')
    #         plt.ylabel('Exceedance rate')
    #         plt.yscale('log')
    #         #fix y axis limits
    #         # plt.ylim(1e-6, 1)
    #         plt.legend()
    #         plt.title(f'Exceedance curve for gauge {select_pt}')
    #         plt.savefig(f'{MLDir}/model/{reg}/PTHA/exceedance_curve_{select_pt}.png')
    #         plt.close()
       
    # #save PTHA tables
    # np.save(f'{MLDir}/model/{reg}/PTHA/true_PTHA_53500.npy',PTHA_table_true)
    # np.save(f'{MLDir}/model/{reg}/PTHA/pred_PTHA_{train_size}.npy',PTHA_table_pred)
    # np.save(f'{MLDir}/model/{reg}/PTHA/sis_PTHA_{str(len(sample_idx))}.npy',PTHA_table_sis)

    # #save RP tables
    # np.save(f'{MLDir}/model/{reg}/PTHA/true_RP_53500.npy',RP_table_true)
    # np.save(f'{MLDir}/model/{reg}/PTHA/pred_RP_{train_size}.npy',RP_table_pred)
    # np.save(f'{MLDir}/model/{reg}/PTHA/sis_RP_{str(len(sample_idx))}.npy',RP_table_sis)

    # #make RP1,RP2,RP3 comparison depth map plots
    # print('Making map plots')
    # for RPi in range(len(RP_list)):
    #     make_map(RP_table_true,RP_table_pred,RP_table_sis,RPi)

# else:
#     print('Error: Invalid mode')

