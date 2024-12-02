#Description: calculate PML and AAL for each asset or foglio from event loss table, lookup asset or foglio value from exposure table and index from agg_id table
import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    mode = sys.argv[2] #reprocess or post
    train_size = sys.argv[3] #eventset size used for training or true
    min_loss = int(sys.argv[4]) #minimum loss threshold
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

def get_loss_pml(loss,rate,threshold,PoE,total_value = 28491684200):
    
    #filter events based on threshold of 500
    # print('Number of events before filtering:', len(loss))
    rate = rate[loss>threshold]
    loss = loss[loss>threshold]
    # print('Number of events after filtering:', len(loss))

    #reset index
    rate = rate.reset_index(drop=True)
    loss = loss.reset_index(drop=True)

    #sort descending
    idx = np.argsort(loss)[::-1]
    sorted_loss = loss[idx]
    sorted_rate = rate[idx]  
    
    #Event loss table
    ELT = pd.DataFrame({'loss':sorted_loss,'rate':sorted_rate})
    ELT['weighted_loss'] = sorted_loss*sorted_rate
    ELT['cumulative_probability'] = ELT['rate'].cumsum()

    AAL = np.sum(ELT['weighted_loss'])
    # print('AAL:',AAL,'\n')

    PML = np.zeros(len(PoE))
    PML_ratio = np.zeros(len(PoE))
    #find the event loss for each RP
    for i,probability_level in enumerate(PoE):
        # print(probability_level,len(ELT['loss'][ELT['cumulative_probability']>probability_level]))
        PML[i] = ELT['loss'][ELT['cumulative_probability']>probability_level].max()
        if PML[i] == np.nan:
            PML_ratio[i] = 0
        else:
            PML_ratio[i] = PML[i]/total_value
    #set missing values to 0
    PML[np.isnan(PML)] = 0
    PML_ratio[np.isnan(PML_ratio)] = 0
    return PML,PML_ratio,ELT,AAL

#common settings
#check if PTHA directory exists
if not os.path.exists(f'{MLDir}/risk/results/risk_calc'):
    os.makedirs(f'{MLDir}/risk/results/risk_calc')

RP = np.logspace(3,8,50,base=10)

PoE = [1/rp for rp in RP]

if mode == 'asset':
    #load data
    event_info = pd.read_csv('3_event_info_agg_loss.csv',header=0)
    loss = pd.read_csv(f'2_asset_event_loss_{train_size}.csv',header=0)
    asset_id = pd.read_csv('0_agg_keys.csv',header=0)
    exposure = pd.read_csv(f'{MLDir}/risk/loss/building_exposure.csv',header=0)

    #prepare PML and AAL table for saving loss calculations
    ncol = 3 + len(PoE) #agg_id, IDAG, value, PoE1 loss to PoE22 loss
    nrows = len(asset_id)
    PML_table = np.zeros((nrows,ncol),dtype=object) #agg_id, IDAG, value, PoE1 loss to PoE22 loss x n-assets
    AAL_table = np.zeros((nrows,4),dtype=object) #agg_id, IDAG, value, AAL x n-assets
    log_table = np.zeros((nrows,4),dtype=object) #agg_id, IDAG, value, count of loss events

    for i in range(nrows):
        if i % 100 == 0:
            print(f'Asset {i}',asset_id.IDAG[i])
        asset_loss = loss[loss['agg_id'] == asset_id.agg_id[i]]
        asset_loss = asset_loss.assign(mean_prob=asset_loss['event_id'].apply(lambda x: event_info.loc[x]['mean_prob']))
        asset_value = exposure[exposure['IDAG'] == asset_id.IDAG[i]]['Value'].values[0]
        
        #fill asset info
        PML_table[i,0] = asset_id.agg_id[i]
        PML_table[i,1] = asset_id.IDAG[i]
        PML_table[i,2] = asset_value
        AAL_table[i,0] = asset_id.agg_id[i]
        AAL_table[i,1] = asset_id.IDAG[i]
        AAL_table[i,2] = asset_value
        log_table[i,0] = asset_id.agg_id[i]
        log_table[i,1] = asset_id.IDAG[i]
        log_table[i,2] = asset_value
        log_table[i,3] = len(asset_loss['loss']>min_loss)
        
        # fill PML values, if loss events for the asset
        if len(asset_loss) != 0:
            pml_aal = get_loss_pml(asset_loss['loss'],asset_loss['mean_prob'],min_loss,PoE,asset_value)
            PML_table[i,3:],AAL_table[i,3] = pml_aal[1],pml_aal[3]
    
    #set nan values to 0
    PML_table = pd.DataFrame(PML_table)
    PML_table.columns = ['agg_id','IDAG','value'] + [f'PoE{i}' for i in range(len(RP))]
    PML_table['IDAG'] = PML_table['IDAG'].apply(lambda x: str(int(x)))
    PML_table.to_csv(f'{MLDir}/risk/results/risk_calc/4_PML_table_{mode}_{train_size}.csv',index=False)
    
    AAL_table = pd.DataFrame(AAL_table)
    AAL_table.columns = ['agg_id','IDAG','value','AAL']
    AAL_table['IDAG'] = AAL_table['IDAG'].apply(lambda x: str(int(x)))
    AAL_table.to_csv(f'{MLDir}/risk/results/risk_calc/5_AAL_table_{mode}_{train_size}.csv',index=False)

    log_table = pd.DataFrame(log_table)
    log_table.columns = ['agg_id','IDAG','value','event_count']
    log_table['IDAG'] = log_table['IDAG'].apply(lambda x: str(int(x)))
    log_table.to_csv(f'{MLDir}/risk/results/risk_calc/6_log_table_{mode}_{train_size}.csv',index=False)

elif mode == 'foglio':
    #load data
    event_info = pd.read_csv('3_event_info_foglio_loss.csv',header=0)
    loss = pd.read_csv(f'2_foglio_event_loss_{train_size}.csv',header=0)
    asset_id = pd.read_csv('0_foglio_keys.csv',header=0)
    exposure = pd.read_csv(f'{MLDir}/risk/exposure/Admin_units/foglio_values.csv',header=0)

    #prepare PML and AAL table for saving loss calculations
    ncol = 3 + len(PoE) #agg_id, IDAG, value, PoE1 loss to PoE22 loss
    nrows = len(asset_id)
    PML_table = np.zeros((nrows,ncol), dtype=object) #agg_id, foglio_id, value, PoE1 loss to PoE22 loss x n-assets
    AAL_table = np.zeros((nrows,4), dtype=object) #agg_id, foglio_id, value, AAL x n-assets
    log_table = np.zeros((nrows,4), dtype=object) #agg_id, foglio_id, value, count of loss events

    for i in range(nrows):
        if i % 100 == 0:
            print(f'Foglio {i}',asset_id.foglio_id[i])
        asset_loss = loss[loss['agg_id'] == asset_id.agg_id[i]]
        asset_loss = asset_loss.assign(mean_prob=asset_loss['event_id'].apply(lambda x: event_info.loc[x]['mean_prob']))
        asset_value = exposure[exposure['Foglio'] == asset_id.foglio_id[i]]['Value'].values[0]
        
        #fill asset info
        PML_table[i,0] = asset_id.agg_id[i]
        PML_table[i,1] = asset_id.foglio_id[i]
        PML_table[i,2] = asset_value
        AAL_table[i,0] = asset_id.agg_id[i]
        AAL_table[i,1] = asset_id.foglio_id[i]
        AAL_table[i,2] = asset_value
        log_table[i,0] = asset_id.agg_id[i]
        log_table[i,1] = asset_id.foglio_id[i]
        log_table[i,2] = asset_value
        log_table[i,3] = len(asset_loss['loss']>min_loss)
        
        # fill PML values, if loss events for the asset
        if len(asset_loss) != 0:
            pml_aal = get_loss_pml(asset_loss['loss'],asset_loss['mean_prob'],min_loss,PoE,asset_value)
            PML_table[i,3:],AAL_table[i,3] = pml_aal[1],pml_aal[3]
    
    PML_table = pd.DataFrame(PML_table)
    PML_table.columns = ['agg_id','foglio_id','value'] + [f'PoE{i}' for i in range(len(RP))]
    PML_table.to_csv(f'{MLDir}/risk/results/risk_calc/4_PML_table_{mode}_{train_size}.csv',index=False)
    
    AAL_table = pd.DataFrame(AAL_table)
    AAL_table.columns = ['agg_id','foglio_id','value','AAL']
    AAL_table.to_csv(f'{MLDir}/risk/results/risk_calc/5_AAL_table_{mode}_{train_size}.csv',index=False)

    log_table = pd.DataFrame(log_table)
    log_table.columns = ['agg_id','foglio_id','value','event_count']
    log_table.to_csv(f'{MLDir}/risk/results/risk_calc/6_log_table_{mode}_{train_size}.csv',index=False)

elif mode == 'const':
        #load data
    event_info = pd.read_csv('3_event_info_const_loss.csv',header=0)
    loss = pd.read_csv(f'2_const_event_loss_{train_size}.csv',header=0)
    asset_id = pd.read_csv('0_const_keys.csv',header=0)
    exposure = pd.read_csv(f'{MLDir}/risk/exposure/Admin_units/const_values.csv',header=0)

    #prepare PML and AAL table for saving loss calculations
    ncol = 3 + len(PoE) #agg_id, IDAG, value, PoE1 loss to PoE22 loss
    nrows = len(asset_id)
    PML_table = np.zeros((nrows,ncol), dtype=object) #agg_id, const_id, value, PoE1 loss to PoE22 loss x n-assets
    AAL_table = np.zeros((nrows,4), dtype=object) #agg_id, const_id, value, AAL x n-assets
    log_table = np.zeros((nrows,4), dtype=object) #agg_id, const_id, value, count of loss events

    for i in range(nrows):
        if i % 100 == 0:
            print(f'Construction {i}',asset_id.const_id[i])
        asset_loss = loss[loss['agg_id'] == asset_id.agg_id[i]]
        asset_loss = asset_loss.assign(mean_prob=asset_loss['event_id'].apply(lambda x: event_info.loc[x]['mean_prob']))
        asset_value = exposure[exposure['Construction'] == asset_id.const_id[i]]['Value'].values[0]
        
        #fill asset info
        PML_table[i,0] = asset_id.agg_id[i]
        PML_table[i,1] = asset_id.const_id[i]
        PML_table[i,2] = asset_value
        AAL_table[i,0] = asset_id.agg_id[i]
        AAL_table[i,1] = asset_id.const_id[i]
        AAL_table[i,2] = asset_value
        log_table[i,0] = asset_id.agg_id[i]
        log_table[i,1] = asset_id.const_id[i]
        log_table[i,2] = asset_value
        log_table[i,3] = len(asset_loss['loss']>min_loss)
        
        # fill PML values, if loss events for the asset
        if len(asset_loss) != 0:
            pml_aal = get_loss_pml(asset_loss['loss'],asset_loss['mean_prob'],min_loss,PoE,asset_value)
            PML_table[i,3:],AAL_table[i,3] = pml_aal[1],pml_aal[3]
    
    PML_table = pd.DataFrame(PML_table)
    PML_table.columns = ['agg_id','const_id','value'] + [f'PoE{i}' for i in range(len(RP))]
    PML_table.to_csv(f'{MLDir}/risk/results/risk_calc/4_PML_table_{mode}_{train_size}.csv',index=False)
    
    AAL_table = pd.DataFrame(AAL_table)
    AAL_table.columns = ['agg_id','const_id','value','AAL']
    AAL_table.to_csv(f'{MLDir}/risk/results/risk_calc/5_AAL_table_{mode}_{train_size}.csv',index=False)

    log_table = pd.DataFrame(log_table)
    log_table.columns = ['agg_id','const_id','value','event_count']
    log_table.to_csv(f'{MLDir}/risk/results/risk_calc/6_log_table_{mode}_{train_size}.csv',index=False)

else:
    print('Error: Invalid mode')

