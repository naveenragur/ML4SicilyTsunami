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
    test_size = sys.argv[2] #eventset size for testing
    mode = sys.argv[3] #train or test
    train_size = sys.argv[4] #eventset size used for training
    mask_size = sys.argv[5] #eventset size for mask used in preprocessing
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

def process_map(eve_name,tmap_array,map_array, min_area=10, save_plots=False, save_output=False, output_file=None):
    # Example usage:
    s = [[1,1,1],
        [1,1,1],
        [1,1,1]]
    map_array[map_array<0.1] = 0
    labeled_array, num_features = label(map_array, structure=s)

    # print("\nNumber of Features:", num_features)
    # print("\nsize of labeled array:", labeled_array.shape)
    # print("\nUnique Labels:", len(np.unique(labeled_array)))

    component_sizes = np.bincount(labeled_array.flatten())
    mask = component_sizes > min_area
    # print("\nThresh Size:", component_sizes.shape)
    # print("\nMask Size:", mask.shape)

    filtered_array = np.zeros_like(map_array)
    filtered_array[np.isin(labeled_array, np.nonzero(mask))] = 1
    
    # Set filtered array as int
    filtered_array = filtered_array.astype(int)

    if save_plots:
        # Plot original map and corrected maps side by side
        fig = plt.figure(figsize=(20, 40))        
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
        
        tmap_array[tmap_array == 0] = np.nan #set 0 values to nan for plotting
        true = ax0.imshow(tmap_array, cmap='RdYlBu', interpolation='none')
        ax0.set_title('True')  
        ax0.invert_yaxis()

        map_array[map_array == 0] = np.nan 
        org = ax1.imshow(map_array, cmap='RdYlBu', interpolation='none')
        ax1.set_title('ML')
        ax1.invert_yaxis()

        map_array[filtered_array == 0] = np.nan
        post = ax2.imshow(map_array, cmap='RdYlBu', interpolation='none')
        ax2.set_title('Postprocess')
        ax2.invert_yaxis()

        # Common colorbar
        fig.colorbar(post, ax=[ax0, ax1, ax2], orientation='horizontal',shrink=0.5)
        plt.title(eve_name)
        plt.savefig(f'{MLDir}/model/{reg}/postprocess/{event}.png')
        plt.close()

    else:
        map_array[filtered_array == 0] = np.nan

    if save_output:
        if output_file:
            np.save(output_file, map_array)
        else:
            print("Error: No output file specified for saving.")
    
    return map_array#, filtered_array

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

if mode == 'post':
    #string template for file filepath
    Dpath = SimDir +'/{:s}/{:s}_flowdepth.nc'
    event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_test_{reg}_{test_size}.txt', dtype='str')
else:
    print('Error: Invalid mode')

model_out = np.load(f'{MLDir}/model/{reg}/out/pred_trainsize{train_size}_testsize{test_size}.npy')
zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
print('zero_mask shape:', zero_mask.shape)
non_zero_list = np.argwhere(~zero_mask).tolist()
empty_model_out = np.zeros_like(model_out)

for i, event in enumerate(event_list):
    if i%1000 == 0:
        print(f'Processing {i}th event:{event}')
    
    #read in data
    Dfile = Dpath.format(event, reg)
    D = xr.open_dataset(Dfile).z
    dtrue = D.values
    D.close()
    
    #prep data
    event_out_2d = np.zeros((y_dim,x_dim))
    event_out_2d[~zero_mask] = model_out[i]
    #postprocess data
    event_out_2d = process_map(event,dtrue, event_out_2d, min_area=10, save_plots=True, save_output=False, output_file=f'{MLDir}/model/{reg}/postprocess/{event}.npy')
    empty_model_out[i,:] = event_out_2d[~zero_mask]

#save postprocessed data
np.save(f'{MLDir}/model/{reg}/out/postprocessed_trainsize{train_size}_testsize{test_size}.npy', empty_model_out)

#save as memap file like reduced onshore depths in preprocessing
onshore_map = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{test_size}_prediction.dat',
                         mode='w+',
                         dtype=float,
                         shape=(empty_model_out.shape[0], empty_model_out.shape[1]))

onshore_map[:] = empty_model_out[:]




    

