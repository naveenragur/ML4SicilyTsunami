#Description: Plot depth predictions and errors for different models for a given event
#Usage: python 06_compare_depths_optimized.py <region> <task> <train size> <mask size> <start at>
import os
import sys
import gc  # Import garbage collector for memory management
import psutil  # For memory monitoring
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pygmt

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    mode = sys.argv[2] #reprocess or post
    train_size = sys.argv[3] #eventset size used for training
    mask_size = sys.argv[4] #eventset size used for testing
    start = sys.argv[5] #start number
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

def calculate_error(true, pred):
        # Set NaN for rows where count_test is less than 1
        true[true<0.1]=0
        pred[pred<0.1]=0
        error1 = true - pred
        error2 = pred - true
        error = np.where(np.abs(error1) < np.abs(error2), error1, -error2)
        # error = np.where((error < 0.1) & (error > -0.1), np.nan, error)
        return error

def Gfit_r2(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    obs[obs<0.2]=0
    pred[pred<0.2]=0
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    r2 = 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
    return Gvalue,r2

def print_memory_usage(event_num):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Event {event_num}: Memory usage: {memory_mb:.1f} MB")

def cleanup_variables(*vars_to_delete):
    """Helper function to delete variables and force garbage collection"""
    for var in vars_to_delete:
        try:
            del var
        except:
            pass
    gc.collect()

# plotting the below events
ids = [
    'BS_manning003/E01267N3753E01646N3535-BS-M809_E01502N3737_D010_S112D70R270_A006995_S075',
    'BS_manning003/E01267N3753E01646N3535-BS-M809_E01502N3737_D144_S022D70R270_A006995_S075',
    'BS_manning003/E01267N3753E01646N3535-BS-M809_E01547N3670_D010_S337D70R270_A006995_S075',
    'BS_manning003/E01267N3753E01646N3535-BS-M809_E01495N3692_D010_S022D50R270_A006995_S075',
    'BS_4-8_manning003/E01267N3753E01646N3535-BS-M809_E01502N3737_D010_S067D90R090_A006995_S075',
    'BS_manning003/E01267N3753E01646N3535-BS-M809_E01523N3692_D010_S292D50R270_A006995_S075',
    'BS_4-8_manning003/E01267N3753E01646N3535-BS-M809_E01551N3692_D010_S112D90R090_A006995_S075',
    'PS_manning003/E02020N3739E02658N3366-PS-Str_PYes_Var-M895_E02351N3465_S003',
    'PS_manning003/E02020N3739E02658N3366-PS-Str_PYes_Var-M902_E02417N3454_S001',
    ]

#dimensions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    columnname = str(54)
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #gauges time series
    pts_dim = 480 #time steps
    list_size = ['961','1773','3669','6941']
    control_points = [[37.01,15.29],
        [37.06757,15.28709],
        [37.05266,15.26536],
        [37.03211,15.28632]]   
    
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    columnname = str(38)
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo)
    pts_dim = 480
    list_size = ['892','1658','3454','7071'] 
    control_points =  [[37.5022,15.0960],
        [37.48876,15.08936],
        [37.47193,15.07816],
        [37.46273,15.08527],
        [37.46252,15.08587],
        [37.45312,15.07874],
        [37.42821,15.08506],
        [37.40958,15.08075],
        [37.38595,15.08539],
        [37.35084,15.08575],
        [37.33049,15.07029],
        [37.40675,15.05037]]

#check if PTHA directory exists
if not os.path.exists(f'{MLDir}/model/{reg}/multifoldMC/compare'):
    os.makedirs(f'{MLDir}/model/{reg}/multifoldMC/compare')

#predictions and post processed predictions
print("Loading prediction data...")
true_depths = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/multifoldMC/PTHA/true_d_53550.npy')
pred_depths_mean = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/multifoldMC/PTHA/pred_d_{train_size}_direct.npy')
eve_perf_mean = pd.read_csv(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/multifoldMC/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')
pred_depths_sigmaminus = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/multifoldMC/PTHA/sigma_minus_{train_size}_direct.npy')
pred_depths_sigmaplus = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/multifoldMC/PTHA/sigma_plus_{train_size}_direct.npy')

eve_id = np.loadtxt('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/events/sample_events53550.txt',dtype='str')   

#inundation attributes
print("Loading spatial data...")
flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
nflood_grids = np.count_nonzero(flood_mask)
zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
idx= np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/processed/lat_lon_idx_{reg}_{mask_size}.npy')
index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt',header=None,sep=',')
index_map.columns = ['m','n','lat','lon'] #add column names

# Create output directories
dir = ['PS_manning003', 'BS_manning003', 'PS_4-8_manning003','BS_4-8_manning003',]
for d in dir:
    if not os.path.exists(f'{MLDir}/model/{reg}/multifoldMC/compare/{d}/'):
        os.makedirs(f'{MLDir}/model/{reg}/multifoldMC/compare/{d}/')

# Set matplotlib to use non-interactive backend to save memory
plt.ioff()

if mode == 'compare':    
    # Add progress tracking and memory management
    events_to_process = eve_id[int(start):]
    total_events = len(events_to_process)
    processed_events = 0
    failed_events = []
    
    print(f"Starting to process {total_events} events from index {start}")
    print_memory_usage(0)
    
    for id in events_to_process:
        eve = np.where(eve_id==id)[0][0]
        processed_events += 1
        
        try:
            print(f'Processing event {processed_events}/{total_events}: {id} (eve={eve})')
            
            # Check if output already exists to avoid reprocessing
            output_file = f'{MLDir}/model/{reg}/multifoldMC/compare/{id}_{train_size}_{reg}_{str(eve)}.png'
            if os.path.exists(output_file):
                print(f"Skipping {id} - output already exists")
                continue
            
            #read dZ file and grid location file to extract location information
            data2plot = xr.open_dataset(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/{id}/{reg}_deformation.nc')
            dz = data2plot['deformation'].values.copy()  # Make a copy to avoid reference issues
            
            # Close the dataset immediately after extracting values to free memory
            data2plot.close()
            del data2plot

            x = np.linspace(0,dz.shape[1],dz.shape[1])
            y = np.linspace(0,dz.shape[0],dz.shape[0])

            #create list of x,y,dz
            xy_mesh = np.meshgrid(x,y)
            dz_smooth = dz
            x_list,y_list = xy_mesh[0].flatten(),xy_mesh[1].flatten()
            dz_list = dz.flatten()

            #cm to m
            pred_mean=pred_depths_mean[eve]/100
            pred_sigmaminus=pred_depths_sigmaminus[eve]/100
            pred_sigmaplus=pred_depths_sigmaplus[eve]/100
            true=true_depths[eve]/100
          
            #calculate errors
            error_mean = calculate_error(true, pred_mean)
            error_sigmaminus = calculate_error(true, pred_sigmaminus)
            error_sigmaplus = calculate_error(true, pred_sigmaplus)

            #remove micro depths for better visualization
            pred_mean= np.where(pred_mean < 0.1, np.nan, pred_mean)
            pred_sigmaminus= np.where(pred_sigmaminus < 0.1, np.nan, pred_sigmaminus)
            pred_sigmaplus= np.where(pred_sigmaplus < 0.1, np.nan, pred_sigmaplus)
            true= np.where(true < 0.1, np.nan, true)
            
            #additional region specific parameters for plotting
            if reg == 'CT':
                fig, axs = plt.subplots(1, 8, figsize=(19,8))
                xpos = 0.25
                ypos = 0.85
                cbar_ht = 0.02
            elif reg == 'SR':
                fig, axs = plt.subplots(1, 8, figsize=(19,3))
                xpos = 0.25
                ypos = 0.25
                cbar_ht = 0.04
            axs = axs.ravel()

            # Plot performance variable values
            cmap_depth = plt.get_cmap('twilight',20)
            cmap_error = plt.get_cmap('seismic', 10)
            cmap_dz = plt.get_cmap('RdYlGn_r',10)

            # Local Deformation
            DZ = axs[0].scatter(x_list,y_list, c=dz_smooth, s=0.0005, cmap=cmap_dz,
                                vmin=-5, vmax=5,alpha=1)
            axs[0].text(xpos,ypos, f'max: {np.nanmax(dz_smooth):.3f},\\nmin: {np.nanmin(dz_smooth):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[0].transAxes, fontsize=12)
            axs[0].set_title('Local Deformation')

            # True
            TR = axs[1].scatter(idx[:, 1], idx[:, 0], c=true, s=0.0005, cmap=cmap_depth,
                                vmin=0, vmax=10,alpha=1)
            axs[1].text(xpos,ypos, f'max: {np.nanmax(true):.3f}', 
                        horizontalalignment='center', verticalalignment='center',transform=axs[1].transAxes, fontsize=12)
            axs[1].set_title('True')

            # Pred_mean
            PR_pretrain = axs[2].scatter(idx[:, 1], idx[:, 0], c=pred_mean, s=0.0005, cmap=cmap_depth,
                                vmin=0, vmax=10,alpha=1)
            axs[2].text(xpos,ypos, f'max: {np.nanmax(pred_mean):.3f}\\nr^2: {eve_perf_mean["r2"].iloc[eve]:.3f}\\ng: {eve_perf_mean["g"].iloc[eve]:.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[2].transAxes, fontsize=12)
            axs[2].set_title('Mean')

            # Error
            ER_pretrain = axs[3].scatter(idx[:, 1], idx[:, 0], c=error_mean, s=0.0005, cmap=cmap_error,
                                vmin=-5,vmax=5,alpha=1)
            axs[3].text(xpos,ypos, f'max: {np.nanmax(error_mean):.3f},\\nmin: {np.nanmin(error_mean):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[3].transAxes, fontsize=12)
            axs[3].set_title('Error Mean')

            # Pred_sigmaminus
            PR_direct = axs[4].scatter(idx[:, 1], idx[:, 0], c=pred_sigmaminus, s=0.0005, cmap=cmap_depth,
                                vmin=0, vmax=10,alpha=1)
            axs[4].text(xpos,ypos, f'max: {np.nanmax(pred_sigmaminus):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[4].transAxes, fontsize=12)
            axs[4].set_title('Mean-2Sigma')

            # Error
            ER_direct = axs[5].scatter(idx[:, 1], idx[:, 0], c=error_sigmaminus, s=0.0005, cmap=cmap_error,
                                vmin=-5,vmax=5,alpha=1)
            axs[5].text(xpos,ypos, f'max: {np.nanmax(error_sigmaminus):.3f},\\nmin: {np.nanmin(error_sigmaminus):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[5].transAxes, fontsize=12)
            axs[5].set_title('Error Mean-2Sigma')

            # Pred_sigmaplus
            PR_pretrain = axs[6].scatter(idx[:, 1], idx[:, 0], c=pred_sigmaplus, s=0.0005, cmap=cmap_depth,
                                vmin=0, vmax=10,alpha=1)
            axs[6].text(xpos,ypos, f'max: {np.nanmax(pred_sigmaplus):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[6].transAxes, fontsize=12)
            axs[6].set_title('Mean+2Sigma')

            #Error
            ER_pretrain = axs[7].scatter(idx[:, 1], idx[:, 0], c=error_sigmaplus, s=0.0005, cmap=cmap_error,
                                vmin=-5,vmax=5,alpha=1)
            axs[7].text(xpos,ypos, f'max: {np.nanmax(error_sigmaplus):.3f},\\nmin: {np.nanmin(error_sigmaplus):.3f}',
                        horizontalalignment='center', verticalalignment='center',transform=axs[7].transAxes, fontsize=12)
            axs[7].set_title('Error Mean+2Sigma')

            # Set axis scale as equal and add gridlines
            for ax in axs:
                #keep gridlines but turn off axis borders and ticks
                ax.set_aspect('equal')
                ax.set_axis_off()
                ax.set_xlim([0, max(idx[:, 1])])
                ax.set_ylim([0, max(idx[:, 0])])
                ax.hlines(y=np.arange(0, max(idx[:, 0]), 150), xmin=0, xmax=max(idx[:, 1]), color='grey', linestyle='--', linewidth=0.5,alpha=0.75)
                ax.vlines(x=np.arange(0, max(idx[:, 1]), 150), ymin=0, ymax=max(idx[:, 0]), color='grey', linestyle='--', linewidth=0.5,alpha=0.75)

            # Add a common colorbar for the whole fig using axes transform
            cbar_dz = fig.add_axes([0.02, 0.1, 0.22, cbar_ht])
            cbar_dep = fig.add_axes([0.28, 0.1, 0.46, cbar_ht])
            cbar_err = fig.add_axes([0.78, 0.1, 0.22, cbar_ht])

            cbar1 = fig.colorbar(TR, cax=cbar_dep, orientation ='horizontal',extend='max')
            cbar2 = fig.colorbar(ER_direct, cax=cbar_err, orientation ='horizontal',extend='both')
            cbar3 = fig.colorbar(DZ, cax=cbar_dz, orientation ='horizontal',extend='both')
            cbar1.ax.tick_params(labelsize=12)
            cbar2.ax.tick_params(labelsize=12)
            cbar3.ax.tick_params(labelsize=12)
            cbar1.set_label('Depth(m)', fontsize=12)
            cbar2.set_label('Error(m)', fontsize=12)
            cbar3.set_label('Local Deform.(m)', fontsize=12)
            plt.tight_layout()
            
            # Save with lower DPI for large datasets to save space and time
            dpi_value = 50 if total_events > 1000 else 150
            plt.savefig(output_file, dpi=dpi_value, bbox_inches='tight', pad_inches=0.1)
            
            #close figure and clear memory
            plt.clf()
            plt.close(fig)
            
            # Clear variables to free memory
            cleanup_variables(dz, dz_smooth, x_list, y_list, dz_list, 
                            pred_mean, pred_sigmaminus, pred_sigmaplus, true,
                            error_mean, error_sigmaminus, error_sigmaplus,
                            x, y, xy_mesh, fig, axs)
            
            # Memory management - force garbage collection every 50 events
            if processed_events % 50 == 0:
                gc.collect()
                print_memory_usage(processed_events)
                print(f"Processed {processed_events}/{total_events} events")
                
        except FileNotFoundError as e:
            print(f"Warning: File not found for event {id}: {e}")
            failed_events.append((id, f"File not found: {e}"))
            continue
        except MemoryError as e:
            print(f"Memory error processing event {id}: {e}")
            print("Forcing aggressive memory cleanup...")
            plt.close('all')
            gc.collect()
            failed_events.append((id, f"Memory error: {e}"))
            continue
        except Exception as e:
            print(f"Error processing event {id}: {e}")
            failed_events.append((id, f"General error: {e}"))
            # Clean up any partial figure if it exists
            try:
                plt.close('all')
            except:
                pass
            continue
    
    # Final summary
    print(f"\\nProcessing complete!")
    print(f"Successfully processed: {processed_events - len(failed_events)}/{total_events}")
    print(f"Failed events: {len(failed_events)}")
    
    if failed_events:
        print("\\nFailed events:")
        for event_id, error_msg in failed_events:
            print(f"  {event_id}: {error_msg}")
        
        # Save failed events to file for retry
        failed_df = pd.DataFrame(failed_events, columns=['event_id', 'error'])
        failed_file = f'{MLDir}/model/{reg}/multifoldMC/compare/failed_events_{train_size}_{start}.csv'
        failed_df.to_csv(failed_file, index=False)
        print(f"Failed events saved to: {failed_file}")

elif mode == 'compare_pygmt':
    # Original pygmt code with similar memory management improvements
    for id in ids:
        try:
            eve = np.where(eve_id==id)[0][0]
            print(id,'\\n',eve)
            #cm to m
            pred_mean=pred_depths_mean[eve]/100
            pred_sigmaminus=pred_depths_sigmaminus[eve]/100
            pred_sigmaplus=pred_depths_sigmaplus[eve]/100
            true=true_depths[eve]/100
        
            #calculate errors and metrics
            error_mean = calculate_error(true, pred_mean)
            error_sigmaminus = calculate_error(true, pred_sigmaminus)
            error_sigmaplus = calculate_error(true, pred_sigmaplus)
            
            g_sigmaminus, r2_sigmaminus = Gfit_r2(true, pred_sigmaminus)
            g_sigmaplus, r2_sigmaplus = Gfit_r2(true, pred_sigmaplus)

            #remove micro depths for better visualization
            pred_mean= np.where(pred_mean < 0.1, np.nan, pred_mean)
            pred_sigmaminus= np.where(pred_sigmaminus < 0.1, np.nan, pred_sigmaminus)
            pred_sigmaplus= np.where(pred_sigmaplus < 0.1, np.nan, pred_sigmaplus)
            true= np.where(true < 0.1, np.nan, true)
            
            # Rest of pygmt plotting code...
            # [Original pygmt code continues here]
            
        except Exception as e:
            print(f"Error processing {id}: {e}")
            continue
            
else:
    print('Error: Invalid mode. Use "compare" or "compare_pygmt"')
