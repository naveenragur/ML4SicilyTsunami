#Description: Plot depth predictions and errors for different models for a given event
import os
import sys
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
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

def calculate_error(true, pred):
        # Set NaN for rows where count_test is less than 1
        error1 = true - pred
        error2 = pred - true
        error = np.where(np.abs(error1) < np.abs(error2), error1, -error2)
        error = np.where((error < 0.1) & (error > -0.1), np.nan, error)
        return error

#plotting the below events
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
if not os.path.exists(f'{MLDir}/model/{reg}/compare'):
    os.makedirs(f'{MLDir}/model/{reg}/compare')

#predictions and post processed predictions
true_depths = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/PTHA/true_d_53550.npy')

pred_depths_nodeform = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/PTHA/pred_d_{train_size}_nodeform.npy')
eve_perf_nodeform = pd.read_csv(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/out/model_nodeform_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')

pred_depths_direct = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/PTHA/pred_d_{train_size}_direct.npy')
eve_perf_direct = pd.read_csv(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')            

pred_depths_pretrain = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/PTHA/pred_d_{train_size}.npy')
eve_perf_pretrain = pd.read_csv(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/{reg}/out/model_coupled_off[64, 128, 256]_on[16, 128, 128]_{train_size}_compile_combined.csv')            

eve_id = np.loadtxt('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/events/sample_events53550.txt',dtype='str')   
#inundation attributes
flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
nflood_grids = np.count_nonzero(flood_mask)
zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
idx= np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/processed/lat_lon_idx_{reg}_{mask_size}.npy')
index_map = pd.read_csv(f'{MLDir}/data/processed/lat_lon_idx_{reg}_{mask_size}.txt',header=None,sep=',')
index_map.columns = ['m','n','lat','lon'] #add column names

if mode == 'compare':
    for id in eve_id:
        eve = np.where(eve_id==id)[0][0]
        print(id,'\n',eve)
        # eve=32145
        # id =eve_id[eve]

        #read dZ file and grid location file to extract location information
        data2plot = xr.open_dataset(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/{id}/{reg}_deformation.nc')
        dz = data2plot['deformation'].values
        # (948 x 1300)
        # x = data2plot['x'].values #1300
        # y = data2plot['y'].values #948

        x = np.linspace(0,dz.shape[1],dz.shape[1])
        y = np.linspace(0,dz.shape[0],dz.shape[0])

        #create list of x,y,dz
        xy_mesh = np.meshgrid(x,y)
        # dz_smooth = gaussian_filter(dz, sigma=50)
        dz_smooth = dz
        x_list,y_list = xy_mesh[0].flatten(),xy_mesh[1].flatten()
        dz_list = dz.flatten()

        #cm to m
        pred_pretrain=pred_depths_pretrain[eve]/100
        pred_direct=pred_depths_direct[eve]/100
        pred_nodeform=pred_depths_nodeform[eve]/100
        true=true_depths[eve]/100
      
        #calculate errors
        error_pretrain = calculate_error(true, pred_pretrain)
        error_direct = calculate_error(true, pred_direct)
        error_nodeform = calculate_error(true, pred_nodeform)

        #remove micro depths for better visualization
        pred_pretrain= np.where(pred_pretrain < 0.1, np.nan, pred_pretrain)
        pred_direct= np.where(pred_direct < 0.1, np.nan, pred_direct)
        pred_nodeform= np.where(pred_nodeform < 0.1, np.nan, pred_nodeform)
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

        # depth_max = max(np.nanmax(pred),np.nanmax(true))
        # error_max = round(np.nanmax(np.abs(error)))
        # dz_max = round(np.nanmax(np.abs(dz_smooth)))

        # Local Deformation
        DZ = axs[0].scatter(x_list,y_list, c=dz_smooth, s=0.0005, cmap=cmap_dz,
                            vmin=-5, vmax=5,alpha=1)
        axs[0].text(xpos,ypos, f'max: {np.nanmax(dz_smooth):.3f},\nmin: {np.nanmin(dz_smooth):.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[0].transAxes, fontsize=12)
        axs[0].set_title('Local Deformation')

        # True
        TR = axs[1].scatter(idx[:, 1], idx[:, 0], c=true, s=0.0005, cmap=cmap_depth,
                            vmin=0, vmax=10,alpha=1)
        axs[1].text(xpos,ypos, f'max: {np.nanmax(true):.3f}', 
                    horizontalalignment='center', verticalalignment='center',transform=axs[1].transAxes, fontsize=12)
        axs[1].set_title('True')

        # Pred_no_def
        PR_pretrain = axs[2].scatter(idx[:, 1], idx[:, 0], c=pred_nodeform, s=0.0005, cmap=cmap_depth,
                            vmin=0, vmax=10,alpha=1)
        axs[2].text(xpos,ypos, f'max: {np.nanmax(pred_nodeform):.3f}\nr^2: {eve_perf_nodeform["r2"].iloc[eve]:.3f}\ng: {eve_perf_nodeform["g"].iloc[eve]:.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[2].transAxes, fontsize=12)
        axs[2].set_title('Without Def. or Pretrain\nPrediction')

        # Error
        ER_pretrain = axs[3].scatter(idx[:, 1], idx[:, 0], c=error_nodeform, s=0.0005, cmap=cmap_error,
                            vmin=-5,vmax=5,alpha=1)
        axs[3].text(xpos,ypos, f'max: {np.nanmax(error_nodeform):.3f},\nmin: {np.nanmin(error_nodeform):.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[3].transAxes, fontsize=12)
        axs[3].set_title('Without Def. or Pretrain\nError')

        # Pred_direct
        PR_direct = axs[4].scatter(idx[:, 1], idx[:, 0], c=pred_direct, s=0.0005, cmap=cmap_depth,
                            vmin=0, vmax=10,alpha=1)
        axs[4].text(xpos,ypos, f'max: {np.nanmax(pred_direct):.3f}\nr^2: {eve_perf_direct["r2"].iloc[eve]:.3f}\ng: {eve_perf_direct["g"].iloc[eve]:.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[4].transAxes, fontsize=12)
        axs[4].set_title('With Def. no Pretrain\nPrediction')

        # Error
        ER_direct = axs[5].scatter(idx[:, 1], idx[:, 0], c=error_direct, s=0.0005, cmap=cmap_error,
                            vmin=-5,vmax=5,alpha=1)
        axs[5].text(xpos,ypos, f'max: {np.nanmax(error_direct):.3f},\nmin: {np.nanmin(error_direct):.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[5].transAxes, fontsize=12)
        axs[5].set_title('With Def. no Pretrain\nError')

        # Pred_pretrain
        PR_pretrain = axs[6].scatter(idx[:, 1], idx[:, 0], c=pred_pretrain, s=0.0005, cmap=cmap_depth,
                            vmin=0, vmax=10,alpha=1)
        axs[6].text(xpos,ypos, f'max: {np.nanmax(pred_pretrain):.3f}\nr^2: {eve_perf_pretrain["r2"].iloc[eve]:.3f}\ng: {eve_perf_pretrain["g"].iloc[eve]:.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[6].transAxes, fontsize=12)
        axs[6].set_title('With Def. and Pretrain\nPrediction')

        #Error
        ER_pretrain = axs[7].scatter(idx[:, 1], idx[:, 0], c=error_pretrain, s=0.0005, cmap=cmap_error,
                            vmin=-5,vmax=5,alpha=1)
        axs[7].text(xpos,ypos, f'max: {np.nanmax(error_pretrain):.3f},\nmin: {np.nanmin(error_pretrain):.3f}',
                    horizontalalignment='center', verticalalignment='center',transform=axs[7].transAxes, fontsize=12)
        axs[7].set_title('With Def. and Pretrain\nError')

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
        plt.savefig(f'{MLDir}/model/{reg}/compare/Compare_TPE_{train_size}_{reg}_{str(eve)}.png',
                    dpi=100, bbox_inches='tight', pad_inches=0.1)
        #close figure
        plt.close(fig)

elif mode == 'compare_pygmt':
    for id in ids:
        eve = np.where(eve_id==id)[0][0]
        print(id,'\n',eve)
        #cm to m
        pred_pretrain=pred_depths_pretrain[eve]/100
        pred_direct=pred_depths_direct[eve]/100
        pred_nodeform=pred_depths_nodeform[eve]/100
        true=true_depths[eve]/100
    
        #calculate errors
        error_pretrain = calculate_error(true, pred_pretrain)
        error_direct = calculate_error(true, pred_direct)
        error_nodeform = calculate_error(true, pred_nodeform)

        #remove micro depths for better visualization
        pred_pretrain= np.where(pred_pretrain < 0.1, np.nan, pred_pretrain)
        pred_direct= np.where(pred_direct < 0.1, np.nan, pred_direct)
        pred_nodeform= np.where(pred_nodeform < 0.1, np.nan, pred_nodeform)
        true= np.where(true < 0.1, np.nan, true)
        
        #additional region specific parameters for plotting
        if reg == 'CT':
            fig, axs = plt.subplots(1, 8, figsize=(19,8))
            cbar_ht = 0.02
            # Load the elevation from netcdf grid
            grid = xr.open_dataset('../../data/processed/CT_defbathy.nc',engine='netcdf4')
            subsize = ["7c","18c"]
            pos = "TL"
        elif reg == 'SR':
            fig, axs = plt.subplots(1, 8, figsize=(19,3))
            cbar_ht = 0.04
            # Load the elevation from netcdf grid
            grid = xr.open_dataset('../../data/processed/SR_defbathy.nc',engine='netcdf4')
            subsize = ["6c","5.5c"]
            pos = "TL"
        #get lat long limits from the grid
        ymin = grid['y'].min().values
        ymax = grid['y'].max().values
        xmin = grid['x'].min().values
        xmax = grid['x'].max().values
        #common color maps
        cptfile_bathy = '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/scripts/PaperIIPlots/PaperI/r2/bathy.cpt'                  
        
        #pygmt plot
        fig = pygmt.Figure()
        with fig.subplot(
                        nrows=1,
                        ncols=7,
                        subsize = subsize,
                        sharey = True,
                        frame = ["ag"],
                        clearance="1c"
            ):
            with fig.set_panel(panel=[0,0]): #true inundation depth
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="berlin", series=[0.1,10,0.5],transparency=50,reverse = False,background=True)     
                filter = ~np.isnan(true)          
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=true[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(true):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                # fig.colorbar(cmap=True, position="JBC+o0/1c+w10c/0.5c+h",frame=["a2","x+lDepth", "y+lm"])
            with fig.set_panel(panel=[0,1]): #nodeform inundation depth
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="berlin", series=[0.1,10,0.5],transparency=0,reverse = False,background=True)              
                filter = ~np.isnan(pred_nodeform) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=pred_nodeform[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(pred_nodeform[filter]):.3f}',font="14p,Helvetica-Bold,black",offset="0.15/0",projection='M6c')
                fig.text(position=pos, text=f'r^2:{eve_perf_nodeform["r2"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
                fig.text(position=pos, text=f'g:{eve_perf_nodeform["g"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-1",projection='M6c')
            with fig.set_panel(panel=[0,2]): #nodeform inundation error
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="polar+h0", transparency=0,series=[-5,5,0.5],background=True)
                filter = ~np.isnan(error_nodeform) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=error_nodeform[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(error_nodeform[filter]):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                fig.text(position=pos, text=f'min:{np.nanmin(error_nodeform[filter]):.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
                # fig.colorbar(cmap=True, position="JBC+o0/1c+w10c/0.5c+h",frame=["a2","x+lError", "y+lm"])
            with fig.set_panel(panel=[0,3]): #with deformation inundation depth
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="berlin", series=[0.1,10,0.5],transparency=0,reverse = False,background=True)              
                filter = ~np.isnan(pred_direct) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=pred_direct[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(pred_direct[filter]):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                fig.text(position=pos, text=f'r^2:{eve_perf_direct["r2"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
                fig.text(position=pos, text=f'g:{eve_perf_direct["g"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-1",projection='M6c')
            with fig.set_panel(panel=[0,4]): #with deformation inundation error
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="polar+h0", transparency=0,series=[-5,5,0.5],background=True)
                filter = ~np.isnan(error_direct) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=error_direct[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(error_direct[filter]):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                fig.text(position=pos, text=f'min:{np.nanmin(error_direct[filter]):.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
            with fig.set_panel(panel=[0,5]): #with deformation and pretrain inundation depth
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="berlin", series=[0.1,10,0.5],transparency=0,reverse = False,background=True)              
                filter = ~np.isnan(pred_pretrain) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=pred_pretrain[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(pred_pretrain[filter]):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                fig.text(position=pos, text=f'r^2:{eve_perf_pretrain["r2"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
                fig.text(position=pos, text=f'g:{eve_perf_pretrain["g"].iloc[eve]:.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-1",projection='M6c')
            with fig.set_panel(panel=[0,6]): #with deformation and pretrain inundation error
                #basemap
                cmap = pygmt.makecpt(cmap=cptfile_bathy,continuous=False)
                fig.grdimage(grid['z'], cmap=True, shading=True,region=[xmin,xmax,ymin,ymax],projection='M6c')
                fig.grdcontour(grid['z'], levels=10, pen='0.5p,white', limit=[-100, 0],projection='M6c')
                #parameter     
                pygmt.makecpt(cmap="polar+h0", transparency=0,series=[-5,5,0.5],background=True)
                filter = ~np.isnan(error_pretrain) 
                fig.plot(x=index_map['lon'][filter], y=index_map['lat'][filter],fill=error_pretrain[filter],style='s0.01c',cmap = True,projection='M6c')
                fig.grdcontour(grid=grid['z'], levels=1,limit=[-0.5, 0.5],annotation=False,projection='M6c',pen='0.5p,black')
                fig.text(position=pos, text=f'max:{np.nanmax(error_pretrain[filter]):.3f}',font="14p,Helvetica-Bold,black",projection='M6c',offset="0.15/0")
                fig.text(position=pos, text=f'min:{np.nanmin(error_pretrain[filter]):.3f}',font="14p,Helvetica-Bold,black",offset="0.15/-0.5",projection='M6c')
        fig.savefig(f'{MLDir}/model/{reg}/compare/Compare_TPE_{train_size}_{reg}_{str(eve)}_pygmt.png',dpi=300)
else:
    print('Error: Invalid mode')

