import numpy as np
import dask.array as da
import sys

####################################
# EMULATION RESULTS
####################################

#Load the NumPy file using Dask, and specify chunk size
size =  sys.argv[1]
pred_d = np.load(f'/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/CT/sigmaMC/PTHA/pred_d_{size}_direct.npy')
num_events, num_sites = pred_d.shape

#Flatten the pred_d array and filter non zero depths using Dask
pred_d_dask = da.from_array(pred_d,chunks="auto").astype(np.float16).flatten()
print(pred_d_dask.shape)
non_zero_idx =  pred_d_dask > 10
pred_d_dask_filtered = da.take(pred_d_dask, non_zero_idx, axis=0)
print(pred_d_dask_filtered.shape)

#Save the flat files
# np.save(f'./pred_d_unfiltered_{size}.npy', pred_d_dask) #large files not saved
np.save(f'./pred_d_dask_{size}_mean.npy', pred_d_dask_filtered)
# np.save(f'./pred_non_zero_idx_{size}.npy', non_zero_idx) #large files not saveds
del pred_d_dask, pred_d_dask_filtered, pred_d

#Repeat each event(id) num_sites times and filter non zero events using Dask
event_id_dask = da.arange(0, num_events, chunks="auto").astype(np.uint16)
event_id_dask = da.repeat(event_id_dask, num_sites).flatten()
event_id_dask_filtered = da.take(event_id_dask, non_zero_idx, axis=0)
print(event_id_dask_filtered.shape)

#save the flat files
np.save(f'./event_id_pred_{size}_mean.npy', event_id_dask_filtered)
del event_id_dask, event_id_dask_filtered

#Repeat each sites(id) nevents times and filter non zero sites using Dask
site_id_dask = da.arange(0, num_sites, chunks="auto").astype(np.uint32)
site_id_dask = da.tile(site_id_dask, num_events).flatten()
site_id_dask_filtered = da.take(site_id_dask, non_zero_idx, axis=0)
print(site_id_dask_filtered.shape)

#save the flat files
np.save(f'./site_id_pred_{size}_mean.npy', site_id_dask_filtered)





