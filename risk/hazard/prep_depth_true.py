import numpy as np
import dask.array as da

###################################
# SIMULATION RESULTS
####################################

#Load the NumPy file using Dask, and specify chunk size
true_d = np.load('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/CT/multifoldMC/PTHA/true_d_53550.npy')
num_events, num_sites = true_d.shape

#Flatten the true_d array and filter non zero depths using Dask
true_d_dask = da.from_array(true_d,chunks="auto").astype(np.float16).flatten()
print(true_d_dask.shape)
non_zero_idx = true_d_dask > 10
true_d_dask_filtered = da.take(true_d_dask, non_zero_idx, axis=0)
print(true_d_dask_filtered.shape)

#Save the flat files
# np.save('./true_d_unfiltered.npy', true_d_dask)  #large file not saved unless needed
np.save('./true_d_dask.npy', true_d_dask_filtered)
# np.save('./true_non_zero_idx.npy', non_zero_idx) #large file not saved unless needed
del true_d_dask, true_d_dask_filtered, true_d

#Repeat each event(id) num_sites times and filter non zero events using Dask
event_id_dask = da.arange(0, num_events, chunks="auto").astype(np.uint16)
event_id_dask = da.repeat(event_id_dask, num_sites).flatten()
event_id_dask_filtered = da.take(event_id_dask, non_zero_idx, axis=0)
print(event_id_dask_filtered.shape)

#Save the flat files
np.save('./event_id_true.npy', event_id_dask_filtered)
del event_id_dask, event_id_dask_filtered

#Repeat each sites(id) nevents times and filter non zero sites using Dask
site_id_dask = da.arange(0, num_sites, chunks="auto").astype(np.uint32)
site_id_dask = da.tile(site_id_dask, num_events).flatten()
site_id_dask_filtered = da.take(site_id_dask, non_zero_idx, axis=0)
print(site_id_dask_filtered.shape)

#Save the flat files
np.save('./site_id_true.npy', site_id_dask_filtered)



