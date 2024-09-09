#given a source/dest dir and folder name list do copy

import os 
import shutil

# #source dir
src = f'/nas/eni01/cat/CT_SR_v1.1/PS_manning003/'
dest = f'/scratch/users/abbate/ML4SicilyTsunami/data/simu/PS_manning003_test/'

# #file with list of folders to copy
file = f'/scratch/users/abbate/ML4SicilyTsunami/data/events/sample_events1212.txt'

# #read file
with open(file) as f:
    content = f.readlines()
content = [x.strip() for x in content]

# #copy folders
for folder in content[:]:
     # print(src+folder,'to', dest+folder)
    shutil.copytree(src+folder, dest+folder)
#   #rename a file in the folder: E01964N3926E02184N3685-PS-Str_PNo_Var-M792_E02117N3755_S000_SR_defbathy.nc to SR_defbathy.nc
    os.rename(dest+folder+'/'+ folder + '_SR_defbathy.nc', dest+folder+'/SR_defbathy.nc')
    

#import os
#import shutil

# Multiple source paths
#src_paths = [
#    '/nas/eni01/cat/CT_SR_v1.1/PS_manning003/',
#    '/nas/eni01/cat/CT_SR_v1.1/PS_4-8_manning003/',
#    '/nas/eni01/cat/CT_SR_v1.1/BS_manning003/',
#    '/nas/eni01/cat/CT_SR_v1.1/BS_4-8_manning003/'
#]

#destination = ['/scratch/users/abbate/ML4SicilyTsunami/data/simu/PS_manning003',
#                '/scratch/users/abbate/ML4SicilyTsunami/data/simu/PS_4-8_manning003',
#                '/scratch/users/abbate/ML4SicilyTsunami/data/simu/BS_manning003',
#                '/scratch/users/abbate/ML4SicilyTsunami/data/simu/BS_4-8_manning003']

#read folder list
#eventfile = f'/scratch/users/abbate/ML4SicilyTsunami/data/events/sample_events53550.txt'

# print(folders[0:10])

# List of files to copy
#file_list = [
#    'grid0_ts.nc',
#    'CT_flowdepth.nc',
#    'SR_flowdepth.nc',
#    'CT_deformation.nc',
#    'SR_deformation.nc'
#]

#def copy_files(src_path, dest_path):
    # folders = os.scandir(src_path) #if we want to copy all folders
    # folders = [folder.name for folder in folders if folder.is_dir() and '.listing' not in folder.name]
    # folders = [folder.name for folder in folders if folder.is_dir() and 'PS_all' not in folder.name]

    #read file for specific event folders
#    with open(eventfile) as f:
#        content = f.readlines()
#    folders = [x.strip() for x in content]

#    for index, folder in enumerate(folders):
#        print('Progress:', round((index / len(folders)) * 100, 2), '%', end='\r')
#        folder_dest_path = os.path.join(dest_path, folder)
#        #check if folder exists in src path
#        if not os.path.exists(os.path.join(src_path, folder)):
            # print('Folder', folder, 'does not exist in', src_path)
#            continue
#        else:
#            os.makedirs(folder_dest_path, exist_ok=True)
#            files = os.scandir(os.path.join(src_path, folder))
#            for file in files:
#                if file.name in file_list and file.is_file():
#                    shutil.copy(file.path, folder_dest_path)
                    # print('Copying', file.name, 'to', folder_dest_path)     
            
        

# Iterate over each source path
# for src_path in src_paths:
    # copy_files(src_path, destination)
    # print('Done copying files of', src_path)

#copy_files(src_paths[0], destination[0])
#print('Done copying files of', src_paths[0])
# copy_files(src_paths[1], destination[1])
# print('Done copying files of', src_paths[1])
#copy_files(src_paths[2], destination[2])
#print('Done copying files of', src_paths[2])
#copy_files(src_paths[3], destination[3])
#print('Done copying files of', src_paths[3])
# print('Done copying files')


