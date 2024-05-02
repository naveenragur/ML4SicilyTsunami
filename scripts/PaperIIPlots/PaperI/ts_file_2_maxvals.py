#!/usr/bin/env python3
import os
import obspy
import netCDF4
from netCDF4 import Dataset
import sys
import argparse

#==========================================================================
def file_exist(file):
    """Check if a file (with full path) exist"""
    if not os.path.isfile(file):
        print("File: ",file," does not exist. Bailing out ...")
        exit()

##################################################

scriptname = sys.argv[0]
numarg     = len(sys.argv) - 1
text       = "Specify -t tsfile_full_path -s outputstem "
parser     = argparse.ArgumentParser( description = text )
parser.add_argument("--tsfile_full_path",  "-t", help="tsfile_full_path", default=None, required=True )
parser.add_argument("--outputstem",  "-s", help="outputstem", default=None, required=True )

args = parser.parse_args()

tsfile_full_path = args.tsfile_full_path
outputstem       = args.outputstem

file_exist( tsfile_full_path )

rootgrp  = Dataset (tsfile_full_path)
# print (rootgrp.data_model)
# print (rootgrp.dimensions)
lonarray = rootgrp.variables["longitude"][:]
latarray = rootgrp.variables["latitude"][:]
timearray = rootgrp.variables["time"][:]
eta_array = rootgrp.variables["eta"][:]
rootgrp.close()
#
outputfile = outputstem + '_coords.dat'
exists = os.path.isfile( outputfile )
if exists:
    print ("Deleting file: ", outputfile  )
    os.remove( outputfile )
else:
    print ("Writing new file: ", outputfile  )

f = open ( outputfile , 'w' )

for i in range( 0 , len(lonarray) ):
    lonstring = "{:.4f}".format( lonarray[i] )
    latstring = "{:.4f}".format( latarray[i] )
    line  = "{:03d}".format(       i ).rjust(4)   + " "
    line += lonstring.rjust(10) + " " + latstring.rjust(10) + "\n"
    f.write( line )
    print ( lonstring.rjust(10)  , latstring.rjust(10)  )

f.close()

outputfile = outputstem + '_maxvals.dat'
exists = os.path.isfile( outputfile )
if exists:
    print ("Deleting file: ", outputfile  )
    os.remove( outputfile )
else:
    print ("Writing new file: ", outputfile  )

f = open ( outputfile , 'w' )

for sensorindex in range( 0, len(lonarray) ):
    maxval = max( eta_array[:,sensorindex] )
    line  = "{:03d}".format(       sensorindex ).rjust(4)   + " "
    line += "{:.5f}".format( maxval ).rjust(10)  + "\n"
    f.write( line )
    # print ( sensorindex, maxval )
#for i in range( 0, len(timearray) ):
#    timestring = "{:.4f}".format( timearray[i] )
#    for sensorindex in range( 0, len(lonarray) ):
#        timestring = timestring + " " + "{:.4f}".format(  eta_array[i,sensorindex] )
#    timestring = timestring + " \n" 
#    print ( timestring )
#    f.write( timestring )

f.close()

exit()
