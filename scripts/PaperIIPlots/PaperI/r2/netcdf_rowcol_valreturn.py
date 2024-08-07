#!/usr/bin/env python3
#
# We specify --row  --col  --varname  --ncfile
# and we simply write out the lat, lon, and value
#
import sys
import os
import numpy    as np
import numpy.ma as ma
import argparse
import netCDF4
from   netCDF4 import Dataset as netcdf_dataset
#==========================================================================
#
# returns funcval, inrange for lat, lon given lats, lons, func
# default funcval = -1, inrange = 0 for arbtrary lat, lon
#
def nearest_latlonfunc( lat, lon, lats, lons, z ):
    funcval = -1.0
    inrange = 0
    dellat  = lats[1] - lats[0]
    dellon  = lons[1] - lons[0]
    latmin  = lats[0]  - dellat
    latmax  = lats[-1] + dellat
    lonmin  = lons[0]  - dellon
    lonmax  = lons[-1] + dellon
    if ( lat < latmin ):
      return funcval, inrange
    if ( lat > latmax ):
      return funcval, inrange
    if ( lon < lonmin ):
      return funcval, inrange
    if ( lon > lonmax ):
      return funcval, inrange
    inrange = 1
    ilat = ( np.abs( lats[:] - lat ) ).argmin()
    ilon = ( np.abs( lons[:] - lon ) ).argmin()
    funcval = z[ilat,ilon]
    string = 'funcval = ' + str(funcval)
    # print ( string )
    return funcval, inrange

#==========================================================================
def file_exist(file):
    """Check if a file (with full path) exist"""
    if not os.path.isfile(file):
        print("File: ",file," does not exist. Bailing out ...")
        exit()

#==========================================================================
def read_elevations_rowcol( row, col, origbathyfile ):
    file_exist( origbathyfile )
    NetCDFu       = netcdf_dataset( origbathyfile )
    try:
        lonsu            = NetCDFu.variables['lon']
    except:
        try:
            lonsu            = NetCDFu.variables['x']
        except:
            print ('exit ; Failed to read lons from file_u ', origbathyfile )
            exit()

    try:
        latsu            = NetCDFu.variables['lat']
    except:
        try:
            latsu            = NetCDFu.variables['y']
        except:
            print ('exit ; Failed to read lats from file_u ', origbathyfile )
            exit()

    orig_topo       = NetCDFu.variables['z'][:,:]

    lon  = lonsu[row]
    lat  = latsu[col]
    height, inrange = nearest_latlonfunc( lat, lon, latsu, lonsu, orig_topo )
    return lat, lon, height, inrange

if __name__ == "__main__":
    #  print("Hello, World!")
    scriptname = sys.argv[0]
    numarg     = len(sys.argv) - 1
    text       = 'Specify --lat [lat} --lon [lon]  --ncfile [ncfile]'
    parser     = argparse.ArgumentParser( description = text )
    parser.add_argument("--ncfile", help="input_nc_file", default=None, required=True )
#    parser.add_argument("--lat", help="latitude", default=None, required=True )
#    parser.add_argument("--lon", help="longitude", default=None, required=True )
    parser.add_argument("--row", help="row", default=None, required=True )
    parser.add_argument("--col", help="col", default=None, required=True )

    args = parser.parse_args()

    ncfile = args.ncfile
#    lat    = float( args.lat )
#    lon    = float( args.lon )
    row    = int( args.row )
    col    = int( args.col )

    lat, lon, height, inrange = read_elevations_rowcol( row, col, ncfile )    
    if ( inrange == 0 ):
        print ("row,col out of range")
        exit()
    lonst="{:.5f}".format(  lon ).rjust(10)
    latst="{:.5f}".format(  lat ).rjust(9)
    heist="{:.5f}".format(  height ).rjust(9)
    print ( latst, lonst, heist )



