#!/usr/bin/env python3
#
# We specify --gridfile  --r2file  --outfile  --ncfile
# The r2file is just the numbers - no coordinates so we need to
# output this with the same references as gridfile
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
def one_netcdf_file_write( lats, lons, valuearray, filename, varname ):
    exists = os.path.isfile( filename )
    if exists:
        os.remove( filename )

    f = netCDF4.Dataset(filename, 'w', format="NETCDF3_CLASSIC")
    f.createDimension('lat', len(lats))
    f.createDimension('lon', len(lons))
    f.description         = 'arbitrary netcdf file'
    f.history             = 'SJG 2020/06/22'
    f.source              = 'PoE_maps.py'
    lats_out              = f.createVariable('lat', float, ('lat',), zlib=True)
    lats_out.long_name    = "latitude"
    lats_out.units        = 'degree_north'
    dellats               = lats[ 1] - lats[0]
    latlo                 = lats[ 0] - dellats*0.5
    lathi                 = lats[-1] + dellats*0.5
    lats_out.actual_range = latlo, lathi
    lats_out[:]           = lats[:]
    lons_out              = f.createVariable('lon', float, ('lon',), zlib=True)
    lons_out.long_name    = "longitude"
    lons_out.units        = 'degree_east'
    dellons               = lons[ 1] - lons[0]
    lonlo                 = lons[ 0] - dellons*0.5
    lonhi                 = lons[-1] + dellons*0.5
    lons_out.actual_range = lonlo, lonhi
    lons_out[:]           = lons[:]
    z_out                 = f.createVariable(varname, "f4", ('lat','lon'), zlib=True)
    z_out[:]              = valuearray[:]
    f.close()
#
    return

#==========================================================================
def process_grid_r2file( gridfile, r2file, outfile ):
    file_exist( gridfile )
    file_exist( r2file )
    NetCDFu       = netcdf_dataset( gridfile )
    try:
        lonsu            = NetCDFu.variables['lon']
    except:
        try:
            lonsu            = NetCDFu.variables['x']
        except:
            print ('exit ; Failed to read lons from file_u ', gridfile )
            exit()

    try:
        latsu            = NetCDFu.variables['lat']
    except:
        try:
            latsu            = NetCDFu.variables['y']
        except:
            print ('exit ; Failed to read lats from file_u ', gridfile )
            exit()

    orig_topo       = NetCDFu.variables['z'][:,:]
    outf            = orig_topo

    NetCDFr       = netcdf_dataset( r2file )
    r2grid        = NetCDFr.variables['r-squared'][:,:]

    #
    # Now all of the r2 values are between zero and 1 so without
    # "hurting" anything, we can put all values between
    # zero and 1.01 in orig_topo to 1.01
    #
    indices = np.logical_and( outf < 1.01, outf > 0.0  )
    outf[ indices                  ] = 1.01
    indices = np.logical_and( r2grid < 1.01, r2grid > 0.0  )
    outf[ indices                  ] = r2grid[ indices ]
    #
    varname = 'r2topo'
    one_netcdf_file_write( latsu, lonsu, outf, outfile, varname )
    #

    # lon  = lonsu[row]
    # lat  = latsu[col]
    # height, inrange = nearest_latlonfunc( lat, lon, latsu, lonsu, orig_topo )
    irc = 0
    return irc

if __name__ == "__main__":
    #
    scriptname = sys.argv[0]
    numarg     = len(sys.argv) - 1
    text       = 'Specify --gridfile [fil] --r2file [fil] --outfile [fil]'
    parser     = argparse.ArgumentParser( description = text )
    parser.add_argument("--gridfile", help="input_topo", default=None, required=True )
    parser.add_argument("--r2file", help="r2file", default=None, required=True )
    parser.add_argument("--outfile", help="outfile", default=None, required=True )

    args = parser.parse_args()

    gridfile = args.gridfile
    r2file   = args.r2file
    outfile  = args.outfile

    irc = process_grid_r2file( gridfile, r2file, outfile )    
    if ( irc != 0 ):
        print ("Error from process_grid_r2file ", irc )
        exit()


