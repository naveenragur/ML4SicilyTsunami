#!/bin/sh
python grid_and_r2fileread.py    \
                   --gridfile     C_CT.grd    \
                   --r2file       r_squared_train.nc   \
                   --outfile      r2_topo_train.nc
python grid_and_r2fileread.py    \
                   --gridfile     C_CT.grd    \
                   --r2file       r_squared_test.nc   \
                   --outfile      r2_topo_test.nc
