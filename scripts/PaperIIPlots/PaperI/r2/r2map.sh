#!/bin/sh
# Colors on https://docs.generic-mapping-tools.org/dev/_images/GMT_RGBchart.png
# set -x
scriptname=./r2map.sh
if [ $# != 1 ]
then
  echo
  echo "USAGE: "
  echo "$scriptname   topor2file "
  echo
  echo
  echo "$scriptname   r2_topo_train.nc  "
  echo "$scriptname   r2_topo_test.nc  "
  echo
  exit 1
fi
#
percentfile=$1
grdCfile=C_CT.grd
percentstem=`basename $percentfile .nc`
#
filestem=${percentstem}_Conly
#
gmtbin=/opt/gmt/gmt4/bin
gmt5bin=/opt/gmt/gmt5/bin
gmt6bin=/opt/gmt/gmt6/bin

${gmt6bin}/gmtset BASEMAP_TYPE FANCY \
       TICK_PEN 1 FRAME_PEN 1 D_FORMAT %.12lg \
       PLOT_DEGREE_FORMAT +D.xx

fil=${filestem}.ps
proj=-JM9

if test -r $fil
then
  rm $fil
fi

lonscale=15.039
latscale=37.475
scalelenkm=2.0
Clatmin=`ncdump $grdCfile | grep y:actual | sed 's_,_ _g' | awk '{print $3}'`
Clatmax=`ncdump $grdCfile | grep y:actual | sed 's_,_ _g' | awk '{print $4}'`
Clonmin=`ncdump $grdCfile | grep x:actual | sed 's_,_ _g' | awk '{print $3}'`
# Clonmax=`ncdump $grdCfile | grep x:actual | sed 's_,_ _g' | awk '{print $4}'`
Clonmax=15.11

reg=-R${Clonmin}/${Clonmax}/${Clatmin}/${Clatmax}
echo $reg
# cptfile=r2only.cpt
cptfile=r2plusbathy.cpt
grid=${grdCfile}
${gmt6bin}/grdimage  ${percentfile}  -C${cptfile}  $reg $proj -P -K -V        -Sb > $fil
${gmtbin}/grdmath  $grid  1.0 MUL = tmp.nc
${gmt6bin}/grdcontour tmp.nc -C5.0             $reg $proj    -K -O -Wthin,beige     >> $fil
${gmt6bin}/grdcontour tmp.nc -C10000.0           $reg $proj    -K -O -Wthin,black     >> $fil
${gmt6bin}/pscoast $reg $proj -Df -N1/2/0/0/0 -Ba0.04f0.02/a0.05f0.025/neWS  -O     -V  \
                 -Lf${lonscale}/${latscale}/${latscale}/${scalelenkm}k  >> $fil
/opt/gmt/gmt6/bin/psconvert $fil -TG -A
/opt/gmt/gmt6/bin/psconvert $fil -Tf -A
