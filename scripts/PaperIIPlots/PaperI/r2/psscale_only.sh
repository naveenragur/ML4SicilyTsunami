#!/bin/sh
cptfile=r2only.cpt
fil=psscale_r2only.ps
gmt4bin=/opt/gmt/gmt4/bin
gmt5bin=/opt/gmt/gmt5/bin
gmt6bin=/opt/gmt/gmt6/bin
#
incha=0.2
inchf=0.1
${gmt6bin}/psscale -D1.00/5.0/8.0/1.0 -C${cptfile} \
         -Ba${incha}f${inchf}::/:"":       > $fil
#
/opt/gmt/gmt6/bin/psconvert $fil -TG -A
/opt/gmt/gmt6/bin/psconvert $fil -Tf -A
