import numpy as np
import matplotlib.pyplot as plt

infile = "catania_ts_sum.txt"
figurepng = "inundation_area_histogram.png"
figurepdf = "inundation_area_histogram.pdf"

with open(infile) as f:
    lines = f.readlines()
    ihstr = [ float( line.split()[0] ) for line in lines]

iharr = np.asarray(ihstr)
bins  = np.arange( 0.0, 30.0, 1.0)

fig=plt.figure(figsize=(6,5))
n, bins, patches = plt.hist(iharr, bins = bins, facecolor='red', align='mid')

# plt.ylim([0.00,10.0])
# plt.xlim([3.45,5.55])
# plt.xticks(bins)
plt.xlabel('Inundated area (km²)', fontsize = 13)
plt.ylabel('Number of scenarios', fontsize = 13)
plt.title(r'Histogram of inundation areas (Catania)', fontsize=14)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.savefig( figurepng , facecolor='w', edgecolor='w', format='png')
plt.savefig( figurepdf , facecolor='w', edgecolor='w', format='pdf')

plt.show()
