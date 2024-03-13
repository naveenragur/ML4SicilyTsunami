import numpy as np
import matplotlib.pyplot as plt

infile = "catania_ts_sum.txt"
figurepng = "MIH_histogram_percentage.png"
figurepdf = "MIH_histogram_percentage.pdf"

with open(infile) as f:
    lines = f.readlines()
    ihstr = [ float( line.split()[1] ) for line in lines]

numelmnts = len( ihstr )
iharr = np.asarray(ihstr)
bins  = np.arange( 0.0, 12.0, 1.0)

fig=plt.figure(figsize=(5,5))
n, bins, patches = plt.hist(iharr,
            weights=100.0 * np.ones(len(iharr)) / len(iharr),
            bins = bins, facecolor='red', align='mid')

# plt.ylim([0.00,10.0])
# plt.xlim([3.45,5.55])
plt.xticks(bins)
plt.xlabel('Maximum inundation height (m)', fontsize = 13)
plt.ylabel('Percentage of scenarios', fontsize = 13)
plt.title(r'Histogram of MIH (Catania)', fontsize=14)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.savefig( figurepng , facecolor='w', edgecolor='w', format='png')
plt.savefig( figurepdf , facecolor='w', edgecolor='w', format='pdf')

plt.show()
