import numpy as np
import matplotlib.pyplot as plt

infile = "catania_ts_sum.txt"
figurepng = "inundation_area_histogram_cumulative_percentage.png"
figurepdf = "inundation_area_histogram_cumulative_percentage.pdf"

with open(infile) as f:
    lines = f.readlines()
    ihstr = [ float( line.split()[0] ) for line in lines]

numelmnts = len( ihstr )
iharr = np.asarray(ihstr)
bins  = np.arange( 0.0, 20.0, 1.0)
# bins  = np.arange( 0.0, 20.0, 0.25)

fig=plt.figure(figsize=(7,5))
n, bins, patches = plt.hist(iharr,
            weights=100.0 * np.ones(len(iharr)) / len(iharr),
            bins = bins, align='mid',
            histtype='step', label = 'Cumulative',
            cumulative=True)
plt.hist(iharr,
            weights=100.0 * np.ones(len(iharr)) / len(iharr),
            bins = bins, align='mid',
            histtype='step', label = 'Reverse Cumulative',
            cumulative=-1)
plt.hist(iharr,
            weights=100.0 * np.ones(len(iharr)) / len(iharr),
            bins = bins, align='mid', facecolor = 'Red',
                             label = 'Percentage of Scenarios',
            cumulative=False)


# plt.ylim([0.00,10.0])
# plt.xlim([3.45,5.55])
plt.xticks(bins)
plt.legend(loc = 'right')
plt.xlabel('Inundated area (km²)', fontsize = 13)
plt.ylabel('Percentage of scenarios', fontsize = 13)
plt.title(r'Histogram of inundation area (Catania)', fontsize=14)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.savefig( figurepng , facecolor='w', edgecolor='w', format='png')
plt.savefig( figurepdf , facecolor='w', edgecolor='w', format='pdf')

plt.show()
