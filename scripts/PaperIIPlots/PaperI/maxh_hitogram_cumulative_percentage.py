import numpy as np
import matplotlib.pyplot as plt

number = "00039"
sensor = "sensor" + number
infile = sensor + ".dat"
figurepng = sensor + "_histogram_cumulative_percentage.png"
figurepdf = sensor + "_histogram_cumulative_percentage.pdf"

with open(infile) as f:
    lines = f.readlines()
    ihstr = [ float( line.split()[2] ) for line in lines]

numelmnts = len( ihstr )
iharr = np.asarray(ihstr)
bins  = np.arange( 0.0, 5.0, 0.25)
binslabel  = np.arange( 0.0, 5.0, 1.00)

fig=plt.figure(figsize=(4.0,5))
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
            bins = bins, align='mid', facecolor = 'Blue',
                             label = 'Percentage of Scenarios',
            cumulative=False)


plt.xticks(binslabel)
# plt.legend(loc = 'right')
plt.xlabel('maxh (m)', fontsize = 13)
plt.ylabel('Percentage of scenarios', fontsize = 13)
titlestring = "sensor " + number
plt.title(titlestring, fontsize=14)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.savefig( figurepng , facecolor='w', edgecolor='w', format='png')
plt.savefig( figurepdf , facecolor='w', edgecolor='w', format='pdf')

plt.show()
