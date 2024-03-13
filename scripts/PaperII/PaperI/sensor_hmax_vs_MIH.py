
import numpy as np
import matplotlib.pyplot as plt

infile  = "sensor00017.dat"

with open(infile) as f:
    lines = f.readlines()
    MIHstr  = [ float( line.split()[0] ) for line in lines]
    AREAstr = [ float( line.split()[1] ) for line in lines]
    maxhstr = [ float( line.split()[2] ) for line in lines]

MIHarr  = np.asarray( MIHstr )
AREAarr = np.asarray( AREAstr )
maxharr = np.asarray( maxhstr )

# plot
fig, ax = plt.subplots()

plt.xlim([0.00,10.0])
plt.ylim([0.00,10.0])
plt.scatter(maxharr, MIHarr, s = 0.1)

plt.show()

