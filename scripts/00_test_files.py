import os
import sys
import numpy as np

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{size}.txt', dtype='str')

#print message
print(f'*** Creating test files for {reg} region with {size} events')
print(f'*** {len(event_list)} events in total')