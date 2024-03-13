
filename       = "maxvalfilestems.txt"

with open( filename ) as f:
    lines = f.readlines()
    stems  = [ line.split()[0] for line in lines ]

numazivalfiles = len( stems)
print( numazivalfiles, " files" )

filename = "params_file.txt"

with open( filename ) as f:
    lines = f.readlines()
    inundationareas = [ float( line.split()[0] ) for line in lines]
    maxheights      = [ float( line.split()[1] ) for line in lines]
    corresstems     = [        line.split()[2]   for line in lines]

areaarray     = []
maxharray     = []
readingsarray = []
for iazifile in range( 0, numazivalfiles ):
    stem     = stems[ iazifile ]
    #
    # Find the index of the parameter arrays
    #
    index = -1
    for iparamline in range( 0, len(corresstems) ):
        if ( stem == corresstems[ iparamline ] ):
            index = iparamline
    if ( index == -1 ):
        print ("No index found for stem ", stem )
        exit()
    areavalue = inundationareas[ index ]
    maxhvalue = maxheights[ index ]
    areaarray.append( areavalue )
    maxharray.append( maxhvalue )
    #
    readingsfile = "../" + stem + "_maxvals.dat"
    with open( readingsfile ) as f:
        lines = f.readlines()
        sensorreadings = [ float( line.split()[1] ) for line in lines]
        print ("file ", readingsfile )
        print ("read sensorreadings", sensorreadings )
    readingsarray.append( sensorreadings )

for isensor in range(0,87):
    filename = "sensor" + "{:05d}".format(  isensor ) + ".dat"
    f = open( filename, "w" )
    for i in range( 0,len( readingsarray ) ):
        sensorreadings = readingsarray[i]
        maxh           = maxharray[i]
        area           = areaarray[i]
        print ("i", i, "sensorreadings ", sensorreadings )
        sensorval      = sensorreadings[isensor]
        line  = "{:.4f}".format(      maxh ).rjust(10)  + " "
        line += "{:.4f}".format(      area ).rjust(10)  + " "
        line += "{:.4f}".format( sensorval ).rjust(10)  + " \n"
        print ( line )
        f.write( line )
    f.close()
        
