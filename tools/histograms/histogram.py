from operator import itemgetter
from collections import defaultdict
from string import *
from math import floor
import builtins

## Computes the frequency of raw data in order to generate an histogram
# @param binSize The size of the bins in the histogram 
# @param inputFile The name of the file where the raw data can be found. The
# file should contain only numbers.
# @param outputFile The name of the output files when the bin frequencies are reported. The file contains a sequence of pairs of values
# \f[ (x_i,n_i) \f]
# such that \f$n_i\f$ is the number of values \f$x\f$ such that 
# \f[ x_i \leq x < x_{i+1} \f]
# @return The dictionary containing the same information as in the output file.
# 
# Example of an input file:
# \code
#1
#2
#2.1
#3
#3.1
#3.4
#4
#4.3
#4.5
#4.2
# \endcode
# Corresponding output file (binsize = 1):
# \code
#1.0 1
#2.0 2
#3.0 3
#4.0 4
# \endcode
def histogram(binSize,inputfile,outputfile):
    f = open(inputfile,'r')
    data = [[float(x) for x in line.split()] for line in f]
    flatten = [item for sublist in data for item in sublist]
    theMin = builtins.min(flatten)
    theMax = builtins.max(flatten)
    print("Number of data items: {}".format(len(flatten)))
    print("Range of data: [{}-{}]".format(theMin,theMax))
    hist = {}
    for x in flatten:
        key = floor(x / binSize) * binSize
        if key in hist:
            hist[key] += 1
        else:
            hist[key] = 1
    res = sorted(hist.items())
    output = open(outputfile,'w')
    output.write('Value'+' '+'Frequency'+'\n')
    for k,v in res:
        s = str(k) + ' ' + str(v) + '\n'
        output.write(s)
    f.close()
    output.close()
    return res


## Computes the frequency of weighted raw data in order to generate an histogram
# @param binSize The size of the bins in the histogram 
# @param inputFile The name of the file where the raw data can be found. The
# file should contain only numbers.
# @param outputFile The name of the output files when the bin frequencies are reported. The file contains a sequence of pairs of values
# \f[ (x_i,n_i) \f]
# such that \f$n_i\f$ is the number of values \f$x\f$ such that 
# \f[ x_i \leq x < x_{i+1} \f]
# @return The dictionary containing the same information as in the output file.
# 
# Example of an input file:
#   first column: data, second column: weight
# \code
#1    0.6
#2    2.3
#2.1  0.2
#3    1.0
#3.1  0.1
#3.4  5.0
#4    3.2
#4.3  1.1
#4.5  0.76
#4.2  0.2
# \endcode
# Corresponding output file (binsize = 1):
# \code
#1.0 0.6
#2.0 2.5
#3.0 6.1
#4.0 5.26
# \endcode
def weightedhistogram(binSize,inputfile,outputfile):
    f = open(inputfile,'r')
    data = []
    for x in f:
        data.append(x.split())
    hist = {}
    for x in data:
        key = floor(float(x[0]) / binSize) * binSize
        if key in hist:
            hist[key] += float(x[1])
        else:
            hist[key] = float(x[1])
    res = sorted(hist.items())
    output = open(outputfile,'w')
    output.write('Value'+' '+'Frequency'+'\n')
    for k,v in res:
        s = str(k) + ' ' + str(v) + '\n'
        output.write(s)
    f.close()
    output.close()
    return res
