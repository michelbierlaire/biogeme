# script to extract a sample from a data set
# Author: Michel Bierlaire
# Sun Nov 30 22:00:11 2014

import sys
import random


filename = str(sys.argv[1])
f = open(str(sys.argv[1]), 'r')
data_filename = 'sampled_'+filename
rest_filename = 'notsampled_'+filename
d = open(data_filename,'w') 
r = open(rest_filename,'w') 

percent = float(str(sys.argv[2])) / 100.0
print("Sample ",sys.argv[2],"% of the data")
print("Percentage: ",percent)
first = 1
total = 0 ;
accepted = 0 ;
for line in f:
    if first:
        first = 0
        print(line,end='',file=d) 
        print(line,end='',file=r) 
    else:
        total += 1 
        rdraw = random.random() 
        if (rdraw <= percent):
            accepted += 1
            print(line,end='',file=d) 
        else:
            print(line,end='',file=r) 
f.close()
d.close()
r.close()
print("Actual sample:         ", 100.0 * accepted/total, "%") 
print("Sampled data file:     ",data_filename)
print("Not sampled data file: ",rest_filename)
