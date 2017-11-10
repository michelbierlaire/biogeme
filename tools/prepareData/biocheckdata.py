# script to check that a biogeme estimation datafile has no error
# Authors: Ricardo Hurtubia and Michel Bierlaire
# Wed Mar 14 08:30:18 2012


import builtins

#from string import *
import string
import sys
import random
import time
import re
import functools
import os

import math

def mean(x):
    sum = 0.0
    for i in x:
        sum += i
    return sum/len(x)
        
def variance(x,mean):
    sum = 0.0
    for i in x:
        sum += (i-mean)*(i-mean)
    return sum / len(x)

def isAscii(s):
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True
    
#from math import *

print(sys.path)

def round_to_n(x, n):
    if not x: return 0
    power = -int(math.floor(math.log10(abs(x)))) + (n - 1)
    factor = (10 ** power)
    return round(x * factor) / factor

filename = os.path.basename(str(sys.argv[1]))
filewithpath = str(sys.argv[1])


argc = len(sys.argv)
if (argc == 1):
        print("Syntax: ",sys.argv[0]," mydata.dat [rowsMerged=1]")
elif (argc == 2):
        mergedRows = 1
else:
        mergedRows = int(sys.argv[2])
        if (mergedRows < 1):
                print("The number of merged rows is incorrect: ",mergedRows)
                sys.exit() 
        else:
                print("Each ",mergedRows, "consecutive rows are merged") ;
Input1 = open(filewithpath, 'r')


htmlfile = filename.replace('.','_')+".html"

print("Check if the file ",filewithpath," is complying with biogeme's requirements.") 
print("Reading data")
data_1 = {}
tab_data = []
headers = []
len_data=0
for line in Input1:
        dataLine = str.rstrip(line)
        data_1[line] = re.compile('\s').split(dataLine)
        row = data_1[line]
#        print(row)
        tab_data.append(row)
        if (len_data == 0):
                print(len(row), "headers: ",row)
                headers = row
        len_data=len_data+1
        if len_data == 500000:        
                print("500000 lines read")
        if len_data == 1000000:        
                print("1000000 lines read")
        if len_data == 1500000:        
                print("1500000 lines read")        
        if len_data == 2000000:        
                print("more than 2000000 lines read")
        
print(" ")

nRows = len_data
nColumns = len(tab_data[0])
print(nRows, "lines")
print(nColumns, "columns")

err=0
for i in range (len(tab_data)):
# for i in range (3):
        if i>0:
                if (mergedRows * len(tab_data[i])) > nColumns:
                        print("Length [",i,"]: ", len(tab_data[i]))
                        print("Length [0]: ", nColumns)
                        print("error in line", i+1," (more columns (",mergedRows * len(tab_data[i]),") than headers (",nColumns,"))")
                        err = 1
                if (mergedRows * len(tab_data[i])) < nColumns:
                        print("error in line", i+1," (less columns (",mergedRows * len(tab_data[i]),") than headers (",nColumns,"))")
                        err = 1
                
                for j in range(len(tab_data[i])):
                        
                        x=tab_data[i][j]
                        try:
                                y=float(x)
                                tab_data[i][j] = y
                        except:        
                                print("error in line", i+1, " (column ",j+1," contains text: ",x,")")
                                err=1

if err==0:
        print("data check finalized, no errors.")
else:
        print("The file does not comply with biogeme's requirements")
        sys.exit()

# Calculate statistics


h = open(htmlfile,'w')
currentTime = time.strftime("%c")

print("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">",file=h)
print("",file=h)
print("<html>",file=h)
print("<head>",file=h)
print("<script src=\"http://transp-or.epfl.ch/biogeme/sorttable.js\"></script>",file=h)
print("<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />",file=h)
print("<title>"+htmlfile+" Statistics "+currentTime+"</title>",file=h)
print("<meta name=\"keywords\" content=\"biogeme, discrete choice, random utility\">",file=h)
print("<meta name=\"description\" content=\"Statistics about "+filewithpath+" ["+currentTime+"]\">",file=h)
print("<meta name=\"author\" content=\"Michel Bierlaire\">",file=h)
print("<style type=text/css>",file=h)
print("<!--table",file=h)
print(".biostyle",file=h)
print("	{font-size:10.0pt;",file=h)
print("	font-weight:400;",file=h)
print("	font-style:normal;",file=h)
print("	font-family:Courier;}",file=h)
print(".boundstyle",file=h)
print("	{font-size:10.0pt;",file=h)
print("	font-weight:400;",file=h)
print("	font-style:normal;",file=h)
print("	font-family:Courier;",file=h)
print("        color:red}",file=h)
print("-->",file=h)
print("</style>",file=h)
print("</head>",file=h)
print("",file=h)
print("<body bgcolor=\"#ffffff\">",file=h)
print("<p>Biogeme home page: <a href='http://biogeme.epfl.ch' target='_blank'>http://biogeme.epfl.ch</a></p>",file=h)
print("<p><a href='http://people.epfl.ch/michel.bierlaire'>Michel Bierlaire</a>, <a href='http://transp-or.epfl.ch'>Transport and Mobility Laboratory</a>, <a href='http://www.epfl.ch'>Ecole Polytechnique F&eacute;d&eacute;rale de Lausanne (EPFL)</a></p>",file=h)
print("<p>This file has automatically been generated on ",file=h)
print(currentTime+"</p>",file=h)

print("<p>Statistics on {}</p>".format(filewithpath),file=h)
print("<p>Total number of data: {}</p>".format(nRows-1),file=h)
print("<p><table border='1'>",file=h)
print("<tr>",file=h)
print("<th>Variable</th>",file=h)
print("<th>Minimum</th>", file=h)
print("<th>Mean</th>", file=h)
print("<th>Variance</th>", file=h)
print("<th>Std dev.</th>", file=h)
print("<th>Maximum</th>", file=h)
print("<th>Nbr of zeros</th>", file=h)
print("<th>Percentage of zeros</th>", file=h)
print("</tr>",file=h)
for c in range(nColumns):
        print("<tr>",file=h)
        print("<td>"+headers[c]+"</td>",file=h)
        theCol = [row[c] for row in tab_data[1:]]
        m = builtins.min(theCol)
        print("<td>"+format(round_to_n(m,3))+"</td>",file=h)
        theMean = mean(theCol)
        m = theMean
        print("<td>"+format(round_to_n(m,3))+"</td>",file=h)
        theVariance = variance(theCol,theMean)
        m = theVariance
        print("<td>"+format(round_to_n(m,3))+"</td>",file=h)
        m = math.sqrt(theVariance)
        print("<td>"+format(round_to_n(m,3))+"</td>",file=h)
        if m == 0:
            print("Variable "+headers[c]+" does not vary in the sample")
        m = max(theCol)
        print("<td>"+format(round_to_n(m,3))+"</td>",file=h)
        mm = theCol.count(0.0)
        print("<td>{}</td>".format(mm),file=h)
        m = 100.0 *float(mm) / float(nRows-1)
        print("<td>"+format(round_to_n(m,3))+"%</td>",file=h)
        print("</tr>",file=h)

print("</table></p>",file=h)
print("</body>",file=h)
print("</html>",file=h)
h.close() ;
print("Statistics are available in "+htmlfile)
