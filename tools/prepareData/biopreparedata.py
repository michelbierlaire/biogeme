# script to prepare a CSV data file in a format requested by biogeme
# Each column containing strings is coded with numbers
# Author: Michel Bierlaire
# Wed Mar 28 11:27:28 2012

import sys

replaceBlanksBy = '_'
stringDelimiters = '"'
defaultCoding='99999'

filename = str(sys.argv[1])

# Count the number semicolons
with open(filename) as f:
    text = f.read().strip()
    nbrSemiColons = text.count(';')


# Count the number of commas
with open(filename) as f:
    text = f.read().strip()
    nbrCommas = text.count(',')

# It is guessed what separator is used for CSV
if (nbrCommas > nbrSemiColons):
    comma = ','
else:
    comma = ';'

print("Comma: ",comma)
print("String delimiters: ",stringDelimiters)

f = open(str(sys.argv[1]), 'r')
legend_filename = 'legend_'+filename 
data_filename = 'biogeme_'+filename

first = 1
entries = []
columns = {} ;
rowNumber = 1
error = 0
for line in f:
    if first:
        first = 0
        firstRow = line.rstrip()
        datafirstRow = firstRow.split(comma)
        for i in range(0,len(datafirstRow)):
            columns[datafirstRow[i]] = i ;
        
        print("Number of headers: ",len(datafirstRow))
        print(datafirstRow) ;
    else:
        rowNumber += 1
        print("Read row ",rowNumber)
        cleanedRow = line.rstrip()
        row = cleanedRow.split(comma)
        if (len(row) != len(datafirstRow)):
            print("Error: ",len(row)," entries in row ",rowNumber, " instead of ",len(datafirstRow))
            error = 1
        entries.append(row) 
    if (error):
        print("Error while reading row",rowNumber)

f.close()

if (error):
    print("Program interrupted due to errors in the input file")
    exit()

stringColumns = [0 for i in range(len(entries[0]))] ;
# Check the first row of data to identify strings
for i,x in enumerate(entries[0]):
    try:
        y = float(x) 
    except:
        stringColumns[i] = 1
        print("Column ",i," is a string: ",x)

coltranslate = {}
for col in range(len(entries[0])):
    if (stringColumns[col]): 
        coltranslate[col] = {} ;
        t = set() 
        for row in entries:
            t.add(row[col])
        code = 0
        for x in t:
            coltranslate[col][x] = code
            code += 1
    
b = open(data_filename,'w')

missingValues = 0

for l in datafirstRow:
    print(l.replace(' ',replaceBlanksBy).replace('.',replaceBlanksBy).replace(stringDelimiters,''),end='\t',file=b)
print(file=b)
for row in entries:
    for col in range(len(row)):
        if (stringColumns[col]): 
            print(coltranslate[col][row[col]],end='\t',file=b)
        else:
            try:
                y = float(row[col])
                print(row[col].replace(' ',''),end='\t',file=b)
            except:
                missingValues += 1
                print(defaultCoding,end='\t',file=b)
                
    print(file=b)
b.close()        
        
l = open(legend_filename,'w') 

for col,legend in coltranslate.items():
    print("+++++++++++++++++++++++++",file=l) 
    print("Legend for column ",datafirstRow[col].replace(' ',replaceBlanksBy).replace('.',replaceBlanksBy).replace(stringDelimiters,''),file=l)
    print("+++++++++++++++++++++++++",file=l) 
    for key in sorted(legend,key=legend.get):
        print(legend[key],":\t",key,file=l)
l.close()
print("{} missing value(s), replaced by {}".format(missingValues,defaultCoding)) 
print("Biogeme data file: ",data_filename)
print("Legend:            ",legend_filename)
print("It is recommended to run 'biocheckdata ",data_filename,"'")
