##
# @file bioMatrix.py
# @author  Michel Bierlaire 
#

## @brief This class implements a matrix object designed to store the
# variance covariance matrix. 
class bioMatrix(object):
    ## Constructor
    # @param dim Dimension of the (square) matrix
    # @param names Array of dimension dim containing the names of the parameters
    # @param values Two-dimensional array of dimension dim x dim containing the entries of the matrix
    # @details Example for the diagonal matrix 
    # \f[\begin{array}{cccc} 7 & 0 & 0 & 0 \\ 0 & 7 & 0 & 0 \\ 0 & 0 & 7 & 0 \\ 0 & 0 & 0 & 7 \end{array}\f]
    # @code
    # names = ["ASC_TRAIN","B_TIME","B_COST","ASC_CAR"]
    # values = [[7.0,0.0,0.0,0.0],[0.0,7.0,0.0,0.0],[0.0,0.0,7.0,0.0],[0.0,0.0,0.0,7.0]]
    # theMatrix = bioMatrix(4,names,values)
    # @endcode
    def __init__(self, dim, names, values):
        ## Dimension of the (square) matrix
        self.dim = dim
        self.names = names
        j = 0 
        self.keys = {}
        for i in names:
            self.keys[i] = j
            j += 1 
        # initialize matrix and fill with zeroes
        self.matrix = []
        for i in names:
            ea_row = []
            for j in range(dim):
                ea_row.append(values[self.keys[i]][j])
            self.matrix.append(ea_row)
 
    ## Set an entry of the matrix. If it is an off-diagonal entry, the
    ## symmetric entry is set to the same value to maintain the
    ## symmetry of the matrix.
    # @param rowname Name of the row
    # @param colname Name of the column
    # @param v Value        
    def setvalue(self, rowname, colname, v):
        self.matrix[self.keys[colname]][self.keys[rowname]] = v
        self.matrix[self.keys[rowname]][self.keys[colname]] = v
 
    ## Get an entry of the matrix. 
    # @param rowname Name of the row
    # @param colname Name of the column
    # @return Value        
    def getvalue(self, rowname, colname):
        return self.matrix[self.keys[colname]][self.keys[rowname]] 

    ## Function called by the print statement
    def __str__(self):
        outStr = ""
        for k in self.names:
            outStr += '%s: %s\n' % (k,self.matrix[self.keys[k]])
        return outStr
