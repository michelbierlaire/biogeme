##
# @file bio_iterator.py
# @author  Michel Bierlaire and Mamy Fetiarison
# @date      Thu Jul  9 16:26:22 2009


listOfIterator = []

## @brief Generic class for an iterator.
# @details Iterators are designed to define loops on the entries of
# the data file. It is assumed that the data file is composed of rows,
# each containing the same number of numeric entries. The first row of
# the file contains the headers, labeling each of these entries. There
# are three types of iterators:
# - row iterators are designed to iterate on rows of the data file. This is the most common type of iterators.
# - meta iterators are designed to iterate on group of rows of the data file. They are typically used for panel data, where several rows correspond to the same individual.
# - draw iterators are designed to iterate on the data file when user
#   defined draws are generated. Note that this is valid only since
#   biogeme 2.4. Before, this iterator was used for the calculation of
#   the integral itself, together with an Operator Sum.
#
# Hierarchical nesting of iterators is possible, so that iterators may have a parent and a child.
class iterator:
    # @param iteratorName Name of the iterator.
    # @param name Name of the set of data that is being iterated. It is either the name of a metaiterator, or the string __dataFile__, when the iterator spans the entire database.
    # @param child Name of the child
    # @param variable Variable that contains the ID of the elements being iterated.
    def __init__(self,iteratorName,name,child,variable):
        self.iteratorName = iteratorName
        self.name = name
        self.child = child
        self.variable = variable
        listOfIterator.append(self) ;

    def __str__(self):
        return "Iterator " + str(self.name) + " iterates on " + str(self.variable)

## @brief  Iterates on the data file for generate the user defined draws (since biogeme 2.4)
# @details It is typically used with the DefineDraws expression to generate user defined draws.
#Typical usage:
# @code
# drawIterator('drawIter') 
# TRIANG = DefineDraws('TRIANG', 0.5 * (bioDraws('BUNIF1')+bioDraws('BUNIF2')),'drawIter')
# @endcode
class drawIterator(iterator):
    ## @param iteratorName Name of the iterator.
    # @param name Name of the set of data that is being iterated. It is typically ignored for draw iterators. Use the default value.
    # @param child Draw iterators have no children. Use the default value. 
    # @param variable Variable that contains the ID of the individuals in the data file. A different draw is generated for each different value of this variable.
    def __init__(self,iteratorName,name="__dataFile__",child="",variable="__rowId__"):
        msg = 'Deprecated syntax. No need to define a draw iterator anymore. Just remove the statement, and use the operator MonteCarlo instead of Sum.'
        raise SyntaxError(msg) 

## @brief meta iterators are designed to iterate on group of rows of the data file. 
# @details They are typically used for panel data, where several rows correspond to the same individual.
# In the example represented in the table below, the meta iterator will identify 4 groups of data: rows 1 to 4, rows 5 to 7, rows 8 to 9 and rows 10 to 11. Note that group 1 and group 4 share the same Id. But the iterator does not take this into account, as only changes of the value of the identifier characterize a change of group. If rows 10 and 11 indeed belong to group 1, the data file must be edited so that they appear directly after row 4.
# @code 
#__rowId__	Id	ObsId	Variables
#        1	1	1	...
#        2	1	2	...
#        3	1	3	...
#        4	1	4	...
#        5	2	1	...
#        6	2	2	...
#        7	2	3	...
#        8	3	1	...
#        9	3	2	...
#       10	1	5	...
#       11	1	6	...
#@endcode
# An example of iterator on this data is
# @code
#metaIterator('personIter','__dataFile__','panelObsIter','Id')
#rowIterator('panelObsIter','personIter')
# @endcode
class metaIterator(iterator):
    ## @param iteratorName Name of the iterator.
    # @param name Name of the set of data that is being iterated. It is either the name of a metaiterator, or the string __dataFile__, when the iterator spans the entire database.
    # @param child Name of the child
    # @param variable Variable that contains the ID of the elements being iterated.
    def __init__(self,iteratorName,name,child,variable):
        iterator.__init__(self,iteratorName,name,child,variable)
        self.type = 'META'
## @brief row iterators are designed to iterate on rows of the data file. 
# @details Examples:
# - Iterator on the data file: @code rowIterator('obsIter') @endcode
# - Iterator on another iterator:
#@code
#metaIterator('personIter','__dataFile__','panelObsIter','Id')
#rowIterator('panelObsIter','personIter')
#@endcode
class rowIterator(iterator):
    ## @param iteratorName Name of the iterator.
    # @param name Name of the set of data that is being iterated. It is either the name of a metaiterator, or the string __dataFile__, when the iterator spans the entire database.
    # @param child Row iterators have no children. Use the default value.
    # @param variable Ignored by row iterators. Use the default value.
    def __init__(self,iteratorName,name="__dataFile__",child="",variable="__rowId__"):
        iterator.__init__(self,iteratorName,name,child,variable)
        self.type = 'ROW'
