##
# @file bio_expression.py
# @author    Michel Bierlaire and Mamy Fetiarison
#

# import hashlib #remove if not needed
# import random #remove if not needed
import string


## @brief Generic class for an operator. Manages the IDs of the various operators.
class Operator:

    num = 'numeric'
    var = 'variable'
    userexpr = 'UserExpression'
    userdraws = 'UserDraws'
    rv = 'randomVariable'
    param = 'Beta'
    mcdraws = 'MCDraw'
    mcunifdraws = 'MCUnifDraw'
    normal = 'N(0,1)'
    uniform = 'U[0,1]'
    uniformSym = 'U[-1,1]'
    absOp, negOp = 'abs', 'minus'
    exp, log = 'exp', 'log'
    bioNormalCdf = 'bioNormalCdf'
    add, sub, mul, div, power = '+', '-', '*', '/', '**'
    andOp, orOp, equal, notEqual = 'and', 'or', '==', '<>'
    greater, greaterEq, less, lessEq = '>', '>=', '<', '<='
    minOp, maxOp, mod = 'min', 'max', 'mod'
    sumOp, prodOp, integralOp, derivativeOp = 'sum', 'prod', 'integral', 'derivative'
    monteCarloOp = 'MonteCarlo'
    monteCarloCVOp = 'MonteCarloControlVariate'
    elemOp, enumOp, logitOp, loglogitOp, multSumOp, multProdOp, bioNormalPdf = (
        'elem',
        'enumerate',
        'logit',
        'loglogit',
        'multSum',
        'multProd',
        'bioNormalPdf',
    )
    mhOp = 'MH'
    bayesMeanOp = 'bioBayesNormalDraw'
    defineOp = 'define'

    # Bounds
    MIN_ZEROOP_INDEX = 0
    MAX_ZEROOP_INDEX = 19
    MIN_UNOP_INDEX = 20
    MAX_UNOP_INDEX = 39
    MIN_BINOP_INDEX = 40
    MAX_BINOP_INDEX = 69
    MIN_ITERATOROP_INDEX = 70
    MAX_ITERATOROP_INDEX = 89

    UNDEF_OP = -1

    # Each operator is associated with an index depending on
    # the above bounds
    operatorIndexDic = {
        num: 0,
        var: 1,
        param: 2,
        normal: 3,
        uniform: 4,
        rv: 5,
        uniformSym: 6,
        userexpr: 7,
        mcdraws: 8,
        mcunifdraws: 9,
        userdraws: 10,
        absOp: 20,
        negOp: 21,
        exp: 30,
        log: 31,
        bioNormalCdf: 33,
        monteCarloOp: 34,
        add: 40,
        sub: 41,
        mul: 42,
        div: 43,
        power: 44,
        andOp: 45,
        orOp: 46,
        equal: 47,
        notEqual: 48,
        greater: 49,
        greaterEq: 50,
        less: 51,
        lessEq: 52,
        minOp: 53,
        maxOp: 54,
        mod: 55,
        sumOp: 70,
        prodOp: 71,
        elemOp: 90,
        enumOp: 91,
        integralOp: 92,
        derivativeOp: 93,
        defineOp: 94,
        logitOp: 95,
        bioNormalPdf: 96,
        multSumOp: 97,
        multProdOp: 98,
        mhOp: 99,
        bayesMeanOp: 100,
        monteCarloCVOp: 101,
        loglogitOp: 102,
    }

    # Return the index associated to an operator
    def getOpIndex(self, op):
        return Operator.operatorIndexDic[op]


Operator = Operator()


## @brief Build an "Expression" object from a numeric value
# @param exp Object of numeric type or of type Expression
def buildExpressionObj(exp):
    ## Check if the object is numeric
    def isNumeric(obj):
        # Consider only ints and floats numeric
        return isinstance(obj, int) or isinstance(obj, float)

    if isNumeric(exp):
        return Numeric(exp)
    else:
        return exp


## @brief Interface for mathematical expressions
class Expression:
    ## Constructor
    def __init__(self):
        self.operatorIndex = UNDEF_OP

    ## @return Return the string representation of the current expression
    def getExpression(self):
        raise NotImplementedError("getExpression must be implemented! ")

    ## @return Return an ID for this expression, can be "xx-no ID" if the sublcass doest not override the function
    def getID(self):
        return str(self.operatorIndex) + "-no ID"

    ## @return Returns a string with the expression
    def __str__(self):
        return self.getExpression()

    ## @return If E is the expression, returns -E
    # @ingroup operators

    def __neg__(self):
        return UnOp(Operator.negOp, self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E + expression
    # @ingroup operators
    def __add__(self, expression):
        return BinOp(Operator.add, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns expression + E
    # @ingroup operators
    def __radd__(self, expression):
        return BinOp(Operator.add, buildExpressionObj(expression), self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E - expression
    # @ingroup operators
    def __sub__(self, expression):
        return BinOp(Operator.sub, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns expression - E
    # @ingroup operators
    def __rsub__(self, expression):
        return BinOp(Operator.sub, buildExpressionObj(expression), self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E * expression
    # @ingroup operators
    def __mul__(self, expression):
        return BinOp(Operator.mul, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns expression * E
    # @ingroup operators
    def __rmul__(self, expression):
        return BinOp(Operator.mul, buildExpressionObj(expression), self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E / expression
    # @ingroup operators
    def __div__(self, expression):
        return BinOp(Operator.div, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns expression / E
    # @ingroup operators
    def __rdiv__(self, expression):
        return BinOp(Operator.div, buildExpressionObj(expression), self)

    ## Support for Python version 3.x
    ## @param expression An another expression
    ## @return  If E is the expression, returns E / expression
    # @ingroup operators
    def __truediv__(self, expression):
        return BinOp(Operator.div, self, buildExpressionObj(expression))

    ## Support for Python version 3.x
    ## @param expression An another expression
    ## @return  If E is the expression, returns expression / E
    # @ingroup operators
    def __rtruediv__(self, expression):
        return BinOp(Operator.div, buildExpressionObj(expression), self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E % expression (modulo)
    # @ingroup operators
    def __mod__(self, expression):
        return BinOp(Operator.modOp, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E ^ expression
    # @ingroup operators
    def __pow__(self, expression):
        return BinOp(Operator.power, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns expression ^ E
    # @ingroup operators
    def __rpow__(self, expression):
        return BinOp(Operator.power, buildExpressionObj(expression), self)

    ## @param expression An another expression
    ## @return  If E is the expression, returns E and expression
    # @ingroup operators
    def __and__(self, expression):
        return BinOp(Operator.andOp, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E or expression
    # @ingroup operators
    def __or__(self, expression):
        return BinOp(Operator.orOp, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E == expression
    # @ingroup operators
    def __eq__(self, expression):
        return BinOp(Operator.equal, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E != expression
    # @ingroup operators
    def __ne__(self, expression):
        return BinOp(Operator.notEqual, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E <= expression
    # @ingroup operators
    def __le__(self, expression):
        return BinOp(Operator.lessEq, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E >= expression
    # @ingroup operators
    def __ge__(self, expression):
        return BinOp(Operator.greaterEq, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E < expression
    # @ingroup operators
    def __lt__(self, expression):
        return BinOp(Operator.less, self, buildExpressionObj(expression))

    ## @param expression An another expression
    ## @return  If E is the expression, returns E > expression
    # @ingroup operators
    def __gt__(self, expression):
        return BinOp(Operator.greater, self, buildExpressionObj(expression))


## @brief Class wrapping an integer or a float value
class Numeric(Expression):
    ## @param number integer or float
    # @ingroup expressions
    def __init__(self, number):
        self.number = number
        self.operatorIndex = Operator.operatorIndexDic[Operator.num]

    def getExpression(self):
        return "(" + str(self.number) + ")"

    def getID(self):
        return str(self.operatorIndex) + "-" + str(self.number)


## @brief Class representing the variables defined in the data file.
# @ingroup expressions
# @details Most
# users will not need it. Biogeme automatically invokes this
# expression for all headers in the data file.
# Example:
# @code
# x = Variable('x')
# @endcode
class Variable(Expression):
    ## @param name name of the variable
    def __init__(self, name):
        self.name = name
        self.operatorIndex = Operator.operatorIndexDic[Operator.var]

    def getExpression(self):
        return str(self.name)

    def getID(self):
        return str(self.operatorIndex) + "-" + str(self.name)


## @brief Class representing a random variable for numerical
# integration.
# @ingroup expressions
# @details Typically used for integrals. Note that nothing is
# said here about the distribution of the random variable. Therefore,
# a density function will have to be specified.
# Example:
# @code
# omega = RandomVariable('omega')
# @endcode
class RandomVariable(Expression):
    ## @param name name of the random variable
    def __init__(self, name):
        self.name = name
        self.operatorIndex = Operator.operatorIndexDic[Operator.rv]

    def getExpression(self):
        return str(self.name)


## @brief Class representing the definition of a new variable.
# @ingroup expressions
# @details Defining a new
# variable is equivalent to add a column to the data file. Note that
# the expression defining the new variable is computed during the
# processing of the data file, before the estimation, so that it saves
# computational time to define variables using this technique.
# Example:
# @code
# TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
# @endcode
# will result in a more efficient estimation than the statement:
# @code
# TRAIN_TT_SCALED = TRAIN_TT / 100.0
# @endcode
class DefineVariable(Expression):
    ## @param name name of the variable
    # @param expression expression to compute the value of the variable
    def __init__(self, name, expression):
        self.name = name
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.userexpr]

    def getExpression(self):
        return self.name + self.expression.getExpression()


## @brief Class representing the definition of a new type of draws.
# @ingroup expressions
# @details The expression defining the draws is computed during the
# processing of the data file, before the estimation, so that it may save
# computational time to define draws using this technique.
class DefineDraws(Expression):
    ## @param name name of the draws
    # @param expression expression to compute the value of the variable
    def __init__(self, name, expression, iteratorName):
        self.name = name
        self.expression = buildExpressionObj(expression)
        self.iteratorName = iteratorName
        self.operatorIndex = Operator.operatorIndexDic[Operator.userdraws]

    def getExpression(self):
        return self.name + self.expression.getExpression()

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class representing a parameter to be estimated.
# @ingroup expressions
# @details It is highly recommended to use the same name as an argument as the
# name of the python variable on the left of the equal sign. Indeed,
# Biogeme has no access to the name of the python variable, and the report
# will use only the name provided as an argument.
# Example:
# @code
# Beta = Beta( 'Beta', 0.0, -10000, 10000, 0)
# @endcode
class Beta(Expression):
    ## @param name name of the parameter
    # @param value starting value of the parameter
    # @param lowerbound minimum value that the parameter can take
    # @param upperbound maximum value that the parameter can take
    # @param status 0 if the parameter must be estimated, 1 if the starting value is maintained
    # @param desc text used in the LaTeX output file (optional, default value:'' [empty string])
    def __init__(self, name, value, lowerbound, upperbound, status, desc=''):
        self.name = name
        self.val = value
        self.lb = lowerbound
        self.ub = upperbound
        self.st = status
        self.desc = desc
        self.operatorIndex = Operator.operatorIndexDic[Operator.param]

    def getExpression(self):
        # return str(self.name)
        return (
            self.name
            + " "
            + str(self.val)
            + " "
            + str(self.lb)
            + " "
            + str(self.ub)
            + " "
            + str(self.st)
            + " "
            + self.desc
        )

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class representing a random variable for integration using
# Monte Carlo simulation.
# @ingroup expressions
class bioDraws(Expression):

    ## @param name name of the draws
    def __init__(self, name):
        print("**** DRAWS", name, " ", Operator.mcdraws)
        self.name = name
        self.operatorIndex = Operator.operatorIndexDic[Operator.mcdraws]

    def getExpression(self):
        return str(self.name)

    def getID(self):
        return str(self.operatorIndex) + "-Draw" + self.getExpression()


## @brief Class representing the uniform draw of a random variable used for integration using Monte Carlo simulation.
# @ingroup expressions
class bioRecycleDraws(Expression):

    ## @param name name of the draws
    def __init__(self, name):
        print("**** RECYCLED DRAWS", name, " ", Operator.mcunifdraws)
        self.name = name
        self.operatorIndex = Operator.operatorIndexDic[Operator.mcunifdraws]
        print("Id: ", self.getID())

    def getExpression(self):
        return "Unif(" + str(self.name) + ")"

    def getID(self):
        return str(self.operatorIndex) + "-Unif" + self.getExpression()


## @brief Class representing a normally distributed random variable for
# simulated integration.
# @ingroup expressions
# @details A different set of draws will be generated
# for each group in the data file. By default, the groups are
# identified by __rowId__, and a set of draws is
# generated for each row in the sample. Another typical case
# corresponds to panel data, where several rows correspond to the
# same individual. If Id identifies individuals in the data set, a
# set of draws will be generated for each individual, and not for
# each observation. Example:
# @code
# Errorcomp = bioNormalDraws('Errorcomp','Id')
# @endcode
class bioNormalDraws(Expression):

    ## @param name name of the random variable
    # @param index name of the identifier of groups of data associated
    # with a different set of draws. (optional, default='__rowId__')
    def __init__(self, name, index='__rowId__'):
        msg = (
            'Deprecated syntax: bioNormalDraws(\''
            + name
            + '\'). Use bioDraws(\''
            + name
            + '\') and BIOGEME_OBJECT.DRAWS = { \''
            + name
            + '\': \'NORMAL\' }'
        )

        raise SyntaxError(msg)


## @brief Class representing a uniformly distributed random variable
# on [-1,1] for simulated integration.
# @ingroup expressions
# @details A different set of draws will be generated
# for each group in the data file. By default, the groups are
# identified by __rowId__, and a set of draws is
# generated for each row in the sample. Another typical case
# corresponds to panel data, where several rows correspond to the
# same individual. If Id identifies individuals in the data set, a
# set of draws will be generated for each individual, and not for
# each observation. Example:
# @code
# Errorcomp = bioUniformSymmetricDraws('Errorcomp','Id')
# @endcode
class bioUniformSymmetricDraws(Expression):

    ## @param name name of the random variable
    # @param index name of the identifier of groups of data associated
    # with a different set of draws. (optional, default='__rowId__')
    def __init__(self, name, index='__rowId__'):
        msg = (
            'Deprecated syntax: bioUniformSymmetricDraws(\''
            + name
            + '\'). Use bioDraws(\''
            + name
            + '\') and BIOGEME_OBJECT.DRAWS = { \''
            + name
            + '\': \'UNIFORMSYM\' }'
        )

        raise SyntaxError(msg)


## @brief Class representing a uniformly distributed random variable
# on [0,1] for simulated integration.
# @ingroup expressions
# @details A different set of draws will be generated
# for each group in the data file. By default, the groups are
# identified by __rowId__, and a set of draws is
# generated for each row in the sample. Another typical case
# corresponds to panel data, where several rows correspond to the
# same individual. If Id identifies individuals in the data set, a
# set of draws will be generated for each individual, and not for
# each observation. Example:
# @code
# Errorcomp = bioUniformDraws('Errorcomp','Id')
# @endcode
class bioUniformDraws(Expression):

    ## @param name name of the random variable
    # @param index name of the identifier of groups of data associated
    # with a different set of draws. (optional, default='__rowId__')
    def __init__(self, name, index='__rowId__'):
        msg = (
            'Deprecated syntax: bioUniformDraws(\''
            + name
            + '\'). Use bioDraws(\''
            + name
            + '\') and BIOGEME_OBJECT.DRAWS = { \''
            + name
            + '\': \'UNIFORM\' }'
        )

        raise SyntaxError(msg)


## @brief Generic class for unary operators
class UnOp(Expression):
    ## @param op Object of type Operator
    ## @param expression any valid bio_expression
    def __init__(self, op, expression):
        self.op = op
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[op]

    def getExpression(self):
        return self.op + "(" + self.expression.getExpression() + ")"


## @brief Class representing the expression for absolute value
# @ingroup expressions
# @details It is not a differentiable operator. If the expression
# contains parameters to be estimated, Biogeme will complain.
# Example:
# @code
# y = abs(2*x-1)
# @endcode
class abs(Expression):
    ## @param expression any valid bio_expression
    def __init__(self, expression):
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.absOp]

    def getExpression(self):
        return Operator.absOp + "(" + self.expression.getExpression() + ")"


## @brief Class representing the expression for exponential
# @ingroup expressions
# @details Roughly speaking, floating point arithmetic allows to represent positive real numbers between \f$10^{-307}\f$ and \f$10^{+308}\f$, which corresponds to a range between \f$e^{-707}\f$ and \f$e^{709}\f$. Make sure that the value of the expression of exp does not lie outside the range [-707,709].
# Example:
# @code
# P = exp(v1) / (exp(v1) + exp(v2))
# @endcode
class exp(Expression):
    ## @param expression any valid bio_expression
    def __init__(self, expression):
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.exp]

    def getExpression(self):
        return Operator.exp + "(" + self.expression.getExpression() + ")"

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class representing the expression for natural logarithm.
# @ingroup expressions
## @details It is the natural logarithm (base e), so that \f$y=\log(x)\f$ means that \f$x=e^y\f$. To compute a logarithm in another base \f$b \neq 1\f$, use the formula \f[ \log_b(x) = \log(x) / \log(b) \f].
class log(Expression):
    ## @param expression any valid bio_expression
    def __init__(self, expression):
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.log]

    def getExpression(self):
        return Operator.log + "(" + self.expression.getExpression() + ")"

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class representing the cumulative distribution function of
# the normal distribution
# @ingroup expressions
# @details The CDF of the normal distribution is
# \f[
#  \mbox{bioNormalCdf}(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x \exp\left(\frac{-t^2}{2}\right) dt
# \f]
# A typical example is the probability of a binary probit model, which is computed as follows:
# @code
# prob = normalCdf(v1-v2)
# @endcode
class bioNormalCdf(Expression):
    ## @param expression any valid bio_expression
    def __init__(self, expression):
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.bioNormalCdf]

    def getExpression(self):
        return Operator.normalCdf + "(" + self.expression.getExpression() + ")"


## @brief Class representing the expression for the maximum of two expressions.
# @ingroup expressions
# @details Note that this operator is not differentiable. If one of
# the two expressions contains parameters to be estimated, Biogeme
# will complain. Example: @code max(x,0) @endcode
class max(Expression):
    ## @param left any valid bio_expression
    ## @param right any valid bio_expression
    def __init__(self, left, right):
        self.left = buildExpressionObj(left)
        self.right = buildExpressionObj(right)
        self.operatorIndex = Operator.operatorIndexDic[Operator.maxOp]

    def getExpression(self):
        return (
            "max(" + self.left.getExpression() + "," + self.right.getExpression() + ")"
        )


## @brief Class representing the expression for the minimum of two expressions.
# @ingroup expressions
# @details Example:
# @code
#  max(x,0)
# @endcode
class min(Expression):
    def __init__(self, left, right):
        self.left = buildExpressionObj(left)
        self.right = buildExpressionObj(right)
        self.operatorIndex = Operator.operatorIndexDic[Operator.minOp]

    def getExpression(self):
        return (
            "min(" + self.left.getExpression() + "," + self.right.getExpression() + ")"
        )


## @brief Generic class for binary operators
class BinOp(Expression):
    ## @param op Object of type Operator
    ## @param left any valid bio_expression
    ## @param right any valid bio_expression
    def __init__(self, op, left, right):
        self.op = op
        self.left = buildExpressionObj(left)
        self.right = buildExpressionObj(right)
        self.operatorIndex = Operator.operatorIndexDic[op]

    def getExpression(self):
        return (
            "(" + self.left.getExpression() + self.op + self.right.getExpression() + ")"
        )

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class representing the Monte Carlo integration of an expression
# @ingroup expressions
class MonteCarlo(Expression):
    ## @param term any valid bio_expression
    def __init__(self, expression):
        self.expression = buildExpressionObj(expression)
        self.operatorIndex = Operator.operatorIndexDic[Operator.monteCarloOp]

    def getExpression(self):
        strexpr = "MonteCarlo"
        strexpr += "(" + self.expression.getExpression() + ")"
        return strexpr


## @brief Class representing the Monte Carlo integration of an
## expression, using a control variate method to decrease the
## variance. The analytical result of the simulation of the second
## argument should be equal to the third.
# @ingroup expressions
class MonteCarloControlVariate(Expression):
    ## @param term any valid bio_expression
    def __init__(self, expression, integrand, integral):
        self.expression = buildExpressionObj(expression)
        self.integrand = buildExpressionObj(integrand)
        self.integral = buildExpressionObj(integral)
        self.operatorIndex = Operator.operatorIndexDic[Operator.monteCarloCVOp]

    def getExpression(self):
        strexpr = "MonteCarloControlVariate"
        strexpr += "(" + self.function.getExpression() + ")"
        return strexpr


## @brief Class representing the sum of the same expression applied to a list of data.
# @ingroup expressions
# @details The concept of iterators identifies a sequence such that,
# for each instance, the value of the variables is read from the data
# file, and an expression can be evaluated. The two expressions
# described in this section consider one iterator and one expression,
# and evaluate the expression for each instance defined by the
# iterator. A sum can then be computed. Example:
# @code
# prob = bioLogit(util,av,CHOICE)
# rowIterator('obsIter')
# loglikelihood = Sum(log(prob),'obsIter')
# @endcode
class Sum(Expression):
    ## @param term any valid bio_expression
    ## @param iteratorName name of an iterator already defined
    def __init__(self, term, iteratorName):
        self.function = buildExpressionObj(term)
        self.iteratorName = iteratorName
        self.operatorIndex = Operator.operatorIndexDic[Operator.sumOp]

    def getExpression(self):
        strexpr = "sum"
        strexpr += "[" + str(self.iteratorName) + "]"
        strexpr += "(" + self.function.getExpression() + ")"
        return strexpr


## @brief Class representing the product of the same expression applied to a list of data.
# @ingroup expressions
# @details The concept of iterators identifies a sequence such that,
# for each instance, the value of the variables is read from the data
# file, and an expression can be evaluated. The two expressions
# described in this section consider one iterator and one expression,
# and evaluate the expression for each instance defined by the
# iterator. A product can then be computed. The following example computes the loglikelihood for a model with panel data.
# @code
# metaIterator('personIter','__dataFile__','panelObsIter','Id')
# rowIterator('panelObsIter','personIter')
#
# condProbIndiv = Prod(prob,'panelObsIter')
# prob_indiv = MonteCarlo(condProbIndiv)
# loglikelihood = Sum(log(prob_indiv),'personIter')
# @endcode
# The iterator personIter iterates on each individual in the file,
# characterized by the identifier Id. The iterator panelObsIter
# iterates on the observations (that is, the rows in the data file)
# associated with the current individual.
#
# Assuming that prob is the likelihood of the observation in one raw,
# for a given set of draws, the following quantities are computed:
# - The conditional probability of the sequence of observations for
#   the current individual n:
#
# \f[ \mbox{condProbIndiv} = P(y_1,\ldots,y_T|\xi_n) = \prod_t P(y_t|\xi_n)\f]
#
# - The unconditional probability of the sequence of observations for the
#   current individual n:
# \f[
# \mbox{prob_indiv} = \int_{\xi_n}P(y_1,\ldots,y_T|\xi_n) \approx \sum_r P(y_1,\ldots,y_T|\xi_r) / R
# \f]
# where \f$\xi_r\f$ are the R draws generated from \f$\xi_n\f$.
#
# - The loglikelihood for all individuals in the sample:
# \f[
# \mbox{loglikelihood} = \sum_n \log(\sum_r P(y_1,\ldots,y_T|\xi_r) / R)
# \f]
class Prod(Expression):
    ## @param term any valid bio_expression
    ## @param iteratorName name of an iterator already defined
    ## @param positive Set it to True if all factors of the product are
    ## strictly positive. In that case, it will be computed as \f[
    ## \prod_r x_r = \exp(\sum_r \ln x_r)\f]
    def __init__(self, term, iteratorName, positive=False):
        self.function = buildExpressionObj(term)
        self.iteratorName = iteratorName
        self.positive = positive
        self.operatorIndex = Operator.operatorIndexDic[Operator.prodOp]

    def getExpression(self):
        strexpr = "prod"
        strexpr += "[" + str(self.iteratorName) + "]"
        strexpr += "(" + self.function.getExpression() + ")"
        return strexpr


## @brief Class performing numerical integration relying on the <a
## href='http://en.wikipedia.org/wiki/Gaussian_quadrature'
## target='_blank'>Gauss-Hermite quadrature</a> to compute
## \f[
##  \int_{-\infty}^{+\infty} f(\omega) d\omega.
## \f]
# @ingroup expressions
# @details As an example, the computation of a normal mixture of logit
# models is performed using the following syntax, where condprob is
# the conditional (logit) choice probability:
# @code
# omega = RandomVariable('omega')
# density = bioNormalPdf(omega)
# result = Integrate(condprob * density,'omega')
# @endcode
# Comments:
#  - The Gauss-Hermite procedure is designed to compute integrals of the form
# \f[
# \int_{-\infty}^{+\infty} e^{-\omega^2} f(\omega) d\omega.
# \f]
# Therefore, Biogeme multiplies the expression by \f$e^{\omega^2}\f$ before applying the Gauss-Hermite algorithm. This is transparent for the user.
#
# - It is usually more accurate to compute an integral using a
# quadrature procedure. However, it should be used only in the
# presence of few (one or two) random variables. The same integral can
# be computed using Monte-Carlo integration using the following
# syntax:
# @code
# BIOGEME_OBJECT.DRAWS = { 'omega': 'NORMAL'}
# omega = bioDraws('omega')
# result = MonteCarlo(condprob)
# @endcode
class Integrate(Expression):
    ## @param term any valid bio_expression representing the expression to integrate
    ## @param v name of the integration variable, previously defined using a bioExpression::RandomVariable statement.
    def __init__(self, term, v):
        self.function = buildExpressionObj(term)
        self.variable = v
        self.operatorIndex = Operator.operatorIndexDic[Operator.integralOp]

    def getExpression(self):
        strexpr = "Integral"
        strexpr += "(" + self.function.getExpression() + "," + variable + ")"
        return strexpr


## @brief Class generating the analytical derivative of an expression.
# @ingroup expressions
# @details  This class generates the expression for
# \f[
# \frac{\partial f(x)}{\partial x}
# \f]
# The computation of derivatives is usually not necessary for
# estimation, as Biogeme automatically generates the analytical
# derivatives of the log likelihood function. It is particularly
# usuful for simulation, to compute the elasticities of complex
# models.
# Example:
# The computation of the elasticity of a model prob with respect to
# the variable TRAIN_TT, say, is obtained as follows, whatever the
# model defined by prob:
# @code
# elasticity = Derive(prob,'TRAIN_TT') * TRAIN_TT / prob
# @endcode
class Derive(Expression):
    ## @param term any valid bio_expression to derive
    ## @param v the variable
    def __init__(self, term, v):
        self.function = buildExpressionObj(term)
        self.variable = v
        self.operatorIndex = Operator.operatorIndexDic[Operator.derivativeOp]

    def getExpression(self):
        strexpr = "Derive"
        strexpr += "(" + self.function.getExpression() + "," + variable + ")"
        return strexpr


## @brief Class representing the probability density function of a
# standardized normally distributed random variable
# @ingroup expressions
# @details The pdf of the normal distribution is
# \f[ \mbox{bioNormalPdf}(x) = \frac{1}{\sqrt{2\pi}}e^{-t^2/2}. \f]
# The computation of a normal mixture of logit models is performed
# using the following syntax, where condprob is the conditional
# (logit) choice probability:
# @code
# omega = RandomVariable('omega')
# density = bioNormalPdf(omega)
# result = Integrate(condprob * density,'omega')
# @endcode
class bioNormalPdf(Expression):
    ## @param term an expression providing the value of the argument of the pdf
    def __init__(self, term):
        self.function = buildExpressionObj(term)
        self.operatorIndex = Operator.operatorIndexDic[Operator.bioNormalPdf]

    def getExpression(self):
        strexpr = "normalPdf"
        strexpr += "(" + self.function.getExpression() + ")"
        return strexpr


## @brief Class extracting an expression from a dictionary.
# @ingroup expressions
# @details A typical example consists in returning the probability of a chosen alternative in a choice model. Consider the following dictionary:
# @code
# P = {	1: exp(v1) / (exp(v1)+exp(v2)+exp(v3)),
# 2: exp(v2) / (exp(v1)+exp(v2)+exp(v3)),
# 3: exp(v3) / (exp(v1)+exp(v2)+exp(v3))}
# @endcode
# and assume that the variable Choice in the data file contains the
# identifier of the chosen alternative of the corresponding
# observation, that is 1, 2 or 3. Then, the probability of the chosen
# alternative is given by
# @code
# chosenProba = Elem(P,Choice)
# @endcode
# If the result of "Choice" does not correspond to any valid entry in the
# dictionary, a warning is issued and the value of the default expression
# is returned. The warning is issued because this situation is usually
# not wanted by the user. To turn off the warning, set the parameter
# warnsForIllegalElements to 0.
class Elem(Expression):
    ## @param dictionary A dictionary (see <a href='http://docs.python.org/py3k/tutorial/datastructures.html' target='_blank'>Python documentation</a>) such
    ## that the indices are numerical values that can match the result
    ## of an expression
    ## @param key expression identifying the entry in the dictionary
    ## @default If the result of key does not correspond to any
    ## valid entry in the dictionary, the value of the default expression
    ## is returned. This argument is optional. If omitted, the default
    ## value is 0.
    def __init__(self, dictionary, key, default=Numeric(0)):
        self.prob = {}  # dictionary
        for k, v in dictionary.items():
            self.prob[k] = buildExpressionObj(v)

        self.choice = buildExpressionObj(key)
        self.default = buildExpressionObj(default)
        self.operatorIndex = Operator.operatorIndexDic[Operator.elemOp]

    def getExpression(self):
        res = "Elem"
        res += "[" + str(self.choice) + "]"
        res += "{"
        for i, v in self.prob.items():
            res += "(" + str(i) + ": " + str(v) + ")"
        res += "}"
        return res

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class calculating a logit choice probability
# @ingroup expressions
# @details Computes the probability given by the logit model, that is
# \f[
#   \frac{a_i e^{V_i}}{\sum_{j=1}^{J} a_j e^{V_j}}
# \f]
# Example:
# @code
# util = {1: v1, 2: v2, 3: v3}
# av = {1: av_1, 2: 1, 3: av_3}
# prob = bioLogit(util,av,CHOICE)
# @endcode
class bioLogit(Expression):
    ## @param util dictionary (see <a
    ## href="http://docs.python.org/py3k/tutorial/datastructures.html"
    ## target="_blank">Python documentation</a>) containing the utility
    ## functions \f$V_i\f$.
    ## @param av dictionary (see <a
    ## href="http://docs.python.org/py3k/tutorial/datastructures.html"
    ## target="_blank">Python documentation</a>) containing the
    ## availability conditions for each alternative \f$a_i\f$ (if 0,
    ## alternative is not available, if different from zero, it is
    ## available). The number of entries and the labeling must be
    ## consistent with utilities.
    ## @param choice expression providing the index of the alternative
    ## for which the probability is being computed.
    def __init__(self, util, av, choice):
        self.prob = {}  # dictionary
        for k, v in util.items():
            self.prob[k] = buildExpressionObj(v)
        self.av = {}
        for k, v in av.items():
            self.av[k] = buildExpressionObj(v)
        self.choice = buildExpressionObj(choice)
        self.operatorIndex = Operator.operatorIndexDic[Operator.logitOp]

    def getExpression(self):
        res = "Logit"
        res += "[" + str(self.choice) + "]"
        res += "{"
        for i, v in self.prob.items():
            res += "(" + str(i) + ": " + str(v) + ")"
        res += "}"
        res += "{"
        for i, v in self.av.items():
            res += "(" + str(i) + ": " + str(v) + ")"
        res += "}"
        return res


## @brief Class calculating the log of a logit choice probability
# @ingroup expressions
# @details Computes the log of the probability given by the logit model, that is
# \f[
#  V_i - \log\left(\sum_{j=1}^{J} a_j e^{V_j}\right)
# \f]
# Example:
# @code
# util = {1: v1, 2: v2, 3: v3}
# av = {1: av_1, 2: 1, 3: av_3}
# logprob = bioLogLogit(util,av,CHOICE)
# @endcode
class bioLogLogit(Expression):
    ## @param util dictionary (see <a
    ## href="http://docs.python.org/py3k/tutorial/datastructures.html"
    ## target="_blank">Python documentation</a>) containing the utility
    ## functions \f$V_i\f$.
    ## @param av dictionary (see <a
    ## href="http://docs.python.org/py3k/tutorial/datastructures.html"
    ## target="_blank">Python documentation</a>) containing the
    ## availability conditions for each alternative \f$a_i\f$ (if 0,
    ## alternative is not available, if different from zero, it is
    ## available). The number of entries and the labeling must be
    ## consistent with utilities.
    ## @param choice expression providing the index of the alternative
    ## for which the probability is being computed.
    def __init__(self, util, av, choice):
        self.prob = {}  # dictionary
        for k, v in util.items():
            self.prob[k] = buildExpressionObj(v)
        self.av = {}
        for k, v in av.items():
            self.av[k] = buildExpressionObj(v)

        self.choice = buildExpressionObj(choice)
        self.operatorIndex = Operator.operatorIndexDic[Operator.loglogitOp]

    def getExpression(self):
        res = "LogLogit"
        res += "[" + str(self.choice) + "]"
        res += "{"
        for i, v in self.prob.items():
            res += "(" + str(i) + ": " + str(v) + ")"
        res += "}"
        res += "{"
        for i, v in self.av.items():
            res += "(" + str(i) + ": " + str(v) + ")"
        res += "}"
        return res


## @brief Class computing the sum of multiple expressions
# @ingroup expressions
# @details Given a list of expressions \f$V_i\$, $i=1,\ldots,n$, it computes
#   \f[ \sum_{i=1}^n V_i \f]. Example:
# @code
# util = {1: v1, 2: v2, 3: v3}
#
# @endcode
class bioMultSum(Expression):

    def __init__(self, terms):
        if type(terms).__name__ == 'list':
            self.type = 1
            self.terms = []
            for k in terms:
                self.terms.append(buildExpressionObj(k))
        elif type(terms).__name__ == 'dict':
            self.type = 2
            self.terms = {}
            for k, v in terms.items():
                self.terms[k] = buildExpressionObj(v)
        else:
            self.type = -1
        self.operatorIndex = Operator.operatorIndexDic[Operator.multSumOp]

    def getExpression(self):
        res = "bioMultSum"
        res += "("
        if self.type == 2:
            for i, v in self.terms.items():
                res += v.getExpression() + ","

        if self.type == 1:
            for k in self.terms:
                res += k.getExpression() + ","

        # remove last coma
        res = res[:-1] + ")"
        return res

    def getID(self):
        return str(self.operatorIndex) + "-" + self.getExpression()


## @brief Class performing a sample enumeration
# @ingroup expressions
# @details The concept of iterators (see the section "Iterators"),
# identifies a sequence such that, for each instance, the value of the
# variables is read from the data file, and an expression can be
# evaluated. In sample enumeration, expressions are computed for each
# instance in the sample and reported in a file.
# Example:
# @code
# simulate = {'Prob. 1': P1, 'Prob. 2': P2, 'Util. 1': v1, 'Util. 2':v2, 'Diff. util.': v1-v2}
# rowIterator('obsIter')
# BIOGEME_OBJECT.SIMULATE = Enumerate(simulate,'obsIter')
# @endcode
# Note that this expression is used exclusively for
# simulation. Contrarily to all other expressions, it does not return
# any value and, consequently, cannot be included in another
# expression. Something like @code X + Enumerate(util,'obsIter') / 2 @endcode does not make any sense.


class Enumerate(Expression):
    ## @param term a dictionary (see <a href="http://docs.python.org/py3k/tutorial/datastructures.html" target="_blank">Python documentation</a>)) of the form
    ## @code
    ## {label: expression, label: expression,...}
    ## @endcode
    ## where label is used to describe the results in the output file, and
    ## expression is a valid expression.
    ## @param iteratorName name of an iterator already defined.
    def __init__(self, term, iteratorName):
        self.theDict = {}
        for k, v in term.items():
            self.theDict[k] = buildExpressionObj(v)
        self.iteratorName = iteratorName
        self.operatorIndex = Operator.operatorIndexDic[Operator.enumOp]

    def getExpression(self):
        strexpr = "Enumerate"
        strexpr += "[" + str(self.iteratorName) + "]"
        strexpr += "{"
        for i, v in self.term.items():
            strexpr += "(" + str(i) + ": " + str(v) + ")"
        strexpr += "}"
        return strexpr


## @brief Class performing draws from the posterior of the mean of a
## normal variable knowing realizations and variance. See Train (2003)
## p. 304 step 1.
# @details Let \f$\beta_n\f$, \f$n=1,\ldots,N\f$ be realizations from a
# multivariate normal distribution \f$N(b,W)\f$. This class draws from
# the posterior of \f$b\f$, that is from \f[ N(\bar{\Beta},W/N) \f].
class bioBayesNormalDraw(Expression):
    def __init__(self, mean, realizations, varcovar):
        self.error = None
        if type(mean).__name__ == 'list':
            self.mean = []
            for k in mean:
                self.mean.append(buildExpressionObj(k))
        else:
            self.error = "Syntax error: the first argument of bioBayesNormalDraw must be a list of expressions. Syntax: [B_TIME, B_COST]. "

        if type(realizations).__name__ == 'list':
            self.realizations = []
            for k in realizations:
                self.realizations.append(buildExpressionObj(k))
        else:
            self.error = "Syntax error: the second argument of bioBayesNormalDraw must be a list of expressions. Syntax: [b_time_rnd, B_COST_RND]"
        if type(varcovar).__name__ == 'list':
            self.varcovar = []
            for k in varcovar:
                row = []
                if type(k).__name__ == 'list':
                    for j in k:
                        row.append(buildExpressionObj(k))
                    self.varcovar.append(row)
                else:
                    self.error = "Syntax error: the third argument of bioBayesNormalDraw must be a list of list of expressions. Syntax: [[ B_TIME_S , B_COVAR ] , [B_COVAR , B_COST_S]]."

        else:
            self.error = "Syntax error: the third argument of bioBayesNormalDraw must be a list of list of expressions. Syntax: [[ B_TIME_S , B_COVAR ] , [B_COVAR , B_COST_S]]."
        self.varcovar = varcovar
        self.operatorIndex = Operator.operatorIndexDic[Operator.bayesMeanOp]


## @brief Class performing draws from densities for Bayesian
## estimation using Metropolis-Hastings algorithm
# @details Values of Beta parameters are drawn from a given density function using a Metropolis-Hastings algorithm.
# Example:
# @code
# BETA = {ASC_CAR, ASC_TRAIN, B_TIME, B_COST}
# prob = bioLogit(util,av,CHOICE)
# rowIterator('obsIter')
# likelihood = Prod(prob,'obsIter')
# BIOGEME_OBJECT.BAYESIAN = MH(BETA,likelihood)
# @endcode
class MH(Expression):
    ## @param Beta a list of the form
    ## @code
    ## {beta1, beta2,...}
    ## @endcode
    ## where beta1, beta2, etc. are defined by the Beta expression.
    ## @param density valid expression representing the density to draw from.

    ## @param warmup number of steps of the Markov chain to perform to
    ## reach stationarity.
    ## @param steps number of steps to skip between two draws.
    def __init__(self, beta, density, warmup, steps):
        if type(beta).__name__ == 'list':
            self.type = 1
            self.beta = []
            for k in beta:
                self.beta.append(buildExpressionObj(k))
        elif type(beta).__name__ == 'dict':
            self.type = 2
            self.beta = {}
            for k, v in beta.items():
                self.beta[k] = buildExpressionObj(v)
        else:
            self.type = -1
        self.density = density
        self.warmup = warmup
        self.steps = steps
        self.operatorIndex = Operator.operatorIndexDic[Operator.mhOp]


# class Define(Expression) :
#   def __init__(self, name, expr, classname=None):
#      self.name = name
#      self.expr = expr
#      self.operatorIndex = Operator.operatorIndexDic[Operator.defineOp]
#
#   def getExpression(self):
#      return str(self.name)
#
#   def assign(self, expr) :
#      self.expr = expr
