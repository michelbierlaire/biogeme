"""Implements some useful functions

:author: Michel Bierlaire

:date: Sun Apr 14 10:46:10 2019

"""

import numpy as np
import biogeme.messaging as msg


def findiff_g(theFunction,x):
    """Calculates the gradient of a function :math`f` using finite differences

    :param theFunction: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`. The other elements are not
                        used.  
    :type theFunction: function

    :param x: argument of the function
    :type x: numpy.array

    :return: numpy vector, same dimension as x, containing the gradient
       calculated by finite differences.
    :rtype: numpy.array

    """
    tau = 0.0000001
    n = len(x)
    g = np.zeros(n)
    f = theFunction(x)[0]
    for i in range(n):
        xi = x.item(i)
        xp = x.copy()
        if (abs(xi) >= 1):
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        xp[i] = xi + s
        fp = theFunction(xp)[0]
        g[i] = (fp - f) / s
    return g

def findiff_H(theFunction,x):
    """Calculates the hessian of a function :math:`f` using finite differences

    :param theFunction: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`, and the second is the
                        gradient of the function.  The other elements
                        are not used.

    :type theFunction: function

    :param x: argument of the function
    :type x: numpy.array
    
    :return: numpy matrix containing the hessian calculated by finite differences.
    :rtype: numpy.array

    """
    tau = 0.0000001
    n = len(x)
    H = np.zeros((n,n))
    g = theFunction(x)[1]
    I = np.eye(n,n)
    for i in range(n):
        xi = x.item(i)
        if (abs(xi) >= 1):
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        ei = I[i]
        gp = theFunction(x + s * ei)[1]
        H[:,i] = (gp-g).flatten() / s
    return H


def checkDerivatives(theFunction,x,names=None,logg=False):
    """Verifies the analytical derivatives of a function by comparing them with finite difference approximations.

    :param theFunction:  A function object that takes a vector as an  argument, and returns a tuple. 

          - The first element of the tuple is the value of the function :math:`f`,
          - the second is the gradient of the function,
          - the third is the hessian.

    :type theFunction: function

    :param x: arguments of the function
    :type x: numpy.array

    :param names: the names of the entries of x (for reporting).
    :type names: list(string)
    :param logg: if True, messages will be displayed. 
    :type logg: bool
    
       
    :return: tuple f,g,h,gdiff,hdiff where

          - f is the value of the function at x,
          - g is the analytical gradient,
          - h is the analytical hessian,
          - gdiff is the difference between the analytical gradient and the finite difference approximation
          - hdiff is the difference between the analytical hessian and the finite difference approximation

    :rtype: float, numpy.array,numpy.array,  numpy.array,numpy.array
    """
    f,g,h = theFunction(x)
    g_num = findiff_g(theFunction,x)
    gdiff = g - g_num
    if logg:
        logger = msg.bioMessage()
        if names is None:
            names = [f"x[{i}]" for i in range(len(x))]
        logger.detailed("x\t\tGradient\tFinDiff\t\tDifference")
        for k in range(len(gdiff)):
            logger.detailed("{:15}\t{:+E}\t{:+E}\t{:+E}".format(names[k],g.item(k),g_num.item(k),gdiff.item(k)))

    h_num = findiff_H(theFunction,x)
    hdiff = h - h_num
    if logg:
        logger.detailed("Row\t\tCol\t\tHessian\tFinDiff\t\tDifference")
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logger.detailed("{:15}\t{:15}\t{:+E}\t{:+E}\t{:+E}".format(names[row],names[col],h[row,col],h_num[row,col],hdiff[row,col]))
    return f,g,h,gdiff,hdiff

def getPrimeNumbers(n):
    """ Get a given number of prime numbers

    :param n: number of primes that are requested
    :type n: int

    :return: array with prime numbers
    :rtype: list(int)
    """
    total = 0
    upperBound = 100
    while total < n:
        upperBound *= 10
        primes = calculatePrimeNumbers(upperBound)
        total = len(primes)
    return primes[0:n]

def calculatePrimeNumbers(upperBound):
    """ Calculate prime numbers

    :param upperBound: prime numbers up to this value will be computed 
    :type upperBound: int

    :return: array with prime numbers
    :rtype: list(int)
    """
    mywork = [i for i in range(0,upperBound+1)]
    max = int(np.ceil(np.sqrt(float(upperBound))))
    # Remove all multiples
    for i in range(2,max+1):
        if mywork[i] != 0:
            for k in range(2*i,upperBound+1,i):
                mywork[k] = 0
    # Gather non zero entries, which are the prime numbers
    myprimes = []
    for i in range(1,upperBound+1):
        if mywork[i] != 0 and mywork[i] != 1:
            myprimes += [mywork[i]]

    return myprimes

def countNumberOfGroups(df,column):
    """ This function counts the number of groups of same value in a column. 
      For instance: 1,2,2,3,3,3,4,1,1  would give 5
    """
    df['_biogroups'] = (df[column] != df[column].shift(1)).cumsum()
    return len(df['_biogroups'].unique())
    
