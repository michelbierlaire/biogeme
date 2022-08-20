"""File knapsack.py

:author: Michel Bierlaire
:date: Fri Feb 19 15:19:14 2021

This illustrates how to use the multi-objective VNS algorithm using a
simple knapsack problem.  There are two objectives: maximize utility
and minimize weight. The weight cannot go beyond capacity.

"""
import numpy as np
from biogeme import vns
import biogeme.exceptions as excep
import biogeme.messaging as msg

logger = msg.bioMessage()
# logger.setDetailed()
logger.setDebug()

# pylint: disable=invalid-name


class oneSack(vns.solutionClass):
    """Implements the virtual class. A solution here is a sack
    configuration.

    """

    def __init__(self, solution):
        super().__init__()
        self.x = solution
        self.objectivesNames = ['Weight', 'Negative utility']
        self.objectives = None

    def isDefined(self):
        """Check if the sack is well defined.

        :return: True if the configuration vector ``x`` is defined,
            and the total weight and total utility are both defined.
        :rtype: bool
        """
        if self.x is None:
            return False
        if self.objectives is None:
            return False
        return True

    def __repr__(self):
        return str(self.x)

    def __str__(self):
        return str(self.x)


def addItems(aSolution, size=1):
    """Add ``size`` items in the sack

    :param aSolution: current sack
    :type aSolution: class oneSack

    :param size: number of items to add into the sack
    :type size: int

    :return: new sack, and number of changes actually made
    :rtype: tuple(class oneSack, int)
    """
    absent = np.where(aSolution.x == 0)[0]
    np.random.shuffle(absent)
    n = min(len(absent), size)
    xplus = aSolution.x.copy()
    xplus[absent[:n]] = 1
    return oneSack(xplus), n


def removeItems(aSolution, size=1):
    """Remove ``size`` items from the sack

    :param aSolution: current sack
    :type aSolution: class oneSack

    :param size: number of items to remove from the sack
    :type size: int

    :return: new sack, and number of changes actually made
    :rtype: tuple(class oneSack, int)
    """
    present = np.where(aSolution.x == 1)[0]
    np.random.shuffle(present)
    n = min(len(present), size)
    xplus = aSolution.x.copy()
    xplus[present[:n]] = 0
    return oneSack(xplus), n


def changeDecisions(aSolution, size=1):
    """Change the status of ``size`` items in the sack.

    :param aSolution: current sack
    :type aSolution: class oneSack

    :param size: number of items to modify
    :type size: int

    :return: new sack, and number of changes actually made
    :rtype: tuple(class oneSack, int)

    """
    n = len(aSolution.x)
    order = np.random.permutation(n)
    xplus = aSolution.x.copy()
    for i in range(min(n, size)):
        xplus[order[i]] = 1 - xplus[order[i]]
    return oneSack(xplus), n


class knapsack(vns.problemClass):
    """Class defining the knapsack problem. Note the inheritance from the
    abstract class used by the VNS algorithm. It guarantees the
    compliance with the requirements of the algorithm.

    """

    def __init__(self, aUtility, aWeight, aCapacity):
        """Ctor"""
        super().__init__()
        self.utility = aUtility
        self.weight = aWeight
        self.capacity = aCapacity
        self.operators = {
            'Add items': addItems,
            'Remove items': removeItems,
            'Change decision for items': changeDecisions,
        }
        self.operatorsManagement = vns.operatorsManagement(
            self.operators.keys()
        )
        self.currentSolution = None
        self.lastOperator = None

    def emptySack(self):
        """
        :return: empty sack of the same dimension.
        :rtype: class oneSack
        """
        z = np.zeros_like(self.utility)
        theSack = oneSack(z)
        return oneSack(z)

    def isValid(self, aSolution):
        """Check if the sack verifies the capacity constraint

        :param aSolution: sack to check
        :type aSolution: class oneSack

        :return: True if the capacity constraint is verified
        :rtype: bool
        """
        self.evaluate(aSolution)
        return aSolution.objectives[0] <= self.capacity, 'Infeasible sack'

    def evaluate(self, aSolution):
        """Calculates the total weight and the total utility of a sack.

        :param aSolution: sack to evaluate
        :type aSolution: class oneSack

        """
        aSolution.objectives = [
            np.inner(aSolution.x, self.weight),
            -np.inner(aSolution.x, self.utility),
        ]

    def describe(self, aSolution):
        """Short description of a sack

        :param aSolution: sack to describe
        :type aSolution: class oneSack

        :return: description
        :rtype: str
        """
        return str(aSolution)

    def generateNeighbor(self, aSolution, neighborhoodSize):
        """Generate a neighbor from the negihborhood of size
        ``neighborhoodSize``using one of the operators

        :param aSolution: current solution
        :type aSolution: class oneSack

        :param neighborhoodSize: size of the neighborhood
        :type neighborhoodSize: int

        :return: number of modifications actually made
        :rtype: int

        """
        # Select one operator.
        self.lastOperator = self.operatorsManagement.selectOperator()
        return self.applyOperator(
            aSolution, self.lastOperator, neighborhoodSize
        )

    def neighborRejected(self, aSolution, aNeighbor):
        """Informs the operator management object that the neighbor has been
        rejected.

        :param aSolution: current solution
        :type aSolution: class oneSack

        :param aNeighbor: proposed neighbor
        :type aNeighbor: class oneSack

        :raise biogemeError: if no operator has been used yet.
        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.decreaseScore(self.lastOperator)

    def neighborAccepted(self, aSolution, aNeighbor):
        """Informs the operator management object that the neighbor has been
        accepted.

        :param aSolution: current solution
        :type aSolution: class oneSack

        :param aNeighbor: proposed neighbor
        :type aNeighbor: class oneSack

        :raise biogemeError: if no operator has been used yet.
        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.increaseScore(self.lastOperator)

    def applyOperator(self, solution, name, size=1):
        """Apply a specific operator on a solution, using a neighborhood of
        size ``size``

        :param solution: current solution
        :type solution: class oneSack

        :param name: name of the operator
        :type name: str

        :param size: size of the neighborhood
        :type size: int

        :return: number of modifications actually made
        :rtype: int

        :raise biogemeError: if the name of the operator is unknown.

        """
        op = self.operators.get(name)
        if op is None:
            raise excep.biogemeError(f'Unknown operator: {name}')
        return op(solution, size)
