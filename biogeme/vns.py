"""File vns.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 16 16:55:30 2020

Multi-objective variable neighborhood search algorithm
"""

# pylint: disable=invalid-name, too-many-statements, too-many-arguments, too-many-branches
# pylint: disable=too-many-locals
import random
import abc
import pickle
import numpy as np
import biogeme.exceptions as excep
import biogeme.messaging as msg

logger = msg.bioMessage()


class problemClass(metaclass=abc.ABCMeta):
    """
    Abstract class defining a problem
    """

    @abc.abstractmethod
    def isValid(self, aSolution):
        """Evaluate  the validity of the solution.

        :param aSolution: solution to be checked
        :type aSolution: :class:`biogeme.vns.solutionClass`

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. why contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """

    @abc.abstractmethod
    def evaluate(self, aSolution):
        """Evaluate the objectives functions of the solution and store them in
        the solution object.

        :param aSolution: solution to be evaluated
        :type aSolution: :class:`biogeme.vns.solutionClass`

        """

    @abc.abstractmethod
    def describe(self, aSolution):
        """Short description of the solution. Used for reporting.

        :param aSolution: solution to be described
        :type aSolution: :class:`biogeme.vns.solutionClass`

        :return: short description of the solution.
        :rtype: str
        """

    @abc.abstractmethod
    def generateNeighbor(self, aSolution, neighborhoodSize):
        """
        Generates a neighbor of the solution.

        :param aSolution: solution to be modified
        :type aSolution: :class:`biogeme.vns.solutionClass`

        :param neighborhoodSize: size of the neighborhood to be applied
        :type neighborhoodSize: int

        :return: a neighbor solution, and the number of changes that have been
                 actually applied.
        :rtype: tuple(:class:`biogeme.vns.solutionClass`, int)
        """

    @abc.abstractmethod
    def neighborRejected(self, aSolution, aNeighbor):
        """Notify that a neighbor has been rejected by the
        algorithm. Used to update the statistics on the operators.

        :param aSolution: solution  modified
        :type aSolution: :class:`biogeme.vns.solutionClass`

        :param aNeighbor: neighbor
        :type aNeighbor: :class:`biogeme.vns.solutionClass`
        """

    @abc.abstractmethod
    def neighborAccepted(self, aSolution, aNeighbor):
        """Notify that a neighbor has been accepted by the
        algorithm. Used to update the statistics on the operators.

        :param aSolution: solution modified
        :type aSolution: :class:`biogeme.vns.solutionClass`

        :param aNeighbor: neighbor
        :type aNeighbor: :class:`biogeme.vns.solutionClass`
        """


class solutionClass(metaclass=abc.ABCMeta):
    """
    This is an abstract class defining a solution.
    """

    def __init__(self):
        self.objectives = None
        self.objectivesNames = None

    @abc.abstractmethod
    def __repr__(self):
        pass

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    @abc.abstractmethod
    def __str__(self):
        pass

    def dominates(self, anotherSolution):
        """Checks if the current solution dominates another one. It is
        assumed that all objectives are minimized.

        :param anotherSolution: the other solution to be compared with.
        :type anotherSolution: solutionClass

        :return: True is self dominates anotherSolution, False otherwise.
        :rtype: bool

        :raise biogemeError: if the value of no value is availaboe for
            the objective functions of ego.

        :raise biogemeError: if the value of no value is availaboe for
            the objective functions of alter.

        :raise biogemeError: if the number of objective functions are
            incompatible.

        """
        if self.objectives[0] is None:
            raise excep.biogemeError(
                'No values are available for '
                'the objective functions of ego.'
            )

        if anotherSolution.objectives[0] is None:
            raise excep.biogemeError(
                'No values are available for '
                'the objective functions of alter.'
            )

        if len(self.objectives) != len(anotherSolution.objectives):
            raise excep.biogemeError(
                f'Incompatible sizes: '
                f'{len(self.objectives)}'
                f' and {len(anotherSolution.objectives)}'
            )

        for my_f, her_f in zip(self.objectives, anotherSolution.objectives):
            if my_f > her_f:
                return False

        return self.objectives != anotherSolution.objectives


class operatorsManagement:
    """
    Class managing the selection and performance analysis of the operators
    """

    def __init__(self, operatorNames):
        """
        :param operatorNames: names of operators
        :type operators: list(str)
        """
        self.scores = {k: 0 for k in operatorNames}
        """ dict of scores obtained by the operators"""

        self.names = list(self.scores)
        """ Names of the operators """

        self.available = {k: True for k in operatorNames}
        """ dict of availability of the operators """

    def increaseScore(self, operator):
        """Increase the score of an operator.

        :param operator: name of the operator
        :type operator: str

        :raise biogemeError: if the operator is not known.
        """
        try:
            self.scores[operator] += 1
        except KeyError as e:
            raise excep.biogemeError(f'Unknown operator: {operator}') from e

    def decreaseScore(self, operator):
        """Decrease the score of an operator. If it has already the minimum
        score, it increases the others.

        :param operator: name of the operator
        :type operator: str

        :raise biogemeError: if the operator is not known.
        """
        try:
            self.scores[operator] -= 1
        except KeyError as e:
            raise excep.biogemeError(f'Unknown operator: {operator}') from e

    def selectOperator(self, minProbability=0.01, scale=0.1):
        """Select an operator based on the scores

        :param minProbability: minimum probability to select any
                               operator. This is meant to avoid
                               degeneracy, that is to have operators
                               that have no chance to be chosen

        :type minProbability: float

        :param scale: parameter for the choice probability
        :type scale: float

        :return: name of the selected operator
        :rtype: str

        :raise biogemeError: if the minimum probability is too large
            for the number of operators. It must be less or equal to 1 /
            N, where N is the number of operators.

        """

        if minProbability > 1.0 / len(self.scores):
            raise excep.biogemeError(
                f'Cannot impose min. probability '
                f'{minProbability} '
                f'with {len(self.scores)} operators'
            )

        maxscore = max(list(self.scores.values()))
        listOfScores = np.array(
            [
                np.exp(scale * (s - maxscore)) if self.available[k] else 0
                for k, s in self.scores.items()
            ]
        )
        if len(listOfScores) == 0:
            return None
        totalScore = sum(listOfScores)
        prob = np.array([float(s) / float(totalScore) for s in listOfScores])
        # Enforce minimum probability
        ok = prob >= minProbability
        tooLow = prob < minProbability
        notzero = prob != 0.0
        update = tooLow & notzero

        reservedTotal = update.sum() * minProbability
        totalHighScores = listOfScores[ok].sum()
        prob[ok] = (1.0 - reservedTotal) * listOfScores[ok] / totalHighScores
        prob[update] = minProbability
        return np.random.choice(self.names, 1, p=prob)[0]


class paretoClass:
    """Class managing the solutions"""

    def __init__(self, maxNeighborhood, archiveInputFile=None):
        """

        :param maxNeighborhood: the maximum size of the neighborhood
            that must be considered
        :type maxNeighborhood: int

        :param archiveInputFile: name of a pickle file contaning sets
                                 from previous runs
        :type archiveInputFile: str

        """

        self.maxNeighborhood = maxNeighborhood
        """the maximum size of the neighborhood that must be considered
        """

        if archiveInputFile is None:
            self.pareto = {}
            """dict containing the pareto set. The keys are the solutions
            (class :class:`biogeme.vns.solutionClass`), and the values are
            the size of the neighborhood that must be applied the next
            time tat the solution is selected by the algorithm.

            """

            self.removed = set()
            """set of solutions that have been in the Pareto set ar some point,
            but have been removed because dominated by another
            solution.
            """
            self.considered = set()
            """set of solutions that have been considered at some point by the
            algorithm
            """
        else:
            if self.restore(archiveInputFile):
                logger.general(
                    f'Pareto set initialized from file with '
                    f'{self.length()} elements.'
                )
            else:
                logger.general(
                    f'Unable to read file {archiveInputFile}. '
                    f'Pareto set empty.'
                )
                self.pareto = {}
                self.removed = set()
                self.considered = set()

    def length(self):
        """
        Obtain the length of the parato set.
        """
        return len(self.pareto)

    def changeNeighborhood(self, sol):
        """
        Change the size of the neighborhood for a solution in the Pareto set.

        :param sol: solution for which the neighborhood size must be increased.
        :type sol: solutionClass
        """
        if sol not in self.pareto:
            return
        self.pareto[sol] += 1

    def resetNeighborhood(self, sol):
        """
        Reset the size of the neighborhood to 1 for a solution.

        :param sol: solution for which the neighborhood size must be reset.
        :type sol: solutionClass
        """
        if sol not in self.pareto:
            return
        self.pareto[sol] = 1

    def add(self, elem):
        """

        - We define the set D as the set of members of the current
          Pareto set that are dominated by the element elem.

        - We define the set S as the set of members of the current
          Pareto set that dominate the element elem.

        If S is empty, we add elem to the Pareto set, and remove all
        elements in D.

        :param elem: elem to be considered for inclusion in the Pareto set.
        :type elem: solutionClass

        :return: True is elem has been included. False otherwise.
        :rtype: bool

        """
        self.considered.add(elem)
        S_dominated = set()
        D_dominating = set()
        for k in self.pareto:
            if elem.dominates(k):
                D_dominating.add(k)
            if k.dominates(elem):
                S_dominated.add(k)
        if S_dominated:
            return False
        self.pareto[elem] = 1
        self.pareto = {
            k: v for k, v in self.pareto.items() if k not in D_dominating
        }
        self.removed |= D_dominating
        return True

    def select(self):
        """
        Select a candidate to be modified during the next iteration.

        :return: a candidate and the neghborhoodsize
        :rtype: tuple(solutionClass, int)

        """

        # Candidates are members of the Pareto set that have not
        # reached the maximum neighborhood size.

        candidates = [
            (k, v) for k, v in self.pareto.items() if v < self.maxNeighborhood
        ]

        if not candidates:
            return None, None

        aSolution, neighborhoodSize = random.choice(candidates)

        return aSolution, neighborhoodSize

    def dump(self, fileName):
        """
        Dump the three sets on a file using pickle

        :param fileName: name of the file where the sets must be saved.
        :type fileName: str

        :raise biogemeError: if a problem has occured during dumping.
        """

        allSets = self.pareto, self.removed, self.considered
        with open(fileName, 'wb') as f:
            pickle.dump(allSets, f)

        with open(fileName, 'rb') as f:
            tmppareto, _, _ = pickle.load(f)
        if len(tmppareto) != len(self.pareto):
            raise excep.biogemeError(
                f'Problem in dumping: {len(tmppareto)} '
                f'instead of {len(self.pareto)}'
            )

    def restore(self, fileName):
        """Restore the Pareto from a file"""
        try:
            with open(fileName, 'rb') as f:
                self.pareto, self.removed, self.considered = pickle.load(f)
        except FileNotFoundError:
            return False

        return True


def vns(
    problem,
    firstSolutions,
    maxNeighborhood=20,
    numberOfNeighbors=10,
    archiveInputFile=None,
    pickleOutputFile=None,
):
    """Multi objective Variable Neighborhood Search

    :param problem: problem description
    :type problem: problemClass

    :param firstSolutions: several models to initialize  the algorithm
    :type firstSolutions: list(solutionClass)

    :param maxNeighborhood: the maximum size of the neighborhood
        that must be considered. Default: 20
    :type maxNeighborhood: int

    :param numberOfNeighbors: if none of this neighbors improves the
                              solution, it is considered that a local
                              optimum has been reached.
    :type numberOfNeighbors: int

    :param archiveInputFile: name of pickle file containing
                             specifications from a previous run of the
                             algorithm. Default: None
    :type archiveInputFile: str

    :param pickleOutputFile: name of file where the set of models are stored:
    :type pickleOutputFile: str

    :return: the pareto set, the set of models that have been in the
             pareto set and then removed, and the set of all models
             considered by the algorithm.
    :rtype: class :class:`biogeme.vns.paretoClass`

    :raise biogemeError: if the first Pareto set is empty.

    """

    if pickleOutputFile is None:
        pickleOutputFile = '__defaultPareto.pickle'

    thePareto = paretoClass(maxNeighborhood, archiveInputFile)

    if firstSolutions is not None:
        for s in firstSolutions:
            valid, why = problem.isValid(s)
            if valid:
                problem.evaluate(s)
                thePareto.add(s)
                logger.detailed(problem.describe(s))
            else:
                logger.warning(problem.describe(s))
                logger.warning(f'Default specification is invalid: {why}')

    if thePareto.length() == 0:
        raise excep.biogemeError(
            'Cannot start the algorithm with an empty Pareto set.'
        )

    logger.general(f'Initial pareto: {thePareto.length()}')
    thePareto.dump(pickleOutputFile)

    solutionToImprove, neighborhoodSize = thePareto.select()

    while solutionToImprove is not None:
        continueWithSolution = True
        n = 0
        while continueWithSolution:
            logger.general(f'----> Neighbor {n} of size {neighborhoodSize}')
            aNeighbor, numberOfChanges = problem.generateNeighbor(
                solutionToImprove, neighborhoodSize
            )

            n += 1
            if n == numberOfNeighbors:
                thePareto.changeNeighborhood(solutionToImprove)
                continueWithSolution = False

            if numberOfChanges == 0:
                logger.warning(
                    f'Neighbor of size {neighborhoodSize}: '
                    f'generation failed'
                )
                thePareto.changeNeighborhood(solutionToImprove)
                continueWithSolution = False
            elif aNeighbor in thePareto.considered:
                problem.neighborRejected(solutionToImprove, aNeighbor)
                logger.general(
                    f'*** Neighbor of size {neighborhoodSize}:'
                    f' already considered***'
                )
                thePareto.changeNeighborhood(solutionToImprove)
            else:
                problem.evaluate(aNeighbor)
                valid, why = problem.isValid(aNeighbor)
                if valid:
                    if thePareto.add(aNeighbor):
                        logger.general('*** New pareto solution: ')
                        logger.detailed(problem.describe(aNeighbor))
                        problem.neighborAccepted(solutionToImprove, aNeighbor)
                        thePareto.dump(pickleOutputFile)
                        thePareto.resetNeighborhood(solutionToImprove)
                        continueWithSolution = False
                    else:
                        logger.general(
                            f'*** Neighbor of size '
                            f'{neighborhoodSize}: dominated***'
                        )
                        problem.neighborRejected(solutionToImprove, aNeighbor)
                        thePareto.changeNeighborhood(solutionToImprove)
                else:
                    logger.general(
                        f'*** Neighbor of size {neighborhoodSize}'
                        f' invalid: {why}***'
                    )
                    problem.neighborRejected(solutionToImprove, aNeighbor)
                    thePareto.changeNeighborhood(solutionToImprove)

        solutionToImprove, neighborhoodSize = thePareto.select()

    return thePareto
