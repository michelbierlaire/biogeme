"""File vns.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 16 16:55:30 2020

Multi-objective variable neighborhood search algorithm
"""

import logging
import random
import abc
from collections import defaultdict
import numpy as np
import biogeme.exceptions as excep
from biogeme.pareto import Pareto

logger = logging.getLogger(__name__)


class ProblemClass(metaclass=abc.ABCMeta):
    """
    Abstract class defining a problem
    """

    def __init__(self, operators):
        self.operators_management = OperatorsManagement(operators)

    @abc.abstractmethod
    def is_valid(self, element):
        """Check the validity of the solution.

        :param element: solution to be checked
        :type element: :class:`biogeme.pareto.SetElement`

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. why contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """

    def generate_neighbor(self, element, neighborhood_size):
        """Generate a neighbor from the negihborhood of size
        ``neighborhood_size``using one of the operators

        :param element: current solution
        :type element: SetElement

        :param neighborhood_size: size of the neighborhood
        :type neighborhood_size: int

        :return: number of modifications actually made
        :rtype: int

        """
        # Select one operator.
        operator = self.operators_management.select_operator()
        return operator(element, neighborhood_size)

    def last_neighbor_rejected(self):
        """Notify that a neighbor has been rejected by the
        algorithm. Used to update the statistics on the operators.

        :param solution: solution  modified
        :type solution: :class:`biogeme.pareto.SetElement`

        :param a_neighbor: neighbor
        :type a_neighbor: :class:`biogeme.pareto.SetElement`
        """
        self.operators_management.decrease_score_last_operator()

    def last_neighbor_accepted(self):
        """Notify that a neighbor has been accepted by the
        algorithm. Used to update the statistics on the operators.

        :param solution: solution modified
        :type solution: :class:`biogeme.pareto.SetElement`

        :param a_neighbor: neighbor
        :type a_neighbor: :class:`biogeme.pareto.SetElement`
        """
        self.operators_management.increase_score_last_operator()


class OperatorsManagement:
    """
    Class managing the selection and performance analysis of the operators
    """

    def __init__(self, operators):
        """Ctor

        :param operators: dict where the keys are the names of the
            operators, and the values are the operators themselves. An
            operator is a function that takes two arguments (the
            current solution and the size of the neighborhood), and
            return a neightbor solution.
        :type operators: dict(str: fct)

        """
        self.operators = operators

        self.scores = {k: 0 for k in operators}
        """ dict of scores obtained by the operators"""

        self.names = list(operators.keys())
        """ Names of the operators """

        self.available = {k: True for k in operators}
        """ dict of availability of the operators """

        self.last_operator_name = None

    def increase_score_last_operator(self):
        """Increase the score of the last operator.

        :raise biogemeError: if the operator is not known.
        """
        try:
            self.scores[self.last_operator_name] += 1
        except KeyError as e:
            raise excep.biogemeError(
                f'Unknown operator: {self.last_operator_name}'
            ) from e

    def decrease_score_last_operator(self):
        """Decrease the score of the last operator. If it has already the minimum
        score, it increases the others.

        :raise biogemeError: if the operator is not known.
        """
        try:
            self.scores[self.last_operator_name] -= 1
        except KeyError as e:
            raise excep.biogemeError(
                f'Unknown operator: {self.last_operator_name}'
            ) from e

    def select_operator(self, min_probability=0.01, scale=0.1):
        """Select an operator based on the scores

        :param min_probability: minimum probability to select any
                               operator. This is meant to avoid
                               degeneracy, that is to have operators
                               that have no chance to be chosen

        :type min_probability: float

        :param scale: parameter for the choice probability
        :type scale: float

        :return: name of the selected operator
        :rtype: str

        :raise biogemeError: if the minimum probability is too large
            for the number of operators. It must be less or equal to 1 /
            N, where N is the number of operators.

        """

        if min_probability > 1.0 / len(self.scores):
            raise excep.biogemeError(
                f'Cannot impose min. probability '
                f'{min_probability} '
                f'with {len(self.scores)} operators. '
                f'Maximum is {1.0 / len(self.scores):.3f}.'
            )

        maxscore = max(list(self.scores.values()))
        list_of_scores = np.array(
            [
                np.exp(scale * (s - maxscore)) if self.available[k] else 0
                for k, s in self.scores.items()
            ]
        )
        if len(list_of_scores) == 0:
            return None
        total_score = sum(list_of_scores)
        prob = np.array([float(s) / float(total_score) for s in list_of_scores])
        # Enforce minimum probability
        ok = prob >= min_probability
        too_low = prob < min_probability
        notzero = prob != 0.0
        update = too_low & notzero

        reserved_total = update.sum() * min_probability
        total_high_scores = list_of_scores[ok].sum()
        prob[ok] = (1.0 - reserved_total) * list_of_scores[ok] / total_high_scores
        prob[update] = min_probability
        self.last_operator_name = np.random.choice(self.names, 1, p=prob)[0]
        return self.operators[self.last_operator_name]


class ParetoClass(Pareto):
    """Class managing the solutions"""

    def __init__(self, max_neighborhood, pareto_file=None):
        """

        :param max_neighborhood: the maximum size of the neighborhood
            that must be considered
        :type max_neighborhood: int

        :param archiveInputFile: name of a pickle file contaning sets
                                 from previous runs
        :type archiveInputFile: str

        """
        super().__init__(pareto_file)
        self.max_neighborhood = max_neighborhood
        """the maximum size of the neighborhood that must be considered
        """
        self.neighborhood_size = defaultdict(int)
        """ dict associating the solutions IDs with the neighborhoodsize"""

        for elem in self.considered:
            self.neighborhood_size[elem] = 1

    def change_neighborhood(self, element):
        """
        Change the size of the neighborhood for a solution in the Pareto set.

        :param element: ID of the solution for which the neighborhood size must be increased.
        :type element: SetElement
        """
        self.neighborhood_size[element] += 1

    def reset_neighborhood(self, element):
        """
        Reset the size of the neighborhood to 1 for a solution.

        :param element: ID of the solution for which the neighborhood size must be reset.
        :type element: biogeme.pareto.SetElement
        """
        self.neighborhood_size[element] = 1

    def add(self, element):
        """Add an element
        :param element: element to be considered for inclusion in the Pareto set.
        :type element: SetElement

        :return: True if solution has been included. False otherwise.
        :rtype: bool
        """
        added = super().add(element)
        if added:
            self.neighborhood_size[element] = 1
        else:
            self.neighborhood_size[element] += 1
        return added

    def select(self):
        """
        Select a candidate to be modified during the next iteration.

        :return: a candidate and the neghborhoodsize
        :rtype: tuple(SolutionClass, int)

        """

        # Candidates are members of the Pareto set that have not
        # reached the maximum neighborhood size.

        candidates = [
            (k, v)
            for k, v in self.neighborhood_size.items()
            if v < self.max_neighborhood
        ]

        if not candidates:
            return None, None
        element, size = random.choice(candidates)
        return element, size


def vns(
    problem,
    first_solutions,
    pareto,
    number_of_neighbors=10,
):
    """Multi objective Variable Neighborhood Search

    :param problem: problem description
    :type problem: ProblemClass

    :param first_solutions: several models to initialize  the algorithm
    :type first_solutions: list(biogeme.pareto.SetElement)

    :param pareto: object managing the Pareto set
    :type pareto: ParetoClass

    :param number_of_neighbors: if none of this neighbors improves the
                              solution, it is considered that a local
                              optimum has been reached.
    :type number_of_neighbors: int

    :return: the pareto set, the set of models that have been in the
             pareto set and then removed, and the set of all models
             considered by the algorithm.
    :rtype: class :class:`biogeme.vns.ParetoClass`

    :raise biogemeError: if the first Pareto set is empty.

    """

    if first_solutions is not None:
        for s in first_solutions:
            valid, why = problem.is_valid(s)
            if valid:
                pareto.add(s)
                logger.info(s)
            else:
                logger.warning(s)
                logger.warning(f'Default specification is invalid: {why}')

    if pareto.length() == 0:
        raise excep.biogemeError('Cannot start the algorithm with an empty Pareto set.')

    logger.info(f'Initial pareto: {pareto.length()}')

    solution_to_improve, neighborhood_size = pareto.select()

    while solution_to_improve is not None:
        continue_with_solution = True
        n = 0
        while continue_with_solution:
            logger.info(f'----> Current solution: {solution_to_improve}')
            logger.info(f'----> Neighbor {n} of size {neighborhood_size}')
            a_neighbor, number_of_changes = problem.generate_neighbor(
                solution_to_improve, neighborhood_size
            )
            logger.info(
                f'----> Neighbor: {a_neighbor} Nbr of changes {number_of_changes}'
            )

            n += 1
            if n == number_of_neighbors:
                pareto.change_neighborhood(solution_to_improve)
                continue_with_solution = False

            if number_of_changes == 0:
                logger.warning(
                    f'Neighbor of size {neighborhood_size}: ' f'generation failed'
                )
                pareto.change_neighborhood(solution_to_improve)
                continue_with_solution = False
            elif a_neighbor in pareto.considered:
                problem.last_neighbor_rejected()
                logger.info(
                    f'*** Neighbor of size {neighborhood_size}:'
                    f' already considered***'
                )
                pareto.change_neighborhood(solution_to_improve)
            else:
                valid, why = problem.is_valid(a_neighbor)
                if valid:
                    if pareto.add(a_neighbor):
                        logger.info('*** New pareto solution: ')
                        logger.info(a_neighbor)
                        problem.last_neighbor_accepted()
                        pareto.dump()
                        pareto.reset_neighborhood(solution_to_improve)
                        continue_with_solution = False
                    else:
                        logger.info(
                            f'*** Neighbor of size '
                            f'{neighborhood_size}: dominated***'
                        )
                        problem.last_neighbor_rejected()
                        pareto.change_neighborhood(solution_to_improve)
                else:
                    logger.info(
                        f'*** Neighbor of size {neighborhood_size}'
                        f' invalid: {why}***'
                    )
                    problem.last_neighbor_rejected()
                    pareto.change_neighborhood(solution_to_improve)

        solution_to_improve, neighborhood_size = pareto.select()

    return pareto
