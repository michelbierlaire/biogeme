"""File pareto.py

:author: Michel Bierlaire, EPFL
:date: Wed Mar 15 11:06:57 2023

Implement a Pareto set for generic purposes.
"""

import logging
from datetime import date
import tomlkit as tk
import biogeme.exceptions as excep
import biogeme.version as bv
try:
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ModuleNotFoundError:
    CAN_PLOT=False

logger = logging.getLogger(__name__)


class SetElement:
    """Specify the elements of the Pareto set. Note that each
    objective is supposed to be minimized.

    """

    def __init__(self, element_id, objectives):
        """Ctor

        :param element_id: identifier of the element
        :type element_id: str

        :param objectives: values of the objective functions
        :type objectives: list(float)
        """
        self.element_id = element_id
        self.objectives = objectives

        for obj in objectives:
            if obj is None:
                raise excep.biogemeError(
                    f'All objectives ust be defined: {objectives}'
                )

    def __eq__(self, other):
        if isinstance(other, SetElement):
            if self.element_id == other.element_id:
                if self.objectives != other.objectives:
                    error_msg = (
                        f'Two elements named {self.element_id} have different '
                        f'objective values: {self.objectives} and '
                        f'{other.objectives}'
                    )
                    raise excep.biogemeError(error_msg)
                return True
        return False

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        return f'{self.element_id} {self.objectives}'

    def __repr__(self):
        return self.element_id

    def dominates(self, other):
        """Verifies if self dominates other.

        :param other: other element to check
        :type other: SetElement
        """
        if len(self.objectives) != len(other.objectives):
            raise excep.biogemeError(
                f'Incompatible sizes: '
                f'{len(self.objectives)}'
                f' and {len(other.objectives)}'
            )
        for my_f, her_f in zip(self.objectives, other.objectives):
            if my_f > her_f:
                return False

        return self.objectives != other.objectives


class Pareto:
    """This object manages a Pareto set for a list of objectives that
    are each minimized.
    """

    def __init__(self, filename=None):
        self.size_init_pareto = 0
        self.size_init_considered = 0
        self.filename = filename
        if filename is None:
            self.pareto = set()
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
            if self.restore():
                logger.debug('RESTORE PARETO FROM FILE')
                self.size_init_pareto = len(self.pareto)
                self.size_init_considered = len(self.considered)
                logger.info(
                    f'Pareto set initialized from file with '
                    f'{self.size_init_considered} elements '
                    f'[{self.size_init_pareto} Pareto].'
                )
                logger.debug('RESTORE PARETO FROM FILE: DONE')

            else:
                logger.info(f'Unable to read file {filename}. ' f'Pareto set empty.')
                self.pareto = set()
                self.removed = set()
                self.considered = set()

    def dump(self):
        """
        Dump the three sets on a file using pickle

        :raise biogemeError: if a problem has occured during dumping.
        """
        if self.filename is None:
            return
        doc = tk.document()
        doc.add(tk.comment(f'Biogeme {bv.getVersion()} [{bv.versionDate}]'))
        doc.add(tk.comment(f'File {self.filename} created on {date.today()}'))
        doc.add(tk.comment(f'{bv.author}, {bv.department}, {bv.university}'))

        pareto_table = tk.table()
        for elem in self.pareto:
            pareto_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Pareto'] = pareto_table

        considered_table = tk.table()
        for elem in self.considered:
            considered_table[elem.element_id] = [
                float(obj) for obj in elem.objectives
            ]
        doc['Considered'] = considered_table

        removed_table = tk.table()
        for elem in self.removed:
            removed_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Removed'] = removed_table

        with open(self.filename, 'w', encoding='utf-8') as f:
            print(tk.dumps(doc), file=f)

    def get_element_from_id(self, the_id):
        """Returns the element of a set given its ID

        :param the_id: identifiers of the element to return
        :type the_id: str

        :return: found element, or None if element not present
        :rtype: SetElement
        """
        found = {elem for elem in self.considered if elem.element_id == the_id}
        if len(found) == 0:
            return None
        if len(found) > 1:
            error_msg = f'There are {len(found)} elements with ID {the_id}'
            raise excep.biogemeError(error_msg)
        return next(iter(found))

    def restore(self):
        """Restore the Pareto from a file"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
                document = tk.parse(content)
        except FileNotFoundError:
            return False

        self.pareto = {
            SetElement(id, list(values)) for id, values in document['Pareto'].items()
        }
        self.considered = {
            SetElement(id, list(values))
            for id, values in document['Considered'].items()
        }
        self.removed = {
            SetElement(id, list(values)) for id, values in document['Removed'].items()
        }
        return True

    def length(self):
        """
        Obtain the length of the pareto sets.
        """
        return len(self.pareto)

    def length_of_all_sets(self):
        """
        Obtain the length of the three sets.
        """
        return len(self.pareto), len(self.considered), len(self.removed)

    def add(self, element):
        """

        - We define the set D as the set of members of the current
          Pareto set that are dominated by the element elem.

        - We define the set S as the set of members of the current
          Pareto set that dominate the element elem.

        If S is empty, we add elem to the Pareto set, and remove all
        elements in D.

        :param element: element to be considered for inclusion in the Pareto set.
        :type element: solutionClass

        :return: True if elemenet has been included. False otherwise.
        :rtype: bool

        """
        if element in self.considered:
            warning_msg = (
                f'Elem {element.element_id} has already been inserted in the set'
            )
            logger.warning(warning_msg)
            return False

        self.considered.add(element)
        S_dominated = set()
        D_dominating = set()
        for k in self.pareto:
            if element.dominates(k):
                D_dominating.add(k)
            if k.dominates(element):
                S_dominated.add(k)
        if S_dominated:
            return False
        self.pareto.add(element)
        self.pareto = {k for k in self.pareto if k not in D_dominating}
        self.removed |= D_dominating
        return True

    def plot(
        self,
        objective_x=0,
        objective_y=1,
        label_x=None,
        label_y=None,
        margin_x=5,
        margin_y=5,
        ax=None,
    ):
        """Plot the members of the set according to two
            objective functions.  They  determine the x- and
            y-coordinate of the plot.

        :param objective_x: index of the objective function to use for the x-coordinate.
        :param objective_x: int

        :param objective_y: index of the objective function to use for the y-coordinate.
        :param objective_y: int

        :param label_x: label for the x_axis
        :type label_x: str

        :param label_y: label for the y_axis
        :type label_y: str

        :param margin_x: margin for the x axis
        :type margin_x: int

        :param margin_y: margin for the y axis
        :type margin_y: int

        :param ax: matplotlib axis for the plot
        :type ax: matplotlib.Axes

        """
        if not CAN_PLOT:
            raise excep.biogemeError('Install matplotlib.')
        ax = ax or plt.gca()

        if self.length() == 0:
            raise excep.biogemeError('Cannot plot an empty Pareto set')

        first_elem = next(iter(self.pareto))
        number_of_objectives = len(first_elem.objectives)

        if number_of_objectives < 2:
            raise excep.biogemeError(
                'At least two objectives functions are required for the plot of '
                'the Pateto set.'
            )

        if objective_x >= number_of_objectives:
            error_msg = (
                f'Index of objective x is {objective_x}, but there are '
                f'only {number_of_objectives}. Give a number between 0 '
                'and {number_of_objectives-1}.'
            )
            raise excep.biogemeError(error_msg)

        par_x = [elem.objectives[objective_x] for elem in self.pareto]
        par_y = [elem.objectives[objective_y] for elem in self.pareto]

        con_x = [elem.objectives[objective_x] for elem in self.considered]
        con_y = [elem.objectives[objective_y] for elem in self.considered]

        rem_x = [elem.objectives[objective_x] for elem in self.removed]
        rem_y = [elem.objectives[objective_y] for elem in self.removed]

        ax.axis(
            [
                min(par_x) - margin_x,
                max(par_x) + margin_x,
                min(par_y) - margin_y,
                max(par_y) + margin_y,
            ]
        )
        ax.plot(par_x, par_y, 'o', label='Pareto')
        ax.plot(rem_x, rem_y, 'x', label='Removed')
        ax.plot(con_x, con_y, ',', label='Considered')
        if label_x is None:
            label_x = f'Objective {objective_x}'
        if label_y is None:
            label_y = f'Objective {objective_y}'

        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.legend()
