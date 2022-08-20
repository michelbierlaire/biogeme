"""Combine several arithmetic expressions and a database to obtain formulas

:author: Michel Bierlaire
:date: Sat Jul 30 12:36:40 2022
"""
import inspect
from collections import namedtuple
import biogeme.exceptions as excep
import biogeme.messaging as msg

ElementsTuple = namedtuple('ElementsTuple', 'expressions indices names')

logger = msg.bioMessage()


class IdManager:
    """Class combining managing the ids of an arithmetic expression."""

    def __init__(
        self, expressions, database, number_of_draws, force_new_ids=False
    ):
        """Ctor

        :param expressions: list of expressions
        :type expressions: list(biogeme.expressions.Expression)

        :param database: database with the variables as column names
        :type database: biogeme.database.Database

        :param number_of_draws: number of draws for Monte-Carlo integration
        :type number_of_draws: int

        :param force_new_ids: if True, new ids are calculated for all
            expressions, even if they have already an ID. If False,
            and some expressions already have an ID, an error is
            raised.
        :type force_new_ids: bool

        :raises biogemeError: if an expression contains a variable and
            no database is provided.

        """
        self.expressions = expressions
        self.database = database
        self.number_of_draws = number_of_draws
        self.elementary_expressions = None
        self.free_betas = None
        self.free_betas_values = None
        self.number_of_free_betas = 0
        self.fixed_betas = None
        self.fixed_betas_values = None
        self.bounds = None
        self.random_variables = None
        self.draws = None
        self.variables = None
        self.requires_draws = False
        for f in self.expressions:
            the_variables = f.setOfVariables()
            if the_variables and database is None:
                raise excep.biogemeError(
                    f'No database is provided and an expression '
                    f'contains variables: {the_variables}'
                )
            if f.embedExpression('MonteCarlo') or f.embedExpression(
                'bioDraws'
            ):
                self.requires_draws = True

        self.prepare()

    def __str__(self):
        return str(self.elementary_expressions.indices)

    def __repr__(self):
        return str(self.elementary_expressions.indices)

    def __eq__(self, other):
        return self.elementary_expressions == other.elementary_expressions

    def audit(self):
        """Performs various checks on the expressions.

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)
        """
        listOfErrors = []
        listOfWarnings = []
        if self.database.isPanel():
            dict_of_variables = (
                self.expressions.dictOfVariablesOutsidePanelTrajectory()
            )
            if dict_of_variables:
                err_msg = (
                    f'Error in the loglikelihood function. '
                    f'Some variables are not inside '
                    f'PanelLikelihoodTrajectory: '
                    f'{dict_of_variables.keys()} .'
                    f'If the database is organized as panel data, '
                    f'all variables must be used inside a '
                    f'PanelLikelihoodTrajectory. '
                    f'If it is not consistent with your model, '
                    f'generate a flat '
                    f'version of the data using the function '
                    f'`generateFlatPanelDataframe`.'
                )
                listOfErrors.append(err_msg)
        return listOfErrors, listOfWarnings

    def changeInitValues(self, betas):
        """Modifies the values of the pameters

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        def get_value(name):
            v = betas.get(name)
            if v is None:
                return self.free_betas.expressions[name].initValue
            return v

        self.free_betas_values = [get_value(x) for x in self.free_betas.names]

    def expressions_names_indices(self, dict_of_elements):
        """Assigns consecutive indices to expressions

        :param dict_of_elements: dictionary of expressions. The keys
            are the names.
        :type dict_of_elements: dict(str: biogeme.expressions.Expression)

        :return: a tuple with the original dictionary, the indices,
            and the sorted names.
        :rtype: ElementsTuple
        """
        indices = {}
        names = {}
        names = sorted(dict_of_elements)
        for i, v in enumerate(names):
            indices[v] = i

        return ElementsTuple(
            expressions=dict_of_elements, indices=indices, names=names
        )

    def prepare(self):
        """Extract from the formulas the literals (parameters,
        variables, random variables) and decide a numbering convention.

        The numbering is done in the following order:

        (i) free betas,
        (ii) fixed betas,
        (iii) random variables for numerical integration,
        (iv) random variables for Monte-Carlo integration,
        (v) variables

        The numbering convention will be performed for all expressions
        together, so that the same elementary expressions in several
        expressions will have the same index.


        """

        # Free parameters (to be estimated), sortedby alphabetical order
        expr = {}
        for f in self.expressions:
            d = f.dictOfBetas(free=True, fixed=False)
            expr = dict(expr, **d)

        self.free_betas = self.expressions_names_indices(expr)

        self.bounds = [
            (
                self.free_betas.expressions[b].lb,
                self.free_betas.expressions[b].ub,
            )
            for b in self.free_betas.names
        ]
        self.number_of_free_betas = len(self.free_betas.names)
        # Fixed parameters (not to be estimated), sorted by alphatical order.
        expr = {}
        for f in self.expressions:
            d = f.dictOfBetas(free=False, fixed=True)
            expr = dict(expr, **d)
        self.fixed_betas = self.expressions_names_indices(expr)

        # Random variables for numerical integration
        expr = {}
        for f in self.expressions:
            d = f.dictOfRandomVariables()
            expr = dict(expr, **d)
        self.random_variables = self.expressions_names_indices(expr)

        # Draws
        expr = {}
        for f in self.expressions:
            d = f.dictOfDraws()
            expr = dict(expr, **d)
        self.draws = self.expressions_names_indices(expr)

        # Variables
        # Here, we do not extract the variables from the
        # formulas. Instead, we use all the variables in the database.
        if self.database is not None:
            variables_names = list(self.database.data.columns.values)
            variables_indices = {}
            for i, v in enumerate(variables_names):
                variables_indices[v] = i
            self.variables = ElementsTuple(
                expressions=None,
                indices=variables_indices,
                names=variables_names,
            )
        else:
            self.variables = ElementsTuple(
                expressions=None, indices=None, names=[]
            )

        # Merge all the names
        elementary_expressions_names = (
            self.free_betas.names
            + self.fixed_betas.names
            + self.random_variables.names
            + self.draws.names
            + self.variables.names
        )

        if len(elementary_expressions_names) != len(
            set(elementary_expressions_names)
        ):
            duplicates = {
                x
                for x in elementary_expressions_names
                if elementary_expressions_names.count(x) > 1
            }
            error_msg = (
                f'The following elementary expressions are defined '
                f'more than once: {duplicates}.'
            )
            raise excep.biogemeError(error_msg)

        elementary_expressions_indices = {
            v: i for i, v in enumerate(elementary_expressions_names)
        }

        self.elementary_expressions = ElementsTuple(
            expressions=None,
            indices=elementary_expressions_indices,
            names=elementary_expressions_names,
        )

        self.free_betas_values = [
            self.free_betas.expressions[x].initValue
            for x in self.free_betas.names
        ]
        self.fixed_betas_values = [
            self.fixed_betas.expressions[x].initValue
            for x in self.fixed_betas.names
        ]

        if self.requires_draws:
            self.database.generateDraws(
                self.draws.expressions, self.draws.names, self.number_of_draws
            )

    def setDataMap(self, sample):
        """Specify the map of the panel data in the expressions

        :param sample: map of the panel data (see
            :func:`biogeme.database.Database.buildPanelMap`)
        :type sample: pandas.DataFrame
        """
        for f in self.expressions:
            f.cpp.setDataMap(sample)

    def setData(self, sample):
        """Specify the sample

        :param sample: map of the panel data (see
            :func:`biogeme.database.Database.buildPanelMap`)
        :type sample: pandas.DataFrame

        """
        for f in self.expressions:
            f.cpp.setData(sample)
