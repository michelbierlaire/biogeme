""" Arithmetic expressions accepted by Biogeme: logit

:author: Michel Bierlaire
:date: Sat Sep  9 15:28:39 2023
"""
import logging
import numpy as np
import biogeme.exceptions as excep
from .base_expressions import Expression
from .numeric_tools import is_numeric
from .numeric_expressions import Numeric

logger = logging.getLogger(__name__)


class LogLogit(Expression):
    """Expression capturing the logit formula.

    It contains one formula for the target alternative, a dict of
    formula for the availabilities and a dict of formulas for the
    utilities

    """

    def __init__(self, util, av, choice):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param av: dictionary where the keys are the identifiers of
                   the alternatives, and the elements are object of
                   type biogeme.expressions.Expression defining the
                   availability conditions. If av is None, all the
                   alternatives are assumed to be always available

        :type av: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)
        self.util = {}  #: dict of utility functions
        for i, e in util.items():
            if is_numeric(e):
                self.util[i] = Numeric(e)
            else:
                if not isinstance(e, Expression):
                    raise excep.BiogemeError(f'This is not a valid expression: {e}')
                self.util[i] = e
        self.av = {}  #: dict of availability formulas
        if av is None:
            self.av = {k: Numeric(1) for k, v in util.items()}
        else:
            for i, e in av.items():
                if is_numeric(e):
                    self.av[i] = Numeric(e)
                else:
                    if not isinstance(e, Expression):
                        raise excep.BiogemeError(f'This is not a valid expression: {e}')
                    self.av[i] = e
        if is_numeric(choice):
            self.choice = Numeric(choice)
            """expression for the chosen alternative"""
        else:
            if not isinstance(choice, Expression):
                raise excep.BiogemeError(f'This is not a valid expression: {choice}')
            self.choice = choice

        self.children.append(self.choice)
        for i, e in self.util.items():
            self.children.append(e)
        for i, e in self.av.items():
            self.children.append(e)

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors = []
        list_of_warnings = []
        for e in self.children:
            err, war = e.audit(database)
            list_of_errors += err
            list_of_warnings += war

        if self.util.keys() != self.av.keys():
            the_error = 'Incompatible list of alternatives in logit expression. '
            consistent = False
            myset = self.util.keys() - self.av.keys()
            if myset:
                mysetContent = ', '.join(f'{str(k)} ' for k in myset)
                the_error += (
                    'Id(s) used for utilities and not for ' 'availabilities: '
                ) + mysetContent
            myset = self.av.keys() - self.util.keys()
            if myset:
                mysetContent = ', '.join(f'{str(k)} ' for k in myset)
                the_error += (
                    ' Id(s) used for availabilities and not ' 'for utilities: '
                ) + mysetContent
            list_of_errors.append(the_error)
        else:
            consistent = True
        listOfAlternatives = list(self.util)
        if database is None:
            choices = np.array([self.choice.getValue_c()])
        else:
            choices = database.valuesFromDatabase(self.choice)
        correctChoices = np.isin(choices, listOfAlternatives)
        indexOfIncorrectChoices = np.argwhere(~correctChoices)
        if indexOfIncorrectChoices.any():
            incorrectChoices = choices[indexOfIncorrectChoices]
            content = '-'.join(
                '{}[{}]'.format(*t)
                for t in zip(indexOfIncorrectChoices, incorrectChoices)
            )
            truncate = 100
            if len(content) > truncate:
                content = f'{content[:truncate]}...'
            the_error = (
                f'The choice variable [{self.choice}] does not '
                f'correspond to a valid alternative for the '
                f'following observations (rownumber[choice]): '
            ) + content
            list_of_errors.append(the_error)

        if consistent:
            if database is None:
                value_choice = self.choice.getValue_c()
                if value_choice not in self.av.keys():
                    the_error = (
                        f'The chosen alternative [{value_choice}] ' f'is not available'
                    )
                    list_of_warnings.append(the_error)
            else:
                choiceAvailability = database.checkAvailabilityOfChosenAlt(
                    self.av, self.choice
                )
                indexOfUnavailableChoices = np.where(~choiceAvailability)[0]
                if indexOfUnavailableChoices.size > 0:
                    incorrectChoices = choices[indexOfUnavailableChoices]
                    content = '-'.join(
                        '{}[{}]'.format(*t)
                        for t in zip(indexOfUnavailableChoices, incorrectChoices)
                    )
                    truncate = 100
                    if len(content) > truncate:
                        content = f'{content[:truncate]}...'
                    the_error = (
                        f'The chosen alternative [{self.choice}] '
                        f'is not available for the following '
                        f'observations (rownumber[choice]): '
                    ) + content
                    list_of_warnings.append(the_error)

        return list_of_errors, list_of_warnings

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the chosen alternative does not correspond
            to any of the utility functions

        :raise BiogemeError: if the chosen alternative does not correspond
            to any of entry in the availability condition

        """
        choice = int(self.choice.getValue())
        if choice not in self.util:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of utility functions: {self.util.keys()}'
            )
            raise excep.BiogemeError(error_msg)
        if choice not in self.av:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of availabilities: {self.av.keys()}'
            )
            raise excep.BiogemeError(error_msg)
        if self.av[choice].getValue() == 0.0:
            return -np.log(0)
        Vchosen = self.util[choice].getValue()
        denom = 0.0
        for i, V in self.util.items():
            if self.av[i].getValue() != 0.0:
                denom += np.exp(V.getValue() - Vchosen)
        return -np.log(denom)

    def __str__(self):
        s = self.getClassName()
        s += f'[choice={self.choice}]'
        s += 'U=('
        first = True
        for i, e in self.util.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        s += 'av=('
        first = True
        for i, e in self.av.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        return s

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of alternatives between ( )
            5. the id of the expression for the chosen alternative, preceeded
               by a comma.
            6. for each alternative, separated by commas:

                 a. the number of the alternative, as defined by the user,
                 b. the id of the expression for the utility,
                 c. the id of the expression for the availability condition.

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
          \\frac{\\exp(-\\beta_2 V_2) }{ \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        listOfSignatures = []
        for e in self.get_children():
            listOfSignatures += e.getSignature()
        signature = f'<{self.getClassName()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.util)})'
        signature += f',{self.choice.get_id()}'
        for i, e in self.util.items():
            signature += f',{i},{e.get_id()},{self.av[i].get_id()}'
        listOfSignatures += [signature.encode()]
        return listOfSignatures


class _bioLogLogit(LogLogit):
    """log of logit formula

    This expression captures the logarithm of the logit formula. It
    contains one formula for the target alternative, a dict of formula
    for the availabilities and a dict of formulas for the utilities It
    uses only the C++ implementation.
    """


class _bioLogLogitFullChoiceSet(LogLogit):
    """This expression captures the logarithm of the logit formula, where
    all alternatives are supposed to be always available.

       It contains one formula for the target alternative and a dict of
       formulas for the utilities. It uses only the C++ implementation.

    """
