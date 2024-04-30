""" Arithmetic expressions accepted by Biogeme: nary operators

:author: Michel Bierlaire
:date: Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations
import logging
from typing import NamedTuple, Iterable, TYPE_CHECKING, Any

from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .elementary_expressions import (
    Variable,
    TypeOfElementaryExpression,
    Elementary,
)
from .beta_parameters import Beta
from .convert import validate_and_convert
from ..deprecated import deprecated

if TYPE_CHECKING:
    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


class ConditionalTermTuple(NamedTuple):
    condition: ExpressionOrNumeric
    term: ExpressionOrNumeric


class ConditionalSum(Expression):
    """This expression returns the sum of a selected list of
    expressions. An expression is considered in the sum only if the
    corresponding key is True (that is, return a non-zero value).


    """

    def __init__(self, list_of_terms: Iterable[ConditionalTermTuple]):
        """Constructor

        :param list_of_terms: list containing the terms and the associated conditions

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        :raise BiogemeError: if the dict of expressions is empty
        :raise BiogemeError: if the dict of expressions is not a dict

        """
        if not list_of_terms:
            raise BiogemeError('The argument of ConditionalSum cannot be empty')

        Expression.__init__(self)

        self.list_of_terms = [
            the_term._replace(
                condition=validate_and_convert(the_term.condition),
                term=validate_and_convert(the_term.term),
            )
            for the_term in list_of_terms
        ]
        for the_term in self.list_of_terms:
            self.children.append(the_term.condition)
            self.children.append(the_term.term)

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        result = 0.0
        for the_term in self.list_of_terms:
            condition = the_term.condition.get_value()
            if condition != 0:
                result += the_term.term.get_value()
        return result

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def __str__(self) -> str:
        s = (
            'ConditionalSum('
            + ', '.join([f'{k}: {v}' for k, v in self.list_of_terms])
            + ')'
        )
        return s

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signature of the expression defining the key
            2. the signatures of all the children expressions,
            3. the name of the expression between < >
            4. the id of the expression between { }
            5. the number of elements between ( )
            6. the id of the expression defining the key
            7. for each element: the value of the key and the id
               of the expression, separated by commas.

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
        list_of_signatures = []
        for key, expression in self.list_of_terms:
            list_of_signatures += key.get_signature()
            list_of_signatures += expression.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.list_of_terms)})'
        for key, expression in self.list_of_terms:
            signature += f',{key.get_id()},{expression.get_id()}'
        list_of_signatures += [signature.encode()]
        return list_of_signatures


class bioMultSum(Expression):
    """This expression returns the sum of several other expressions.

    It is a generalization of 'Plus' for more than two terms
    """

    def __init__(
        self,
        list_of_expressions: list[ExpressionOrNumeric] | dict[Any:ExpressionOrNumeric],
    ):
        """Constructor

        :param list_of_expressions: list of objects representing the
                                     terms of the sum.

        :type list_of_expressions: list(biogeme.expressions.Expression)

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        :raise BiogemeError: if the list of expressions is empty
        :raise BiogemeError: if the list of expressions is neither a dict nor a list
        """
        if not list_of_expressions:
            raise BiogemeError('The argument of bioMultSum cannot be empty')

        Expression.__init__(self)

        if isinstance(list_of_expressions, dict):
            items = list_of_expressions.values()
        elif isinstance(list_of_expressions, list):
            items = list_of_expressions
        else:
            raise BiogemeError('Argument of bioMultSum must be a dict or a list.')

        for expression in items:
            self.children.append(validate_and_convert(expression))

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        result = 0.0
        for e in self.get_children():
            result += e.get_value()
        return result

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def __str__(self) -> str:
        s = 'bioMultSum(' + ', '.join([f'{e}' for e in self.get_children()]) + ')'
        return s


class Elem(Expression):
    """This returns the element of a dictionary. The key is evaluated
    from an expression and must return an integer, possibly negative.
    """

    def __init__(
        self,
        dict_of_expressions: dict[int, ExpressionOrNumeric],
        key_expression: ExpressionOrNumeric,
    ):
        """Constructor

        :param dict_of_expressions: dict of objects with numerical keys.
        :type dict_of_expressions: dict(int: biogeme.expressions.Expression)

        :param key_expression: object providing the key of the element
                              to be evaluated.
        :type key_expression: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)

        self.keyExpression = validate_and_convert(key_expression)
        self.children.append(self.keyExpression)

        self.dict_of_expressions = {}  #: dict of expressions
        for k, v in dict_of_expressions.items():
            self.dict_of_expressions[k] = validate_and_convert(v)
            self.children.append(self.dict_of_expressions[k])

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the calcuated key is not present in
            the dictionary.
        """
        key = int(self.keyExpression.get_value())
        if key in self.dict_of_expressions:
            return self.dict_of_expressions[key].get_value()

        error_msg = (
            f'Key {key} is not present in the dictionary. '
            f'Available keys: {self.dict_of_expressions.keys()}'
        )
        raise BiogemeError(error_msg)

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def __str__(self) -> str:
        s = '{{'
        first = True
        for k, v in self.dict_of_expressions.items():
            if first:
                s += f'{k}:{v}'
                first = False
            else:
                s += f', {k}:{v}'
        s += f'}}[{self.keyExpression}]'
        return s

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signature of the expression defining the key
            2. the signatures of all the children expressions,
            3. the name of the expression between < >
            4. the id of the expression between { }
            5. the number of elements between ( )
            6. the id of the expression defining the key
            7. for each element: the value of the key and the id
               of the expression, separated by commas.

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
        list_of_signatures = []
        list_of_signatures += self.keyExpression.get_signature()
        for i, e in self.dict_of_expressions.items():
            list_of_signatures += e.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.dict_of_expressions)})'
        signature += f',{self.keyExpression.get_id()}'
        for i, e in self.dict_of_expressions.items():
            signature += f',{i},{e.get_id()}'
        list_of_signatures += [signature.encode()]
        return list_of_signatures


class LinearTermTuple(NamedTuple):
    beta: Beta
    x: Variable


class bioLinearUtility(Expression):
    """When the utility function is linear, it is expressed as a list of
    terms, where a parameter multiplies a variable.
    """

    def __init__(self, list_of_terms: list[LinearTermTuple]):
        """Constructor

        :param list_of_terms: a list of tuple. Each tuple contains first
             a Beta parameter, second the name of a variable.
        :type list_of_terms: list(biogeme.expressions.Expression,
            biogeme.expressions.Expression)

        :raises biogeme.exceptions.BiogemeError: if the object is not
                        a list of tuples (parameter, variable)

        """
        Expression.__init__(self)

        the_error = ""
        first = True

        for b, v in list_of_terms:
            if not isinstance(b, Beta):
                if first:
                    the_error += (
                        'Each element of the bioLinearUtility '
                        'must be a tuple (parameter, variable). '
                    )
                    first = False
                the_error += f' Expression {b} is not a parameter.'
            if not isinstance(v, Variable):
                if first:
                    the_error += (
                        'Each element of the list should be '
                        'a tuple (parameter, variable).'
                    )
                    first = False
                the_error += f' Expression {v} is not a variable.'
        if not first:
            raise BiogemeError(the_error)

        self.betas, self.variables = zip(*list_of_terms)

        self.betas = list(self.betas)  #: list of parameters

        self.variables = list(self.variables)  #: list of variables

        self.listOfTerms = list(zip(self.betas, self.variables))
        """ List of terms """

        self.children += self.betas + self.variables

    def __str__(self) -> str:
        return ' + '.join([f'{b} * {x}' for b, x in self.listOfTerms])

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str, Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        if the_type == TypeOfElementaryExpression.BETA:
            return {x.name: x for x in self.betas}

        if the_type == TypeOfElementaryExpression.FREE_BETA:
            return {x.name: x for x in self.betas if x.status == 0}

        if the_type == TypeOfElementaryExpression.FIXED_BETA:
            return {x.name: x for x in self.betas if x.status != 0}

        if the_type == TypeOfElementaryExpression.VARIABLE:
            return {x.name: x for x in self.variables}

        return {}

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of terms in the utility ( )
            5. for each term:

                a. the id of the Beta parameter
                b. the unique id of the Beta parameter
                c. the name of the parameter
                d. the id of the variable
                e. the unique id of the variable
                f. the name of the variable

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
        list_of_signatures = []
        for e in self.get_children():
            list_of_signatures += e.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.listOfTerms)})'
        for b, v in self.listOfTerms:
            signature += (
                f',{b.get_id()},{b.elementaryIndex},{b.name},'
                f'{v.get_id()},{v.elementaryIndex},{v.name}'
            )
        list_of_signatures += [signature.encode()]
        return list_of_signatures
