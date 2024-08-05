""" Arithmetic expressions accepted by Biogeme: numeric expressions

:author: Michel Bierlaire
:date: Sat Sep  9 15:27:17 2023
"""

from __future__ import annotations
import logging


from .base_expressions import Expression
from ..deprecated import deprecated


logger = logging.getLogger(__name__)


class Numeric(Expression):
    """
    Numerical expression for a simple number
    """

    def __init__(self, value: float | int | bool):
        """Constructor

        :param value: numerical value
        :type value: float
        """
        Expression.__init__(self)
        self.value = float(value)  #: numeric value

    def __str__(self) -> str:
        return '`' + str(self.value) + '`'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.value

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the value, preceeded by a comma.

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
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f',{self.value}'
        return [signature.encode()]
