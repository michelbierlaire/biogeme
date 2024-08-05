""" Representation of unknown parameters

:author: Michel Bierlaire
:date: Sat Apr 20 14:54:16 2024

"""

from __future__ import annotations

import logging

from biogeme.deprecated import deprecated
from biogeme.exceptions import BiogemeError
from biogeme.expressions import TypeOfElementaryExpression
from biogeme.expressions.elementary_expressions import Elementary


logger = logging.getLogger(__name__)


class Beta(Elementary):
    """
    Unknown parameters to be estimated from data.
    """

    def __init__(
        self,
        name: str,
        value: float,
        lowerbound: float | None,
        upperbound: float | None,
        status: int,
    ):
        """Constructor

        :param name: name of the parameter.
        :param value: default value.
        :param lowerbound: if different from None, imposes a lower
          bound on the value of the parameter during the optimization.
        :param upperbound: if different from None, imposes an upper
          bound on the value of the parameter during the optimization.
        :param status: if different from 0, the parameter is fixed to
          its default value, and not modified by the optimization algorithm.

        :raise BiogemeError: if the first parameter is not a str.

        :raise BiogemeError: if the second parameter is not an int or a float.
        """

        if not isinstance(value, (int, float)):
            error_msg = (
                f"The second parameter for {name} must be "
                f"a float and not a {type(value)}: {value}"
            )
            raise BiogemeError(error_msg)
        if not isinstance(name, str):
            error_msg = (
                f"The first parameter must be a string and "
                f"not a {type(name)}: {name}"
            )
            raise BiogemeError(error_msg)
        Elementary.__init__(self, name)
        self.initValue = value
        self.lb = lowerbound
        self.ub = upperbound
        self.status = status
        self.betaId = None

    def set_id_manager(self, id_manager: 'IdManager | None'):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may need to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        if id_manager is None:
            self.elementaryIndex = None
            self.betaId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        if self.status != 0:
            self.betaId = self.id_manager.fixed_betas.indices[self.name]
        else:
            self.betaId = self.id_manager.free_betas.indices[self.name]

    def __str__(self) -> str:
        return (
            f"Beta('{self.name}', {self.initValue}, {self.lb}, "
            f"{self.ub}, {self.status})"
        )

    def fix_betas(
        self,
        beta_values: dict[str, float],
        prefix: str | None = None,
        suffix: str | None = None,
    ):
        """Fix all the values of the Beta parameters appearing in the
        dictionary

        :param beta_values: dictionary containing the betas to be
            fixed (as key) and their value.
        :type beta_values: dict(str: float)

        :param prefix: if not None, the parameter is renamed, with a
            prefix defined by this argument.
        :type prefix: str

        :param suffix: if not None, the parameter is renamed, with a
            suffix defined by this argument.
        :type suffix: str

        """
        if self.name in beta_values:
            self.initValue = beta_values[self.name]
            self.status = 1
            if prefix is not None:
                self.name = f"{prefix}{self.name}"
            if suffix is not None:
                self.name = f"{self.name}{suffix}"

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
            return {self.name: self}

        if the_type == TypeOfElementaryExpression.FREE_BETA and self.status == 0:
            return {self.name: self}

        if the_type == TypeOfElementaryExpression.FIXED_BETA and self.status != 0:
            return {self.name: self}

        return {}

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the Beta is not fixed.
        """
        return self.initValue

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def change_init_values(self, betas: dict[str, float]):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        value = betas.get(self.name)
        if value is not None and value != self.initValue:
            if self.status != 0:
                warning_msg = (
                    f'Parameter {self.name} is fixed, but its value '
                    f'is changed from {self.initValue} to {value}.'
                )
                logger.warning(warning_msg)
            self.initValue = value

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the parameter,
            4. the status between [ ]
            5. the unique ID,  preceded by a comma
            6. the Beta ID,  preceded by a comma


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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            parameter
        """
        if self.elementaryIndex is None:
            error_msg = (
                f"No id has been defined for elementary " f"expression {self.name}."
            )
            raise BiogemeError(error_msg)
        if self.betaId is None:
            raise BiogemeError(f"No id has been defined for parameter {self.name}.")

        signature = f"<{self.get_class_name()}>"
        signature += f"{{{self.get_id()}}}"
        signature += (
            f'"{self.name}"[{self.status}],' f"{self.elementaryIndex},{self.betaId}"
        )
        return [signature.encode()]
