""" Implements piecewise linear formulation

:author: Michel Bierlaire
:date: Wed Oct 25 09:43:41 2023
"""

import logging

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    Beta,
    Variable,
    BinaryMin,
    BinaryMax,
    Numeric,
    MultipleSum,
    ExpressionOrNumeric,
)

logger = logging.getLogger(__name__)


def piecewise_variables(
    variable: Expression | str, thresholds: list[float]
) -> list[Expression]:
    """Generate the variables to include in a piecewise linear specification.

    If there are K thresholds, K-1 variables are generated. The first
    and last thresholds can be defined as None, corresponding to
    :math:`-\\infty` and :math:`+\\infty`,respectively. If :math:`t` is
    the variable of interest, for each interval :math:`[a:a+b[`, we
    define a variable defined as:

    .. math:: x_{Ti} =\\left\\{  \\begin{array}{ll} 0 & \\text{if }
              t < a \\\\ t-a & \\text{if } a \\leq t < a+b \\\\ b  &
              \\text{otherwise}  \\end{array}\\right. \\;\\;\\;x_{Ti} =
              \\max(0, \\min(t-a, b))

    :param variable: variable for which we need the piecewise linear
       transform. The expression itself or the name of the variable
       can be given.

    :param thresholds: list of thresholds

    :return: list of variables to for the piecewise linear specification.

    :raise BiogemeError: if the thresholds are not defined properly,
        as only the first and the last thresholds can be set
        to None.

    .. seealso:: :meth:`piecewiseFormula`

    """
    eye = len(thresholds)
    if eye == 0:
        error_msg = 'No threshold has been provided.'
        raise BiogemeError(error_msg)
    if all(t is None for t in thresholds):
        error_msg = (
            'All thresholds for the piecewise linear specification are set to None.'
        )
        raise BiogemeError(error_msg)
    if None in thresholds[1:-1]:
        error_msg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise BiogemeError(error_msg)

    # If the name of the variable is given, we transform it into an expression.
    if isinstance(variable, str):
        variable = Variable(variable)
    if not isinstance(variable, Variable):
        error_msg = f'Expression of type Variable expected, not {type(variable)}'
        raise BiogemeError(error_msg)

    # First variable
    if thresholds[0] is None:
        results = [BinaryMin(variable, thresholds[1])]
    else:
        b = thresholds[1] - thresholds[0]
        results = [BinaryMax(Numeric(0), BinaryMin(variable - thresholds[0], b))]

    for i in range(1, eye - 2):
        b = thresholds[i + 1] - thresholds[i]
        results += [BinaryMax(Numeric(0), BinaryMin(variable - thresholds[i], b))]

    # Last variable
    if thresholds[-1] is None:
        results += [BinaryMax(0, variable - thresholds[-2])]
    else:
        b = thresholds[-1] - thresholds[-2]
        results += [BinaryMax(Numeric(0), BinaryMin(variable - thresholds[-2], b))]
    return results


def piecewise_formula(
    variable: str | Variable,
    thresholds: list[float],
    betas: list[ExpressionOrNumeric] | None = None,
) -> Expression:
    """Generate the formula for a piecewise linear specification.

    If there are K thresholds, K-1 variables are generated. The first
    and last thresholds can be defined as None, corresponding to
    :math:`-\\infty` and :math:`+\\infty`, respectively. If :math:`t` is
    the variable of interest, for each interval :math:`[a:a+b[`, we
    define a variable defined as:

    .. math:: x_{Ti} =\\left\\{  \\begin{array}{ll} 0 & \\text{if }
              t < a \\\\ t-a & \\text{if } a \\leq t < a+b \\\\ b  &
              \\text{otherwise}  \\end{array}\\right. \\;\\;\\;x_{Ti} =
              \\max(0, \\min(t-a, b))

    New variables and new parameters are automatically created to
    obtain the specification

    .. math:: \\sum_{i=1}^{K-1} \\beta_i x_{Ti}

    :param variable: name of the variable for which we need the
        piecewise linear transform.

    :param thresholds: list of thresholds

    :param betas: list of Beta parameters to be used in the
        specification.  The number of entries should be the number of
        thresholds, minus one. If None, for each interval, the
        parameter Beta('beta_VAR_interval',0, None, None, 0) is used,
        where var is the name of the variable. Default: none.

    :return: expression of  the piecewise linear specification.

    :raise BiogemeError: if the thresholds are not defined properly,
        which means that only the first and the last threshold can be set
        to None.

    :raise BiogemeError: if the length of list ``initialexpr.Betas`` is
        not equal to the length of ``thresholds`` minus one.

    .. seealso:: :meth:`piecewiseVariables`

    """
    if isinstance(variable, Variable):
        the_variable = variable
        the_name = variable.name
    elif isinstance(variable, str):
        the_name = variable
        the_variable = Variable(f'{variable}')
    else:
        error_msg = (
            'The first argument of piecewiseFormula must be the '
            'name of a variable, or the variable itself..'
        )
        raise BiogemeError(error_msg)

    eye = len(thresholds)
    if all(t is None for t in thresholds):
        error_msg = (
            'All thresholds for the piecewise linear specification are set to None.'
        )
        raise BiogemeError(error_msg)
    if None in thresholds[1:-1]:
        error_msg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise BiogemeError(error_msg)
    if betas is not None:
        if len(betas) != eye - 1:
            error_msg = (
                f'As there are {eye} thresholds, a total of {eye-1} '
                f'Beta parameters are needed, and not {len(betas)}.'
            )
            raise BiogemeError(error_msg)

    the_vars = piecewise_variables(the_variable, thresholds)
    if betas is None:
        betas = []
        for i, a_threshold in enumerate(thresholds[:-1]):
            next_threshold = thresholds[i + 1]
            a_name = 'minus_inf' if a_threshold is None else f'{a_threshold}'
            next_name = 'inf' if next_threshold is None else f'{next_threshold}'
            betas.append(
                Beta(f'beta_{the_name}_{a_name}_{next_name}', 0, None, None, 0)
            )

    terms = [beta * the_vars[i] for i, beta in enumerate(betas)]

    return MultipleSum(terms)


def piecewise_as_variable(
    variable: str | Variable, thresholds: list[float], betas: list[Beta] | None = None
) -> Expression:
    """Generate the formula for a piecewise linear specification, seen
    as a transformed variable.

    If there are K thresholds, K-1 variables are generated. The first
    and last thresholds can be defined as None, corresponding to
    :math:`-\\infty` and :math:`+\\infty`, respectively. If :math:`t` is
    the variable of interest, for each interval :math:`[a:a+b[`, we
    define a variable defined as:

    .. math:: x_{Ti} =\\left\\{  \\begin{array}{ll} 0 & \\text{if }
              t < a \\\\ t-a & \\text{if } a \\leq t < a+b \\\\ b  &
              \\text{otherwise}  \\end{array}\\right. \\;\\;\\;x_{Ti} =
              \\max(0, \\min(t-a, b))

    The specification this is returned is

    .. math:: x_{T1} + \\sum_{i=2}^{K-1} \beta_i x_{Ti}

    :param variable: name of the variable for which we need the
        piecewise linear transform.
    :type variable: string

    :param thresholds: list of thresholds
    :type thresholds: list(float)

    :param betas: list of Beta parameters to be used in the
        specification.  The number of entries should be the number of
        thresholds, minus two. If None, for each interval, the
        parameter Beta('beta_VAR_interval',0, None, None, 0) is used,
        where var is the name of the variable. Default: none.
    :type betas:
        list(biogeme.expresssions.Beta)

    :return: expression of  the piecewise linear specification.
    :rtype: biogeme.expressions.expr.Expression

    :raise BiogemeError: if the thresholds are not defined properly,
        which means that only the first and the last threshold can be set
        to None.

    :raise BiogemeError: if the length of list ``initialexpr.Betas`` is
        not equal to the length of ``thresholds`` minus one.

    .. see also:: :meth:`piecewiseVariables`

    """
    if isinstance(variable, Variable):
        the_variable = variable
        the_name = variable.name
    elif isinstance(variable, str):
        the_name = variable
        the_variable = Variable(f'{variable}')
    else:
        error_msg = (
            'The first argument of piecewiseFormula must be the '
            'name of a variable, or the variable itself..'
        )
        raise BiogemeError(error_msg)

    eye = len(thresholds)
    if all(t is None for t in thresholds):
        error_msg = (
            'All thresholds for the piecewise linear specification are set to None.'
        )
        raise BiogemeError(error_msg)
    if None in thresholds[1:-1]:
        error_msg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise BiogemeError(error_msg)
    if betas is not None:
        if len(betas) != eye - 2:
            error_msg = (
                f'As there are {eye} thresholds, a total of {eye-2} '
                f'Beta parameters are needed, and not {len(betas)}.'
            )
            raise BiogemeError(error_msg)

    theVars = piecewise_variables(the_variable, thresholds)
    if betas is None:
        betas = []
        for i, a_threshold in enumerate(thresholds[1:-1]):
            next_threshold = thresholds[i + 2]
            a_name = 'minus_inf' if a_threshold is None else f'{a_threshold}'
            next_name = 'inf' if next_threshold is None else f'{next_threshold}'
            betas.append(
                Beta(f'beta_{the_name}_{a_name}_{next_name}', 0, None, None, 0)
            )

    terms = [beta * theVars[i] for i, beta in enumerate(betas)]

    return theVars[0] + MultipleSum(terms)


def piecewise_function(x: float, thresholds: list[float], betas: list[float]) -> float:
    """Plot a piecewise linear specification.

    If there are K thresholds, K-1 variables are generated. The first
    and last thresholds can be defined as None, corresponding to
    :math:`-\\infty` and :math:`+\\infty`, respectively. If :math:`t` is
    the variable of interest, for each interval :math:`[a:a+b[`, we
    define a variable defined as:

    .. math:: x_{Ti} =\\left\\{  \\begin{array}{ll} 0 & \\text{if }
              t < a \\\\ t-a & \\text{if } a \\leq t < a+b \\\\ b  &
              \\text{otherwise}  \\end{array}\\right. \\;\\;\\;x_{Ti} =
              \\max(0, \\min(t-a, b))

    :param x: value at which the piecewise specification must be avaluated
    :type x: float

    :param thresholds: list of thresholds
    :type thresholds: list(float)

    :param betas: list of the Beta parameters.  The number of entries
                         should be the number of thresholds, plus
                         one.
    :type betas: list(float)

    :return: value of the numpy function
    :rtype: float

    :raise BiogemeError: if the thresholds are not defined properly,
        which means that only the first and the last threshold can be set
        to None.


    """
    eye = len(thresholds)
    if all(t is None for t in thresholds):
        error_msg = (
            'All thresholds for the piecewise linear specification are set to None.'
        )
        raise BiogemeError(error_msg)
    if None in thresholds[1:-1]:
        error_msg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise BiogemeError(error_msg)
    if len(betas) != eye - 1:
        error_msg = (
            f'As there are {eye} thresholds, a total of {eye-1} values '
            f'are needed to initialize the parameters. But '
            f'{len(betas)} are provided'
        )
        raise BiogemeError(error_msg)

    # If the first threshold is not -infinity, we need to check if
    # x is beyond it.
    if thresholds[0] is not None:
        if x < thresholds[0]:
            return 0
    rest = x
    total = 0
    for i, v in enumerate(betas):
        if thresholds[i + 1] is None:
            total += v * rest
            return total
        if x < thresholds[i + 1]:
            total += v * rest
            return total
        total += v * (
            thresholds[i + 1] - (0 if thresholds[i] is None else thresholds[i])
        )
        rest = x - thresholds[i + 1]
    return total
