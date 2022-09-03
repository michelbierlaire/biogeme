""" Implements various models.

:author: Michel Bierlaire
:date: Fri Mar 29 17:13:14 2019
"""

# Too constraining
# pylint: disable=invalid-name
# pylint: disable=too-many-lines

import biogeme.exceptions as excep
import biogeme.messaging as msg

import biogeme.expressions as expr

logger = msg.bioMessage()


def loglogit(V, av, i):
    """The logarithm of the logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param i: id of the alternative for which the probability must be
              calculated.
    :type i: int

    :return: choice probability of alternative number i.
    :rtype: biogeme.expressions.expr.Expression
    """

    if av is None:
        return expr._bioLogLogitFullChoiceSet(V, av=None, choice=i)

    return expr._bioLogLogit(V, av, i)


def logit(V, av, i):
    """The logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param i: id of the alternative for which the probability must be
              calculated.
    :type i: int

    :return: choice probability of alternative number i.
    :rtype: biogeme.expressions.expr.Expression

    """
    if av is None:
        return expr.exp(expr._bioLogLogitFullChoiceSet(V, av=None, choice=i))

    return expr.exp(expr._bioLogLogit(V, av, i))


def boxcox(x, ell):
    """Box-Cox transform

    .. math:: B(x, \\ell) = \\frac{x^{\\ell}-1}{\\ell}.

    It has the property that

    .. math:: \\lim_{\\ell \\to 0} B(x,\\ell)=\\log(x).

    To avoid numerical difficulties, if :math:`\\ell < 10^{-5}`,
    the McLaurin approximation is used:

    .. math:: \\log(x) + \\ell \\log(x)^2 + \\frac{1}{6} \\ell^2 \\log(x)^3
              + \\frac{1}{24} \\ell^3 \\log(x)^4.

    :param x: a variable to transform.
    :type x: biogeme.expressions.expr.Expression
    :param ell: parameter of the transformation.
    :type ell: biogeme.expressions.expr.Expression

    :return: the Box-Cox transform
    :rtype: biogeme.expressions.expr.Expression
    """

    regular = (x**ell - 1.0) / ell
    mclaurin = (
        expr.log(x)
        + ell * expr.log(x) ** 2
        + ell**2 * expr.log(x) ** 3 / 6.0
        + ell**3 * expr.log(x) ** 4 / 24.0
    )
    smooth = expr.Elem({0: regular, 1: mclaurin}, ell < expr.Numeric(1.0e-5))
    return expr.Elem({0: smooth, 1: expr.Numeric(0)}, x == 0)


def piecewise(variable, thresholds):
    """Obsolete function. Present for compatibility only"""
    errorMsg = (
        'The function "piecewise" is obsolete and has been replaced '
        'by "piecewiseVariables". Its use has changed. Please refer '
        'to the documentation.'
    )
    raise excep.biogemeError(errorMsg)


def piecewiseVariables(variable, thresholds):
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
    :type variable: biogeme.expressions.expr.Expression or str

    :param thresholds: list of thresholds
    :type thresholds: list(float)

    :return: list of variables to for the piecewise linear specification.
    :rtype: list(biogeme.expressions.expr.Expression)

    :raise biogemeError: if the thresholds are not defined properly,
        as only the first and the last thresholds can be set
        to None.

    .. seealso:: :meth:`piecewiseFormula`

    """
    eye = len(thresholds)
    if all(t is None for t in thresholds):
        errorMsg = (
            'All thresholds for the piecewise linear specification '
            'are set to None.'
        )
        raise excep.biogemeError(errorMsg)
    if None in thresholds[1:-1]:
        errorMsg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise excep.biogemeError(errorMsg)

    # If the name of the variable is given, we transform it into an expression.
    if isinstance(variable, str):
        variable = expr.Variable(variable)

    # First variable
    if thresholds[0] is None:
        results = [expr.bioMin(variable, thresholds[1])]
    else:
        b = thresholds[1] - thresholds[0]
        results = [
            expr.bioMax(
                expr.Numeric(0), expr.bioMin(variable - thresholds[0], b)
            )
        ]

    for i in range(1, eye - 2):
        b = thresholds[i + 1] - thresholds[i]
        results += [
            expr.bioMax(
                expr.Numeric(0), expr.bioMin(variable - thresholds[i], b)
            )
        ]

    # Last variable
    if thresholds[-1] is None:
        results += [expr.bioMax(0, variable - thresholds[-2])]
    else:
        b = thresholds[-1] - thresholds[-2]
        results += [
            expr.bioMax(
                expr.Numeric(0), expr.bioMin(variable - thresholds[-2], b)
            )
        ]
    return results


def piecewiseFormula(variable, thresholds, betas=None):
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

    New variables and new parameters are automatically created.

    :param variable: name of the variable for which we need the
        piecewise linear transform.
    :type variable: string

    :param thresholds: list of thresholds
    :type thresholds: list(float)

    :param betas: list of beta parameters to be used in the
        specification.  The number of entries should be the number of
        thresholds, minus one. If None, for each interval, the
        parameter Beta('beta_VAR_interval',0, None, None, 0) is used,
        where var is the name of the variable. Default: none.
    :type betas:
        list(biogeme.expresssions.Beta)

    :return: expression of  the piecewise linear specification.
    :rtype: biogeme.expressions.expr.Expression

    :raise biogemeError: if the thresholds are not defined properly,
        which means that only the first and the last threshold can be set
        to None.

    :raise biogemeError: if the length of list ``initialexpr.Betas`` is
        not equal to the length of ``thresholds`` minus one.

    .. seealso:: :meth:`piecewiseVariables`

    """
    if isinstance(variable, expr.Variable):
        the_variable = variable
        the_name = variable.name
    elif isinstance(variable, str):
        the_name = variable
        the_valiable = expr.Variable(f'{variable}')
    else:
        errorMsg = (
            'The first argument of piecewiseFormula must be the '
            'name of a variable, or the variable itself..'
        )
        raise excep.biogemeError(errorMsg)

    eye = len(thresholds)
    if all(t is None for t in thresholds):
        errorMsg = (
            'All thresholds for the piecewise linear specification '
            'are set to None.'
        )
        raise excep.biogemeError(errorMsg)
    if None in thresholds[1:-1]:
        errorMsg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise excep.biogemeError(errorMsg)
    if betas is not None:
        if len(betas) != eye - 1:
            errorMsg = (
                f'As there are {eye} thresholds, a total of {eye-1} '
                f'Beta parameters are needed, and not {len(betas)}.'
            )
            raise excep.biogemeError(errorMsg)

    theVars = piecewiseVariables(expr.Variable(f'{variable}'), thresholds)
    if betas is None:
        betas = []
        for i, a_threshold in enumerate(thresholds[:-1]):
            next_threshold = thresholds[i + 1]
            a_name = 'minus_inf' if a_threshold is None else f'{a_threshold}'
            next_name = (
                'inf' if next_threshold is None else f'{next_threshold}'
            )
            betas.append(
                expr.Beta(
                    f'beta_{variable}_{a_name}_{next_name}', 0, None, None, 0
                )
            )

    terms = [beta * theVars[i] for i, beta in enumerate(betas)]

    return expr.bioMultSum(terms)


def piecewiseFunction(x, thresholds, betas):
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

    :param betas: list of the beta parameters.  The number of entries
                         should be the number of thresholds, plus
                         one.
    :type betas: list(float)

    :return: value of the numpy function
    :rtype: float

    :raise biogemeError: if the thresholds are not defined properly,
        which means that only the first and the last threshold can be set
        to None.


    """
    eye = len(thresholds)
    if all(t is None for t in thresholds):
        errorMsg = (
            'All thresholds for the piecewise linear specification '
            'are set to None.'
        )
        raise excep.biogemeError(errorMsg)
    if None in thresholds[1:-1]:
        errorMsg = (
            'For piecewise linear specification, only the first and '
            'the last thresholds can be None'
        )
        raise excep.biogemeError(errorMsg)
    if len(betas) != eye - 1:
        errorMsg = (
            f'As there are {eye} thresholds, a total of {eye-1} values '
            f'are needed to initialize the parameters. But '
            f'{len(betas)} are provided'
        )
        raise excep.biogemeError(errorMsg)

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


def logmev(V, logGi, av, choice):
    """Log of the choice probability for a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param logGi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
            (e^{V_1},\\ldots,e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type logGi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability of the MEV model, given by
    :rtype: biogeme.expressions.expr.Expression

    .. math:: V_i + \\ln G_i(e^{V_1},\\ldots,e^{V_J}) -
              \\ln\\left(\\sum_j e^{V_j + \\ln G_j(e^{V_1},
              \\ldots,e^{V_J})}\\right)

    """
    H = {i: v + logGi[i] for i, v in V.items()}
    if av is None:
        logP = expr._bioLogLogitFullChoiceSet(H, av=None, choice=choice)
    else:
        logP = expr._bioLogLogit(H, av, choice)
    return logP


def mev(V, logGi, av, choice):
    """Choice probability for a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)


    :param logGi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type logGi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: Choice probability of the MEV model, given by

    .. math:: \\frac{e^{V_i + \\ln G_i(e^{V_1},
              \\ldots,e^{V_J})}}{\\sum_j e^{V_j +
              \\ln G_j(e^{V_1},\\ldots,e^{V_J})}}

    :rtype: biogeme.expressions.expr.Expression
    """
    return expr.exp(logmev(V, logGi, av, choice))


def logmev_endogenousSampling(V, logGi, av, correction, choice):

    """Log of choice probability for a MEV model, including the
    correction for endogenous sampling as proposed by `Bierlaire, Bolduc
    and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
       http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param logGi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
                  (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type logGi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correstion terms
                       of each alternative.
    :type correction: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1}, \\ldots,e^{V_J}) + \\omega_i -
              \\ln\\left(\\sum_j e^{V_j +
              \\ln G_j(e^{V_1}, \\ldots, e^{V_J})+ \\omega_j}\\right)

    where :math:`\\omega_i` is the correction term for alternative :math:`i`.

    :rtype: biogeme.expressions.expr.Expression

    """
    H = {i: v + logGi[i] + correction[i] for i, v in V.items()}
    logP = expr._bioLogLogit(H, av, choice)
    return logP


def mev_endogenousSampling(V, logGi, av, correction, choice):
    """Choice probability for a MEV model, including the correction
    for endogenous sampling as proposed by
    `Bierlaire, Bolduc and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
           http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param logGi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type logGi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correstion terms
                       of each alternative.
    :type correction: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1}, \\ldots, e^{V_J}) + \\omega_i -
              \\ln\\left(\\sum_j e^{V_j + \\ln G_j(e^{V_1},\\ldots,e^{V_J})+
              \\omega_j}\\right)

    where :math:`\\omega_i` is the correction term for alternative :math:`i`.

    :rtype: biogeme.expressions.expr.Expression

    """
    return expr.exp(
        logmev_endogenousSampling(V, logGi, av, correction, choice)
    )


def getMevGeneratingForNested(V, availability, nests):
    """Implements the  MEV generating function for the nested logit model

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :return: a dictionary mapping each alternative id with the function

    .. math:: G(e^{V_1},
              \\ldots,e^{V_J}) =  \\sum_m \\left( \\sum_{\\ell \\in C_m}
              y_\\ell^{\\mu_m}\\right)^{\\frac{\\mu}{\\mu_m}}

    where :math:`G` is the MEV generating function.

    :rtype: biogeme.expressions.expr.Expression

    """

    termsForNests = []
    for m in nests:
        if availability is None:
            sumdict = [expr.exp(m[0] * V[i]) for i in m[1]]
        else:
            sumdict = [
                expr.Elem(
                    {0: 0.0, 1: expr.exp(m[0] * V[i])},
                    availability[i] != expr.Numeric(0),
                )
                for i in m[1]
            ]
        theSum = expr.bioMultSum(sumdict)
        termsForNests.append(theSum**1.0 / m[0])
    return expr.bioMultSum(termsForNests)


def getMevForNested(V, availability, nests):
    """Implements the derivatives of MEV generating function for the
    nested logit model

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :return: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}(e^{V_1},
              \\ldots,e^{V_J}) = e^{(\\mu_m-1)V_i}
              \\left(\\sum_{i=1}^{J_m} e^{\\mu_m V_i}\\right)^
              {\\frac{1}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.

    :rtype: dict(int:biogeme.expressions.expr.Expression)

    """

    logGi = {}
    for m in nests:
        if availability is None:
            sumdict = [expr.exp(m[0] * V[i]) for i in m[1]]
        else:
            sumdict = [
                expr.Elem(
                    {0: 0.0, 1: expr.exp(m[0] * V[i])},
                    availability[i] != expr.Numeric(0),
                )
                for i in m[1]
            ]
        theSum = expr.bioMultSum(sumdict)
        for i in m[1]:
            logGi[i] = (m[0] - 1.0) * V[i] + (1.0 / m[0] - 1.0) * expr.log(
                theSum
            )
    return logGi


def getMevForNestedMu(V, availability, nests, mu):
    """Implements the MEV generating function for the nested logit model,
    including the scale parameter

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability
        of each alternative, indexed
        by numerical ids. Must be consistent with V, or
        None. In this case, all alternatives are supposed to be
        always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA, [1, 2, 3]
           nestb = MUB, [4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :param mu: scale parameter
    :type mu: biogeme.expressions.expr.Expression

    :return: a dictionary mapping each alternative id with the function

        .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
                  \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
                  e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.

    :rtype: dict(int:biogeme.expressions.expr.Expression)

    """

    logGi = {}
    for m in nests:
        if availability is None:
            sumdict = [expr.exp(m[0] * V[i]) for i in m[1]]
        else:
            sumdict = [
                expr.Elem(
                    {0: 0.0, 1: expr.exp(m[0] * V[i])}, availability[i] != 0
                )
                for i in m[1]
            ]
        theSum = expr.bioMultSum(sumdict)
        for i in m[1]:
            logGi[i] = (
                expr.log(mu)
                + (m[0] - 1.0) * V[i]
                + (mu / m[0] - 1.0) * expr.log(theSum)
            )
    return logGi


def nested(V, availability, nests, choice):
    """Implements the nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability
                         of each alternative, indexed by numerical
                         ids. Must be consistent with V, or None. In
                         this case, all alternatives are supposed to
                         be always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests. Each item is also
        a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a list containing the list of identifiers of the
          alternatives belonging to the nest.

        Example::

            nesta = MUA, [1, 2, 3]
            nestb = MUB, [4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: choice probability for the nested logit model,
             based on the derivatives of the MEV generating function produced
             by the function getMevForNested

    :rtype: biogeme.expressions.expr.Expression

    :raise biogemeError: if the definition of the nests is invalid.
    """

    ok, message = checkValidityNestedLogit(V, nests)
    if not ok:
        raise excep.biogemeError(message)

    logGi = getMevForNested(V, availability, nests)
    P = mev(V, logGi, availability, choice)
    return P


def lognested(V, availability, nests, choice):
    """Implements the log of a nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
        alternative (:math:`a_i` in the above formula), indexed
        by numerical ids. Must be consistent with V, or
        None. In this case, all alternatives are supposed to be
        always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA, [1, 2, 3]
            nestb = MUB, [4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of choice probability for the nested logit model,
             based on the derivatives of the MEV generating function produced
             by the function getMevForNested

    :rtype: biogeme.expressions.expr.Expression

    :raise biogemeError: if the definition of the nests is invalid.
    """
    ok, message = checkValidityNestedLogit(V, nests)
    if not ok:
        raise excep.biogemeError(message)
    logGi = getMevForNested(V, availability, nests)
    logP = logmev(V, logGi, availability, choice)
    return logP


def nestedMevMu(V, availability, nests, choice, mu):
    """Implements the nested logit model as a MEV model, where mu is also
    a parameter, if the user wants to test different normalization
    schemes.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression  representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :param mu: expression producing the value of the top-level scale parameter.
    :type mu:  biogeme.expressions.expr.Expression

    :return: the nested logit choice probability based on the following
             derivatives of the MEV generating function:

    .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
              \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
              e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

    Where :math:`m` is the (only) nest containing alternative :math:`i`, and
    :math:`G` is the MEV generating function.

    :rtype: biogeme.expressions.expr.Expression

    """
    return expr.exp(lognestedMevMu(V, availability, nests, choice, mu))


def lognestedMevMu(V, availability, nests, choice, mu):
    """Implements the log of the nested logit model as a MEV model, where
    mu is also a parameter, if the user wants to test different
    normalization schemes.


    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression  representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA, [1, 2, 3]
            nestb = MUB, [4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :param mu: expression producing the value of the top-level scale parameter.
    :type mu:  biogeme.expressions.expr.Expression

    :return: the log of the nested logit choice probability based on the
        following derivatives of the MEV generating function:

        .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
                  \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
                  e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.

    :rtype: biogeme.expressions.expr.Expression

    """

    logGi = getMevForNestedMu(V, availability, nests, mu)
    logP = logmev(V, logGi, availability, choice)
    return logP


def cnl_avail(V, availability, nests, choice):
    """Same as cnl. Maintained for backward compatibility

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests. Each item is
        also a tuple containing two items

        - an object of type biogeme.expressions.expr.Expression  representing
          the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression
    """
    return cnl(V, availability, nests, choice)


def cnl(V, availability, nests, choice):
    """Implements the cross-nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression


    """
    return expr.exp(logcnl(V, availability, nests, choice))


def logcnl_avail(V, availability, nests, choice):
    """Same as logcnl. Maintained for backward compatibility

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression
    """
    return logcnl(V, availability, nests, choice)


def getMevForCrossNested(V, availability, nests):
    """Implements the MEV generating function for the cross nested logit
    model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int: biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
        alternative, indexed
        by numerical ids. Must be consistent with V, or
        None. In this case, all alternatives are supposed to be
        always available.

    :type availability: dict(int: biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: log of the choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression

    """

    Gi_terms = {}
    logGi = {}
    for i in V:
        Gi_terms[i] = []
    biosum = {}
    for m in nests:
        if availability is None:
            biosum = expr.bioMultSum(
                [
                    a ** (m[0]) * expr.exp(m[0] * (V[i]))
                    for i, a in m[1].items()
                ]
            )
        else:
            biosum = expr.bioMultSum(
                [
                    availability[i] * a ** (m[0]) * expr.exp(m[0] * (V[i]))
                    for i, a in m[1].items()
                ]
            )
        for i, a in m[1].items():
            Gi_terms[i] += [
                a ** (m[0])
                * expr.exp((m[0] - 1) * (V[i]))
                * biosum ** ((1.0 / m[0]) - 1.0)
            ]
    for k in V:
        logGi[k] = expr.log(expr.bioMultSum(Gi_terms[k]))
    return logGi


def logcnl(V, availability, nests, choice):
    """Implements the log of the cross-nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA , alphaA
            nestb = MUB , alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression

    :raise biogemeError: if the definition of the nests is invalid.
    """
    ok, message = checkValidityCNL(V, nests)
    if not ok:
        raise excep.biogemeError(message)
    if message != '':
        logger.warning(f'CNL: {message}')
    logGi = getMevForCrossNested(V, availability, nests)
    logP = logmev(V, logGi, availability, choice)
    return logP


def cnlmu(V, availability, nests, choice, mu):
    """Implements the cross-nested logit model as a MEV model with
    the homogeneity parameters is explicitly involved

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests. Each
        item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression representing
          the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :param mu: Homogeneity parameter :math:`\\mu`.
    :type mu: biogeme.expressions.expr.Expression

    :return: choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression
    """
    return expr.exp(logcnlmu(V, availability, nests, choice, mu))


def getMevForCrossNestedMu(V, availability, nests, mu):
    """Implements the MEV generating function for the cross-nested logit
    model as a MEV model with the homogeneity parameters is explicitly
    involved.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression representing
          the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param mu: Homogeneity parameter :math:`\\mu`.
    :type mu: biogeme.expressions.expr.Expression

    :return: log of the choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression

    """
    Gi_terms = {}
    logGi = {}
    for i in V:
        Gi_terms[i] = []
    biosum = {}
    for m in nests:
        if availability is None:
            biosum = expr.bioMultSum(
                [
                    a ** (m[0] / mu) * expr.exp(m[0] * (V[i]))
                    for i, a in m[1].items()
                ]
            )
        else:
            biosum = expr.bioMultSum(
                [
                    availability[i]
                    * a ** (m[0] / mu)
                    * expr.exp(m[0] * (V[i]))
                    for i, a in m[1].items()
                ]
            )
        for i, a in m[1].items():
            Gi_terms[i] += [
                a ** (m[0] / mu)
                * expr.exp((m[0] - 1) * (V[i]))
                * biosum ** ((mu / m[0]) - 1.0)
            ]
    for k in V:
        logGi[k] = expr.log(mu * expr.bioMultSum(Gi_terms[k]))
    return logGi


def logcnlmu(V, availability, nests, choice, mu):
    """Implements the log of the cross-nested logit model as a MEV model
    with the homogeneity parameters is explicitly involved.


    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests. Each item is
        also a tuple containing two items

        - an object of type biogeme.expressions. expr.Expression representing
          the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA , alphaA
            nestb = MUB , alphaB
            nests = nesta, nestb

    :type nests: tuple

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :param mu: Homogeneity parameter :math:`\\mu`.
    :type mu: biogeme.expressions.expr.Expression

    :return: log of the choice probability for the cross-nested logit model.
    :rtype: biogeme.expressions.expr.Expression

    :raise biogemeError: if the definition of the nests is invalid.

    """
    ok, message = checkValidityCNL(V, nests)
    if not ok:
        raise excep.biogemeError(message)
    logGi = getMevForCrossNestedMu(V, availability, nests, mu)
    logP = logmev(V, logGi, availability, choice)
    return logP


def checkValidityNestedLogit(V, nests):
    """Verifies if the nested logit model is indeed based on a partition
    of the choice set.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)
    :param nests: A tuple containing as many items as nests. Each item is also
        a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

          nesta = MUA, [1, 2, 3]
          nestb = MUB, [4, 5, 6]
          nests = nesta, nestb

    :type nests: tuple


    :return: a tuple ok, message, where the message explains the
             problem is the nested structure is not OK.
    :rtype: tuple(bool, str)

    """

    ok = True
    message = ''

    fullChoiceSet = {i for i, v in V.items()}
    unionOfNests = set.union(*[set(n[1]) for n in nests])
    if fullChoiceSet != unionOfNests:
        ok = False
        d1 = fullChoiceSet.difference(unionOfNests)
        d2 = unionOfNests.difference(fullChoiceSet)
        if d1:
            message += (
                f'Alternative(s) in the choice set, but not in any nest:'
                f' {d1}\n'
            )
        if d2:
            message += (
                f'Alternative(s) in a nest, but not in the choice set: '
                f'{d2}\n'
            )

    # Consider all pairs of nests and verify that the intersection is empty

    allPairs = [(n1, n2) for n1 in nests for n2 in nests if n1 != n2]
    for (n1, n2) in allPairs:
        inter = set(n1[1]).intersection(n2[1])
        if inter:
            ok = False
            message += (
                f'Two nests contain the following alternative(s): '
                f'{inter}\n'
            )

    if ok:
        message = 'The nested logit model is based on a partition. '

    return ok, message


def checkValidityCNL(V, nests):
    """Verifies if the cross-nested logit specifciation is valid

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items

        - an object of type biogeme.expressions.expr.Expression  representing
          the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: a tuple ok, message, where the message explains the
             problem is the nested structure is not OK.
    :rtype: tuple(bool, str)
    """

    ok = True
    message = ''

    alt = {i: [] for i in V}
    number = 0
    for mu, alpha in nests:
        for i, a in alpha.items():
            if a != 0.0:
                alt[i].append(a)
        number += 1

    problems_zero = []
    problems_one = []
    for i, ell in alt.items():
        if not ell:
            problems_zero.append(i)
            ok = False
        if len(ell) == 1 and isinstance(ell[0], expr.Expression):
            problems_one.append(i)

    if problems_zero:
        message += f'Alternative(s) not in any nest: {problems_zero}'

    if problems_one:
        message += (
            f' Alternative in exactly one nest, '
            f'and parameter alpha is defined by an '
            f'expression, and may not be constant: {problems_one}'
        )

    return ok, message
