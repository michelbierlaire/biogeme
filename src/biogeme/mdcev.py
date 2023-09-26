"""Implementation of the contribution to the log likelihood of the
MDCEV model.

See the technical report for a description of the various versions.

The functions involved in this module have a similar list of arguments. We document them here once for all:

- number_of_chosen_alternatives: number of alternative with non zero
    consumption. Note that it would be possible to calculate it from
    the consumption_quantities, but it would be too expensive in terms
    of calculation. Ideally should be stored as a variable in the
    database. Typically of type Variable.

- consumed_quantities: a dictionary associating the id of the
    alternatives with the observed number of times they are
    chosen. Typically, the values are of type Variable.

- baseline_utilities: a dictionary of baseline utility functions for
    each alternative. Typically, they are linear in the parameters. In
    the document, it is expressed as the inner product of beta and x.

- alpha_parameters: a dictionary of expressions for the alpha
    parameters appearing the specification of some the utility
    functions. See the technical report for details.

- gamma_parameters: a dictionary of expressions for the gamma
    parameters appearing the specification of the utility
    function. See the technical report for details.

- prices: a dictionary of expressions for the prices of each
      alternative.
    
- scale_parameter: expression for the scale parameter. Usually a
    numeric constant or a parameter. If None, the scale parameter is
    assumed to be fixed to 1, and is not included in the formulation.



:author: Michel Bierlaire
:date: Wed Aug 23 16:29:10 2023

"""
from biogeme.expressions import exp, Elem, log, bioMultSum
from biogeme.mdcev_models import translated, generalized, gamma_profile, non_monotonic


def mdcev(
    number_of_chosen_alternatives,
    consumed_quantities,
    specific_model,
    scale_parameter=None,
):
    """Generate the Biogeme formula for the log probability of the MDCEV model

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
    :type number_of_chosen_alternatives: biogeme.expression.Expression

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param specific_model: a tuple containing dictionaries of expressions
        calculating

            - the utility functions.
            - the log of the entries of the determinant,
            - the inverse of the sames entries,
    :type specific_model: dict[int: biogeme.expression.Expression]

    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`
    :type scale_parameter: biogeme.expressions.Expression

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    # utility of chosen goods
    terms = [
        Elem({0: 0.0, 1: util}, consumed_quantities[i] > 0)
        for i, util in specific_model.utilities.items()
    ]
    if scale_parameter is None:
        baseline_term = bioMultSum(terms)
    else:
        baseline_term = scale_parameter * bioMultSum(terms)

    # Determinant: first term
    terms = [
        Elem({0: 0.0, 1: z}, consumed_quantities[i] > 0)
        for i, z in specific_model.log_determinant_entries.items()
    ]
    first_determinant = bioMultSum(terms)

    # Determinant: second term
    terms = [
        Elem({0: 0.0, 1: z}, consumed_quantities[i] > 0)
        for i, z in specific_model.inverse_of_determinant_entries.items()
    ]
    second_determinant = log(bioMultSum(terms))

    # Logsum
    if scale_parameter is None:
        terms = [exp(util) for util in specific_model.utilities.values()]
    else:
        terms = [
            scale_parameter * exp(util) for util in specific_model.utilities.values()
        ]
    logsum_term = number_of_chosen_alternatives * log(bioMultSum(terms))

    log_prob = baseline_term + first_determinant + second_determinant - logsum_term
    # Scale parameter
    if scale_parameter is not None:
        log_prob += (number_of_chosen_alternatives - 1) * scale_parameter

    return log_prob


def mdcev_translated(
    number_of_chosen_alternatives,
    consumed_quantities,
    baseline_utilities,
    alpha_parameters,
    gamma_parameters,
    scale_parameter=None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the translated utility function.

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
    :type number_of_chosen_alternatives: biogeme.expression.Expression

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`
    :type scale_parameter: biogeme.expressions.Expression

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = translated(
        baseline_utilities,
        consumed_quantities,
        alpha_parameters,
        gamma_parameters,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=scale_parameter,
    )


def mdcev_generalized(
    number_of_chosen_alternatives,
    consumed_quantities,
    baseline_utilities,
    alpha_parameters,
    gamma_parameters,
    prices=None,
    scale_parameter=None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the generalized translated utility function

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
    :type number_of_chosen_alternatives: biogeme.expression.Expression

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation :mod:`biogeme.mdcev`
    :type prices: biogeme.expressions.Expression

    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`
    :type scale_parameter: biogeme.expressions.Expression


    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = generalized(
        baseline_utilities,
        consumed_quantities,
        alpha_parameters,
        gamma_parameters,
        prices,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=scale_parameter,
    )


def mdcev_gamma(
    number_of_chosen_alternatives,
    consumed_quantities,
    baseline_utilities,
    gamma_parameters,
    prices=None,
    scale_parameter=None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the linear expenditure system.

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
    :type number_of_chosen_alternatives: biogeme.expression.Expression

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation :mod:`biogeme.mdcev`
    :type prices: biogeme.expressions.Expression

    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`
    :type scale_parameter: biogeme.expressions.Expression

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = gamma_profile(
        baseline_utilities,
        consumed_quantities,
        gamma_parameters,
        prices,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=scale_parameter,
    )


def mdcev_non_monotonic(
    number_of_chosen_alternatives,
    consumed_quantities,
    baseline_utilities,
    mu_utilities,
    alpha_parameters,
    gamma_parameters,
    prices=None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the linear expenditure system.

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
    :type number_of_chosen_alternatives: biogeme.expression.Expression

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation :mod:`biogeme.mdcev`
    :type prices: biogeme.expressions.Expression

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = non_monotonic(
        baseline_utilities,
        mu_utilities,
        consumed_quantities,
        alpha_parameters,
        gamma_parameters,
        prices,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=None,
    )
