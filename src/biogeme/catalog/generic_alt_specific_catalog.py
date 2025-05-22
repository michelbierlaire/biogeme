"""Defines  a catalog containing generic and alternative specific specifications

Michel Bierlaire
Sun Apr 27 2025, 15:49:25
"""

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta
from biogeme.segmentation import DiscreteSegmentationTuple
from .catalog import Catalog
from .controller import Controller
from .segmentation_catalog import SegmentedParameters, segmentation_catalogs


def generic_alt_specific_catalogs(
    generic_name: str,
    beta_parameters: list[Beta],
    alternatives: tuple[str, ...],
    potential_segmentations: tuple[DiscreteSegmentationTuple, ...] | None = None,
    maximum_number=5,
):
    """Generate catalogs selecting generic or alternative specific coefficients

    :param generic_name: name associated with all the parameters in the catalog
    :type generic_name: str

    :param beta_parameters: coefficients of interest
    :type beta_parameters: list(biogeme.expressions.Beta)

    :param alternatives: names of the alternatives
    :type alternatives: tuple(str)

    :param potential_segmentations: tuple of potential segmentations, or None
    :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

    :param maximum_number: maximum number of segmentations to consider
    :type maximum_number: int

    :return: a list of catalogs for each alternative
    :rtype: list(dict(str: biogeme.catalog.Catalog))
    """
    if len(alternatives) < 2:
        error_msg = (
            f'An alternative specific specification requires at least 2 '
            f'alternatives, and not {len(alternatives)}'
        )
        raise BiogemeError(error_msg)

    if not isinstance(beta_parameters, list):
        error_msg = (
            f'Argument "beta_parameters" of function '
            f'"{generic_alt_specific_catalogs.__name__}" must be a list.'
        )
        raise BiogemeError(error_msg)

    wrong_indices = []
    for index, beta in enumerate(beta_parameters):
        if not isinstance(beta, Beta):
            wrong_indices.append(index)

    if wrong_indices:
        error_msg = (
            f'The entries at the following indices are not Beta expressions: '
            f'{wrong_indices}'
        )
        raise BiogemeError(error_msg)

    # We first generate the alternative specific versions of the parameters
    generic_parameters = beta_parameters
    the_segmented_parameters = SegmentedParameters(
        beta_parameters=generic_parameters,
        alternatives=alternatives,
    )

    # If applicable, we apply the potential segmentations
    if potential_segmentations:
        segmented_catalogs = segmentation_catalogs(
            generic_name=generic_name,
            beta_parameters=the_segmented_parameters.all_parameters,
            potential_segmentations=potential_segmentations,
            maximum_number=maximum_number,
        )

    def get_expression(param_index: int, alternative: str | None):
        """Returns either the parameter, or the segmented version if applicable"""

        if potential_segmentations:
            the_index = the_segmented_parameters.get_index(param_index, alternative)
            return segmented_catalogs[the_index]
        return the_segmented_parameters.get_beta(param_index, alternative)

    # We now control for generic or alternative specific with a single
    # controller for all catalogs
    the_controller = Controller(
        controller_name=f'{generic_name}_gen_altspec',
        specification_names=('generic', 'altspec'),
    )

    # We organize the catalogs as a list of dict
    results = []
    for index, beta in enumerate(beta_parameters):
        the_dict = {
            alternative: Catalog.from_dict(
                catalog_name=f'{beta.name}_{alternative}_gen_altspec',
                dict_of_expressions={
                    'generic': get_expression(index, None),
                    'altspec': get_expression(index, alternative),
                },
                controlled_by=the_controller,
            )
            for alternative in alternatives
        }
        results.append(the_dict)
    return results
