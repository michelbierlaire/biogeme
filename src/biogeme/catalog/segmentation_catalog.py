"""Defines  a segmentation catalog

Michel Bierlaire
Wed Apr 16 18:35:02 2025

"""

from __future__ import annotations

import logging
from itertools import product

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, NamedExpression
from biogeme.segmentation import DiscreteSegmentationTuple, Segmentation

from .catalog import Catalog
from .configuration import SELECTION_SEPARATOR, SEPARATOR
from .controller import Controller

logger = logging.getLogger(__name__)


def segmentation_catalogs(
    generic_name: str,
    beta_parameters: list[Beta],
    potential_segmentations: tuple[DiscreteSegmentationTuple, ...],
    maximum_number: int,
) -> list[Catalog]:
    """Generate catalogs for potential segmentations of a parameter

    :param generic_name: name used for the definition of the group of catalogs
    :param beta_parameters: list of parameters to be segmented
    :param potential_segmentations: tuple of potential segmentations
    :param maximum_number: maximum number of segmentations to consider
    """

    for segmentation in potential_segmentations:
        for key, value in segmentation.mapping.items():
            if SEPARATOR in value or SELECTION_SEPARATOR in value:
                error_msg = (
                    f'Invalid segment name for variable {segmentation.variable.name}='
                    f'{key}: [{value}]. Characters [{SEPARATOR}] and '
                    f'[{SELECTION_SEPARATOR}] are reserved for specification coding.'
                )
                raise BiogemeError(error_msg)

    def get_name_from_combination(combination: tuple[bool, ...]) -> str:
        """Assign a name to a combination"""
        if sum(combination) == 0:
            return 'no_seg'

        return '-'.join(
            [
                segment.variable.name
                for keep, segment in zip(combination, potential_segmentations)
                if keep
            ]
        )

    def get_expression_from_combination(
        the_beta_parameter: Beta, combination: tuple[bool, ...]
    ) -> Expression:
        """Assign an expression to a combination"""
        if sum(combination) == 0:
            # No segmentation
            return the_beta_parameter

        selected_expressions = (
            segment
            for keep, segment in zip(combination, potential_segmentations)
            if keep
        )

        the_segmentation = Segmentation(the_beta_parameter, selected_expressions)
        return the_segmentation.segmented_beta()

    if not isinstance(beta_parameters, list):
        error_msg = (
            f'A list is expected for beta_parameters, and not an object of type '
            f'{type(beta_parameters)}'
        )
        raise BiogemeError(error_msg)

    list_of_possibilities = [
        combination
        for combination in product([False, True], repeat=len(potential_segmentations))
        if sum(combination) <= maximum_number
    ]
    catalogs = []
    names = [
        get_name_from_combination(combination) for combination in list_of_possibilities
    ]
    the_controller = Controller(controller_name=generic_name, specification_names=names)
    for beta_parameter in beta_parameters:
        named_expressions = [
            NamedExpression(
                name=get_name_from_combination(combination),
                expression=get_expression_from_combination(beta_parameter, combination),
            )
            for combination in list_of_possibilities
        ]
        name = f'segmented_{beta_parameter.name}'
        catalog = Catalog(
            catalog_name=name,
            named_expressions=named_expressions,
            controlled_by=the_controller,
        )
        catalogs.append(catalog)
    return catalogs


class SegmentedParameters:
    """Class managing the names of segmented and alternative specific parameters"""

    def __init__(self, beta_parameters: list[Beta], alternatives: tuple[str, ...]):
        """Constructor"""

        # The parameters are organized as follows:
        #   - all generic parameters,
        #   - all parameters associated with the first alternative,
        #   - all parameters associated with the second alternative,
        #   - etc.
        self.beta_parameters: list[Beta] = beta_parameters
        self.all_parameters: list[Beta] = beta_parameters.copy()
        self.alternatives: tuple[str, ...] = alternatives
        for alternative in self.alternatives:
            self.all_parameters += [
                Beta(
                    name=f'{beta.name}_{alternative}',
                    value=beta.init_value,
                    lowerbound=beta.lower_bound,
                    upperbound=beta.upper_bound,
                    status=beta.status,
                )
                for beta in beta_parameters
            ]

    def get_index(self, beta_index: int, alternative: str | None):
        """Returns the index in the list of the Beta parameter with
            the given index specific to the given alternative

        :param beta_index: index of the Beta in the generic list
        :type beta_index: int

        :param alternative: name of the alternative, or None for the generic parameter
        :type alternative: str or None

        """
        if alternative is None:
            return beta_index
        alt_index = self.alternatives.index(alternative)
        return beta_index + (alt_index + 1) * len(self.beta_parameters)

    def get_beta(self, beta_index: int, alternative: str | None):
        """Return the Beta parameters for the given index and given alternative

        :param beta_index: index of the Beta in the generic list
        :type beta_index: int

        :param alternative: name of the alternative, or None for the generic parameter
        :type alternative: str or None

        """
        return self.all_parameters[self.get_index(beta_index, alternative)]
