"""Class that provides some automatic specification for segmented parameters

Michel Bierlaire
Thu Apr 3 12:07:44 2025
"""

from __future__ import annotations
from collections.abc import Iterable
from itertools import product
from typing import NamedTuple, TYPE_CHECKING

import pandas as pd

from .one_segmentation import OneSegmentation
from .segmentation_context import DiscreteSegmentationTuple
from biogeme.exceptions import BiogemeError
from biogeme.results_processing import EstimationResults

if TYPE_CHECKING:
    from biogeme.expressions import Beta, Expression, MultipleSum


class Segmentation:
    """Segmentation of a parameter, possibly with multiple socio-economic variables"""

    def __init__(
        self,
        beta: Beta,
        segmentation_tuples: Iterable[DiscreteSegmentationTuple],
        prefix: str = 'segmented',
    ):
        """Ctor

        :param beta: parameter to be segmented
        :param segmentation_tuples: characterization of the segmentations
        :param prefix: prefix to be used to generate the name of the
            segmented parameter
        """
        segmentation_tuples = tuple(segmentation_tuples)
        if not segmentation_tuples:
            raise BiogemeError('segmentation_tuples cannot be empty')
        self.beta: Beta = beta
        self.segmentations: tuple[OneSegmentation, ...] = tuple(
            OneSegmentation(beta, s) for s in segmentation_tuples
        )
        self.prefix = prefix

    def get_beta_ref_name(self) -> str:
        """
        Add a suffix to the name of the parameter
        """
        return self.beta.name + '_ref'

    def beta_ref_code(self) -> str:
        """Constructs the Python code for the parameter

        :return: Python code
        :rtype: str
        """

        beta_name = f"'{self.get_beta_ref_name()}'"
        return (
            f'Beta({beta_name}, {self.beta.init_value}, {self.beta.lower_bound}, '
            f'{self.beta.upper_bound}, {self.beta.status})'
        )

    def get_reference_beta(self) -> Beta:
        """Obtain the reference beta"""
        from biogeme.expressions import Beta

        beta_name = self.get_beta_ref_name()
        return Beta(
            beta_name,
            self.beta.init_value,
            self.beta.lower_bound,
            self.beta.upper_bound,
            self.beta.status,
        )

    def segmented_beta(self) -> Expression:
        """Create an expressions that combines all the segments

        :return: combined expression
        :rtype: biogeme.expressions.Expression

        """
        from biogeme.expressions import MultipleSum

        ref_beta = self.get_reference_beta()
        terms = [ref_beta]
        terms += [
            element for s in self.segmentations for element in s.list_of_expressions()
        ]

        return MultipleSum(terms)

    def segmented_code(self) -> str:
        """Create the Python code for an expressions that combines all the segments

        :return: Python code for the combined expression
        :rtype: str
        """
        result = '\n'.join(
            [
                s.beta_code(c, assignment=True)
                for s in self.segmentations
                for c in s.mapping.values()
            ]
        )
        result += '\n'

        terms = [self.beta_ref_code()]
        terms += [element for s in self.segmentations for element in s.list_of_code()]

        if len(terms) == 1:
            result += terms[0]
        else:
            joined_terms = ', '.join(terms)
            result += f'{self.prefix}_{self.beta.name} = bioMultSum([{joined_terms}])'
        return result

    def calculates_estimated_values(
        self, estimation_results: EstimationResults
    ) -> pd.DataFrame:
        """Calculates the estimated values of the parameter for each segment.

        :param estimation_results: results of the estimation
        :return: a pandas data frame with the definition of the segments and the corresponding values for the
        coefficient
        """

        class SegmentationValue(NamedTuple):
            segmentation: OneSegmentation
            value: str

        all_segmentations = [
            list(
                SegmentationValue(segment, value)
                for value in segment.segmentation_tuple.mapping.values()
            )
            for segment in self.segmentations
        ]

        # Use itertools.product to generate all combinations
        beta_values = estimation_results.get_beta_values()
        ref_beta_name = self.get_reference_beta().name
        ref_value = beta_values[ref_beta_name]
        list_of_rows = []
        for combination in product(*all_segmentations):
            the_row = {
                element.segmentation.variable.name: element.value
                for element in combination
            }
            the_row['parameter estimate'] = ref_value
            for element in combination:
                the_name = element.segmentation.beta_name(category=element.value)
                if the_name != ref_beta_name:
                    the_value = beta_values[the_name]
                    the_row['parameter estimate'] += the_value
            list_of_rows.append(the_row)
        df = pd.DataFrame(list_of_rows)
        return df
