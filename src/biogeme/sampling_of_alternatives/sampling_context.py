"""Defines a class that characterized the context to apply sampling of alternatives

:author: Michel Bierlaire
:date: Wed Sep  6 14:38:31 2023
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    list_of_variables_in_expression,
)
from biogeme.nests import NestsForCrossNestedLogit
from biogeme.partition import Partition, Segment

logger = logging.getLogger(__name__)

MEV_PREFIX = '_MEV_'
LOG_PROBA_COL = '_log_proba'
MEV_WEIGHT = '_mev_weight'
CNL_PREFIX = '_CNL_'


class StratumTuple(NamedTuple):
    """A stratum is an element of a partition of the full choice set,
    combined with the number of alternatives that must be sampled.
    """

    subset: Segment
    sample_size: int


class CrossVariableTuple(NamedTuple):
    """A cross variable is a variable that involves socio-economic
    attributes of the individuals, and attributes of the
    alternatives. It can only be calculated after the sampling has
    been made.
    """

    name: str
    formula: Expression


@dataclass
class SamplingContext:
    """Class gathering the data needed to perform an estimation with
    samples of alternatives

    :param the_partition: Partition used for the sampling.

    :param sample_sizes: number of alternative to draw from each segment.

    :param individuals: Pandas data frame containing all the
        individuals as rows. One column must contain the choice of
        each individual.

    :param choice_column: name of the column containing the choice of
        each individual.

    :param alternatives: Pandas data frame containing all the
        alternatives as rows. One column must contain a unique ID
        identifying the alternatives. The other columns contain
        variables to include in the data file.

    :param id_column: name of the column containing the Ids of the alternatives.

    :param utility_function: definition of the generic utility function

    :param combined_variables: definition of interaction variables

    :param mev_partition: If a second choice set need to be sampled
        for the MEV terms, the corresponding partition is provided
        here.

    """

    the_partition: Partition
    sample_sizes: Iterable[int]
    individuals: pd.DataFrame
    choice_column: str
    alternatives: pd.DataFrame
    id_column: str
    biogeme_file_name: str
    utility_function: Expression
    combined_variables: list[CrossVariableTuple]
    mev_partition: Partition | None = None
    mev_sample_sizes: Iterable[int] | None = None
    cnl_nests: NestsForCrossNestedLogit | None = None

    def check_expression(self, expression: Expression) -> None:
        """Verifies if the variables contained in the expression can be found in the databases"""
        variables = {
            var.name
            for var in list_of_variables_in_expression(the_expression=expression)
        }
        for variable in variables:
            if (
                variable not in self.individuals.columns
                and variable not in self.alternatives.columns
                and all(variable != t.name for t in self.combined_variables)
            ):
                error_msg = (
                    f'Invalid expression. Variable "{variable}" has not been found in '
                    f'the provided database'
                )
                raise BiogemeError(error_msg)

    def check_partition(self) -> None:
        """Check if the partition is truly a partition. If not, an exception is raised

        :raise BiogemeError: if some elements are present in more than one subset.

        :raise BiogemeError: if the size of the union of the subsets does
           not match the expected total size

        :raise BiogemeError: if an alternative in the partition does
            not appear in the database of alternatives

        :raise BiogemeError: if a segment is empty

        :raise BiogemeError: if the number of sampled alternatives in
            a stratum is incorrect , that is zero, or larger than the
            stratum size..

        """
        # Verify that all requested alternatives appear in the database of alternatives
        for stratum in self.sampling_protocol:
            n = len(stratum.subset)
            if n == 0:
                error_msg = 'A stratum is empty'
                raise BiogemeError(error_msg)
            k = stratum.sample_size
            if k > n:
                error_msg = f'Cannot draw {k} elements in a stratum of size {n}'
                raise BiogemeError(error_msg)

            if k == 0:
                error_msg = 'At least one alternative must be selected in each segment'
                raise BiogemeError(error_msg)

            for alt in stratum.subset:
                if alt not in self.alternatives[self.id_column].values:
                    error_msg = (
                        f'Alternative {alt} does not appear in the database of '
                        f'alternatives'
                    )
                    raise BiogemeError(error_msg)

    def check_mev_partition(self) -> None:
        """Check if the partition is a partition of the MEV
        alternatives. It does not need to cover the full choice set"""

        if self.mev_partition:
            if self.mev_sample_sizes is None:
                error_msg = (
                    'If mev_partition is defined, mev_sample_size must also be defined'
                )
                raise BiogemeError(error_msg)

        if self.mev_sample_sizes:
            if self.mev_partition is None:
                error_msg = (
                    'If mev_sample_sizes is defined, mev_partition must also be defined'
                )
                raise BiogemeError(error_msg)

        if self.cnl_nests and self.mev_partition:
            if self.cnl_nests.mev_alternatives != self.mev_partition.full_set:
                in_nest_not_in_partition = (
                    self.cnl_nests.mev_alternatives - self.mev_partition.full_set
                )
                in_partition_not_in_nest = (
                    self.mev_partition.full_set - self.cnl_nests.mev_alternatives
                )
                error_msg = ''
                if in_nest_not_in_partition:
                    error_msg += (
                        f'The following alternative(s) belong to a nest but not to the'
                        f' partition for the sample: {in_nest_not_in_partition}. '
                    )
                if in_partition_not_in_nest:
                    error_msg += (
                        f'The following alternative(s) belong to the partition for '
                        f'the MEV sample, but not to any nest: {in_partition_not_in_nest}'
                    )

    def check_valid_alternatives(self, set_of_ids: set[int]) -> None:
        """Check if the IDs in set are indeed valid
            alternatives. Typically used to check if a nest is well
            defined

        :param set_of_ids: set of identifiers to check

        :raise BiogemeError: if at least one id is invalid.
        """
        if (
            not pd.Series(list(set_of_ids))
            .isin(self.alternatives[self.id_column])
            .all()
        ):
            missing_values = set_of_ids - set(self.alternatives[self.id_column])
            raise BiogemeError(
                f'The following IDs are not valid alternative IDs: {missing_values}'
            )

    def include_cnl_alphas(self) -> None:
        if self.cnl_nests is None:
            return
        for nest in self.cnl_nests:
            column_name = f'{CNL_PREFIX}{nest.name}'
            self.alternatives[column_name] = self.alternatives[self.id_column].map(
                lambda x: (
                    self.cnl_nests.get_alpha_values(alternative_id=x)[nest.name]
                    if x in nest.dict_of_alpha
                    else 0.0
                )
            )

    def __post_init__(self) -> None:
        # Check for empty utility function
        if self.utility_function is None:
            raise BiogemeError('No utility function has been provided')

        # Check for empty strings
        if not self.choice_column:
            raise BiogemeError('choice_column should not be an empty string.')

        if not self.id_column:
            raise BiogemeError('id_column should not be an empty string.')

        # Validate that the DataFrames are not empty
        if self.individuals.empty or self.alternatives.empty:
            raise BiogemeError(
                'DataFrames individuals or alternatives should not be empty.'
            )

        self.check_partition()
        logger.debug('Check if there is a MEV partition')
        if self.mev_partition or self.mev_sample_sizes:
            logger.debug('Yes, there is a MEV partition')
            self.check_mev_partition()
        else:
            logger.debug('No, there is no MEV partition')

        # If CNL nests are defined, check that the alphas are all
        # fixed and that the nests have a name.
        if self.cnl_nests:
            if not self.cnl_nests.all_alphas_fixed():
                error_msg = 'For the CNL model, all alpha parameters must be fixed.'
                raise BiogemeError(error_msg)
            if not self.cnl_nests.check_names():
                error_msg = 'For the CNL model, all nests must have a name.'
                raise BiogemeError(error_msg)
        self.number_of_alternatives = self.alternatives.shape[0]
        self.number_of_individuals = self.individuals.shape[0]

        # Validate that choice_column is in the individuals DataFrame
        if self.choice_column not in self.individuals.columns:
            raise BiogemeError(
                f'{self.choice_column} is not a column in the individuals DataFrame.'
            )

        # Validate that id_column is in the alternatives DataFrame
        if self.id_column not in self.alternatives.columns:
            raise BiogemeError(
                f'{self.id_column} is not a column in the alternatives DataFrame.'
            )

        # Check for data types
        if not self.individuals[self.choice_column].dtype in [int, float]:
            raise BiogemeError(
                f'Column {self.choice_column} in data frame "individuals" should '
                f'be of type int or float.'
            )

        if not self.alternatives[self.id_column].dtype in [int, float]:
            raise BiogemeError(
                f'Column {self.id_column} in alternatives should be of type int or float.'
            )

        self.check_expression(self.utility_function)
        for cross_variable in self.combined_variables:
            self.check_expression(cross_variable.formula)

        self.include_cnl_alphas()

    @property
    def attributes(self) -> set[str]:
        """List of attributes for the choice model"""
        return set(self.alternatives.columns) | {
            combined_variable.name for combined_variable in self.combined_variables
        }

    @property
    def mev_prefix(self) -> str:
        """Build the prefix for the MEV columns"""
        return '' if self.mev_partition is None else MEV_PREFIX

    @property
    def sampling_protocol(self) -> list[StratumTuple]:
        """Provides a list of strata characterizing the sampling"""
        return [
            StratumTuple(subset=segment, sample_size=size)
            for segment, size in zip(self.the_partition, self.sample_sizes)
        ]

    @property
    def mev_sampling_protocol(self) -> list[StratumTuple] | None:
        """Provides a list of strata characterizing the MEV sampling"""
        if self.mev_partition is None:
            return None
        return [
            StratumTuple(subset=segment, sample_size=size)
            for segment, size in zip(self.mev_partition, self.mev_sample_sizes)
        ]

    @property
    def total_sample_size(self) -> int:
        """Sample size"""
        return sum(stratum.sample_size for stratum in self.sampling_protocol)

    @property
    def total_mev_sample_size(self) -> int:
        """Sample size"""
        if self.mev_partition is None:
            return 0
        return sum(stratum.sample_size for stratum in self.mev_sampling_protocol)

    def reporting(self) -> str:
        """Summarizes the configuration specified by the context object."""
        result = {
            'Size of the choice set': self.alternatives.shape[0],
            'Main partition': (
                f'{self.the_partition.number_of_segments()} segment(s) of size '
                f'{", ".join([str(len(segment)) for segment in self.the_partition])}'
            ),
            'Main sample': f'{self.total_sample_size}: ',
        }

        result['Main sample'] += ', '.join(
            [
                f'{stratum.sample_size}/{len(stratum.subset)}'
                for stratum in self.sampling_protocol
            ]
        )
        if self.mev_partition:
            result['Nbr of MEV alternatives'] = len(self.mev_partition.full_set)
            result['MEV partition'] = (
                f'{self.mev_partition.number_of_segments()} segment(s) of size '
                f'{", ".join([str(len(segment)) for segment in self.mev_partition])}'
            )
            result['MEV sample'] = f'{self.total_mev_sample_size}: '
            result['MEV sample'] += ', '.join(
                [
                    f'{stratum.sample_size}/{len(stratum.subset)}'
                    for stratum in self.mev_sampling_protocol
                ]
            )

        output = ''
        for section, description in result.items():
            output += f'{section}: {description}\n'
        return output
