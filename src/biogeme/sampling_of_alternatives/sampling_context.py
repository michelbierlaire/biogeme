""" Defines a class that characterized the context to apply sampling of alternatives

:author: Michel Bierlaire
:date: Wed Sep  6 14:38:31 2023
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional
import inspect
import pandas as pd
from biogeme.expressions import Expression, TypeOfElementaryExpression
from biogeme.exceptions import BiogemeError

MEV_PREFIX = 'MEV_'
LOG_PROBA_COL = '_log_proba'
MEV_WEIGHT = '_mev_weight'


class StratumTuple(NamedTuple):
    """A stratum is an element of a partition of the full choice set,
    combined with the number of alternatives that must be sampled.
    """

    subset: set[int]
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

    :param partition: Partition used for the sampling. Each
       StratumTuple contains a set of IDs characterizing the subset,
       and the sample size, that is the number of alternatives to
       randomly draw from the subset.

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

    :param second_partition: If a second choice set need to be sampled
        for the MEV terms, the corresponding partitition is provided
        here.

    """

    partition: list[StratumTuple]
    individuals: pd.DataFrame
    choice_column: str
    alternatives: pd.DataFrame
    id_column: str
    biogeme_file_name: str
    utility_function: Expression
    combined_variables: list[CrossVariableTuple]
    second_partition: Optional[list[StratumTuple]] = None

    def check_expression(self, expression: Expression) -> None:
        """Verifies if the variables contained in the expression can be found in the databases"""
        variables = expression.set_of_elementary_expression(
            TypeOfElementaryExpression.VARIABLE
        )
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

    def check_partition(self, partition: tuple[StratumTuple]) -> None:
        """Check if the partition is truly a partition. If not, an exception is raised

        :param partition: partition to check

        :raise BiogemeError: if some elements are present in more than one subset.

        :raise BiogemeError: if the size of the union of the subsets does
           not match the expected total size

        :raise BiogemeError: if an alternative in the partition does
            not appear in the database of alternatives

        :raise BiogemeError: if a stratum is empty

        :raise BiogemeError: if the number of sampled alternatives in
            a stratum is incorrect , that is zero, or larger than the
            stratum size..

        :raise BiogemeError: if the partition is not a tuple

        :raise BiogemeError: if the partition is an empty tuple.

        :raise BiogemeError: if an object in the partition is not a StratumTuple
        """
        if not isinstance(partition, (list, tuple)):
            if isinstance(partition, StratumTuple):
                partition = (partition,)
            else:
                raise BiogemeError(
                    f'partition: Expected argument to be a tuple, and not '
                    f'{type(partition)}. Remember that, if the tuple'
                )

        if len(partition) == 0:
            raise BiogemeError('partition: Expected non-empty tuple')

        if not all(isinstance(item, StratumTuple) for item in partition):
            raise BiogemeError(
                'partition: All elements of the tuple must be of type StratumTuple'
            )

        nbr_unique_elements = len(set.union(*[s.subset for s in partition]))
        total_nbr = sum(list(len(s.subset) for s in partition))
        if nbr_unique_elements != total_nbr:
            error_msg = (
                f'This is not a partition. There are {nbr_unique_elements} '
                f'unique elements, and the total size of the partition '
                f'is {total_nbr}. Some elements are therefore present '
                f'in more than one subset.'
            )
            raise BiogemeError(error_msg)

        if nbr_unique_elements != self.number_of_alternatives:
            error_msg = (
                f'The partitions contain {nbr_unique_elements} alternatives '
                f'while there are {self.number_of_alternatives} in the database'
            )
            raise BiogemeError(error_msg)

        # Verify that all requested alternatives appear in the database of alternatives
        for stratum in partition:
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

    def check_valid_alternatives(self, set_of_ids: set[int]):
        """Check if the IDs in set are indeed valid
            alternatives. Typically used to check if a nest is well
            defined

        :param set_of_ids: set of identifiers to check

        :raise BiogemeError: if at least one id is invalid.
        """
        print('Check: ', set_of_ids)
        print('Valid ids: ', self.alternatives[self.id_column])
        if (
            not pd.Series(list(set_of_ids))
            .isin(self.alternatives[self.id_column])
            .all()
        ):
            missing_values = set_of_ids - set(self.alternatives[self.id_column])
            raise BiogemeError(
                f'The following IDs are not valid alternative IDs: {missing_values}'
            )

    def __post_init__(self):
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

        self.check_partition(self.partition)
        if self.second_partition is not None:
            self.check_partition(self.second_partition)

        self.sample_size = sum(stratum.sample_size for stratum in self.partition)
        self.second_sample_size = (
            None
            if self.second_partition is None
            else sum(stratum.sample_size for stratum in self.second_partition)
        )
        self.check_expression(self.utility_function)
        for cross_variable in self.combined_variables:
            self.check_expression(cross_variable.formula)

        self.attributes = set(self.alternatives.columns) | {
            combined_variable.name for combined_variable in self.combined_variables
        }

        self.mev_prefix = '' if self.second_partition is None else MEV_PREFIX
