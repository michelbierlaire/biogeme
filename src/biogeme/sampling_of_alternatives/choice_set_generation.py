""" Module in charge of functionalities related to the choice set generation

    If only one partition is provided (second_partition is None), a
    sample is first generated independently on the choice. It is then
    sorted in such a way that the alternatives sampled from the
    stratum where the choice is contained are listed at the end. And,
    if the chosen alternative happens to have been selected, it is
    placed at the end. The data is then organized as follows. The
    first colum corresponds to the chosen alternative and is labeled
    0. The following columns are the sampled alternatives, for a total
    of 1+J, numbered from 0 to J.

    For logit, all alternatives except the last one must be used: 0 to
    J-1.  For MEV models, the approximation of the sum capturing the
    nests requires a sample not based on the choice. This sample is
    available in columns 1 to J.

    Now, if second_partition is defined, a second sample is generated,
    dedicated to the approximation of the MEV terms.

:author: Michel Bierlaire
:date: Thu Sep  7 16:24:05 2023
"""
import os
import copy
import logging
import pandas as pd
from biogeme.expressions import Expression, TypeOfElementaryExpression
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from .sampling_context import SamplingContext, MEV_PREFIX
from .sampling_of_alternatives import SamplingOfAlternatives


logger = logging.getLogger(__name__)


class ChoiceSetsGeneration:
    """Class in charge of generationg the choice sets for each individual."""

    def __init__(self, context: SamplingContext):
        """Constructor

        :param context: contains all the information that is needed to
        perform the sampling of alternatives.
        :type context: SamplingContext
        """

        self.sampling_of_alternatives = SamplingOfAlternatives(context)

        self.alternatives = context.alternatives
        self.individuals = context.individuals
        self.choice_column = context.choice_column
        self.number_of_individuals = context.number_of_individuals
        self.id_column = context.id_column
        self.partition = context.partition
        self.second_partition = context.second_partition
        self.combined_variables = context.combined_variables
        self.biogeme_file_name = context.biogeme_file_name
        self.sample_size = context.sample_size
        self.biogeme_data = None

    def get_attributes_from_expression(self, expression: Expression) -> set[str]:
        """Extract the names of the attributes of alternatives from an expression"""
        variables = expression.set_of_elementary_expression(
            TypeOfElementaryExpression.VARIABLE
        )
        attributes = set(self.alternatives.columns)
        return variables & attributes

    def process_row(self, individual_row: pd.Series) -> dict:
        """Process one row of the individual database

        :param individual_row: rwo corresponding to one individual

        :return: a dictionnary containing the data for the extended row
        """
        choice = individual_row[self.choice_column]

        (
            first_sample,
            chosen_alternative,
        ) = self.sampling_of_alternatives.sample_alternatives(
            partition=self.partition, chosen=choice
        )

        # Rename columns for chosen_alternative
        chosen_alternative = chosen_alternative.add_suffix('_0')
        chosen_dict = chosen_alternative.iloc[0].to_dict()

        # Create the columnns
        flattened_first_series = first_sample.stack()
        flattened_first_dict = {
            (f'{col_name}_{row+1}'): value
            for (row, col_name), value in flattened_first_series.items()
        }

        row_data = individual_row.to_dict()
        row_data.update(chosen_dict)
        row_data.update(flattened_first_dict)

        if self.second_partition is not None:
            second_sample, _ = self.sampling_of_alternatives.sample_alternatives(
                partition=self.second_partition, chosen=None
            )

            # Rename columns for second_sample without multi-level index
            flattened_second_series = second_sample.stack()

            # We use "row+1" so that the alternatives are numnered
            # starting from 1, consistently with the first partition.
            flattened_second_dict = {
                (f'{MEV_PREFIX}{col_name}_{row+1}'): value
                for (row, col_name), value in flattened_second_series.items()
            }
            row_data.update(flattened_second_dict)

        return row_data

    def define_new_variables(self, database: Database):
        """Create the new variables

        :param database: database, in Biogeme format.
        """
        for new_variable in self.combined_variables:
            for index in range(self.sample_size + 1):
                copy_expression = copy.deepcopy(new_variable.formula)
                attributes = self.get_attributes_from_expression(copy_expression)
                copy_expression.rename_elementary(attributes, suffix=f'_{index}')
                database.DefineVariable(f'{new_variable.name}_{index}', copy_expression)
            if self.second_partition is not None:
                for index in range(1, self.sample_size + 1):
                    copy_expression = copy.deepcopy(new_variable.formula)
                    attributes = self.get_attributes_from_expression(copy_expression)
                    copy_expression.rename_elementary(
                        attributes, prefix=MEV_PREFIX, suffix=f'_{index}'
                    )
                    database.DefineVariable(
                        f'{MEV_PREFIX}{new_variable.name}_{index}', copy_expression
                    )

    def sample_and_merge(self, overwrite: bool = False) -> 'biogeme.database.Database':
        """Loops on the individuals and generate a choice set for each of them

        :param overwrite: if True, athe file is overwritten if it
            already exists. If False, an exception is raised if the file
            already exists.

        :return: database for Biogeme
        """
        if os.path.exists(self.biogeme_file_name) and not overwrite:
            biogeme_data = pd.read_csv(self.biogeme_file_name)
            biogeme_database = Database('merged_data', biogeme_data)
            return biogeme_database

        logger.info(
            f'Generating {self.sample_size} alternatives for '
            f'{self.number_of_individuals} observations'
        )
        biogeme_data = self.individuals.apply(
            self.process_row, axis=1, result_type='expand'
        )
        biogeme_database = Database('merged_data', biogeme_data)
        self.define_new_variables(biogeme_database)
        biogeme_data.to_csv(self.biogeme_file_name, index=False)
        logger.info(f'File {self.biogeme_file_name} has been created.')
        return biogeme_database
