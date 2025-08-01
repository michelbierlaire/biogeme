"""Module in charge of functionalities related to the choice set generation


    For thew main sample, all alternatives except the last one must be used: 0 to
    J-1.  For MEV models, the approximation of the sum capturing the
    nests requires another sample not based on the choice.

:author: Michel Bierlaire
:date: Fri Oct 27 12:50:06 2023
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from biogeme.database import Database
from biogeme.expressions import (
    Expression,
    OldNewName,
    list_of_variables_in_expression,
    rename_all_variables,
)
from .sampling_context import MEV_PREFIX, SamplingContext
from .sampling_of_alternatives import SamplingOfAlternatives

tqdm.pandas()

logger = logging.getLogger(__name__)


class ChoiceSetsGeneration:
    """Class in charge of generating the choice sets for each individual."""

    def __init__(self, context: SamplingContext):
        """Constructor

        :param context: contains all the information that is needed to
            perform the sampling of alternatives.

        """

        self.sampling_of_alternatives = SamplingOfAlternatives(context)

        self.alternatives = context.alternatives
        self.individuals = context.individuals
        self.choice_column = context.choice_column
        self.number_of_individuals = context.number_of_individuals
        self.id_column = context.id_column
        self.partition = context.sampling_protocol
        self.second_partition = context.mev_sampling_protocol
        self.combined_variables = context.combined_variables
        self.biogeme_file_name = context.biogeme_file_name
        self.total_sample_size = context.total_sample_size
        self.second_sample_size = context.total_mev_sample_size
        self.cnl_nests = context.cnl_nests
        self.biogeme_data = None

    def get_attributes_from_expression(self, expression: Expression) -> set[str]:
        """Extract the names of the attributes of alternatives from an expression"""
        variables = {
            var.name
            for var in list_of_variables_in_expression(the_expression=expression)
        }
        attributes = set(self.alternatives.columns)
        return variables & attributes

    def process_row(self, individual_row: pd.Series) -> dict:
        """Process one row of the individual database

        :param individual_row: row corresponding to one individual

        :return: a dictionary containing the data for the extended row
        """
        choice = individual_row[self.choice_column]

        first_sample = self.sampling_of_alternatives.sample_alternatives(chosen=choice)

        # Create the columns
        flattened_first_series: pd.Series[float, tuple[int, str]] = first_sample.stack()
        flattened_first_dict = {
            f'{col_name}_{row}': value
            for (row, col_name), value in flattened_first_series.items()
        }

        row_data = individual_row.to_dict()
        row_data.update(flattened_first_dict)

        if self.second_partition is not None:
            second_sample = self.sampling_of_alternatives.sample_mev_alternatives()

            # Rename columns for second_sample without multi-level index
            flattened_second_series = second_sample.stack()

            flattened_second_dict = {
                f"{MEV_PREFIX}{col_name}_{row}": value
                for (row, col_name), value in flattened_second_series.items()
            }
            row_data.update(flattened_second_dict)

        return row_data

    def define_new_variables(self, database: Database):
        """Create the new variables

        :param database: database, in Biogeme format.
        """
        total_iterations = len(self.combined_variables) * self.total_sample_size
        with tqdm(
            total=total_iterations, desc="Defining new variables..."
        ) as progress_bar:
            for new_variable in self.combined_variables:
                for index in range(self.total_sample_size):
                    copy_expression = new_variable.formula.deep_flat_copy()
                    attributes = self.get_attributes_from_expression(copy_expression)
                    renaming_list = [
                        OldNewName(old_name=attribute, new_name=f'{attribute}_{index}')
                        for attribute in attributes
                    ]
                    rename_all_variables(
                        expr=copy_expression, renaming_list=renaming_list
                    )
                    database.define_variable(
                        f"{new_variable.name}_{index}", copy_expression
                    )
                    progress_bar.update(1)
                if self.second_partition is not None:
                    for index in range(self.second_sample_size):
                        copy_expression = new_variable.formula.deep_flat_copy()
                        attributes = self.get_attributes_from_expression(
                            copy_expression
                        )
                        renaming_list = [
                            OldNewName(
                                old_name=attribute,
                                new_name=f'{MEV_PREFIX}{attribute}_{index}',
                            )
                            for attribute in attributes
                        ]
                        rename_all_variables(
                            expr=copy_expression,
                            renaming_list=renaming_list,
                        )
                        database.define_variable(
                            f"{MEV_PREFIX}{new_variable.name}_{index}", copy_expression
                        )

    def sample_and_merge(self, recycle: bool = False) -> Database:
        """Loops on the individuals and generate a choice set for each of them

        :param recycle: if True, if the data file already exists, it is not re-created.

        :return: database for Biogeme
        """
        if recycle:
            if os.path.exists(self.biogeme_file_name):
                biogeme_data = pd.read_csv(self.biogeme_file_name)
                biogeme_database = Database("merged_data", biogeme_data)
                return biogeme_database
            warning_msg = f"File {self.biogeme_file_name} does not exist."
            logger.warning(warning_msg)

        size = (
            f'{self.total_sample_size}'
            if self.second_sample_size is None
            else f'{self.total_sample_size} + {self.second_sample_size}'
        )
        logger.info(
            f"Generating {size} alternatives for "
            f"{self.number_of_individuals} observations"
        )
        biogeme_data = self.individuals.progress_apply(
            self.process_row, axis=1, result_type="expand"
        )
        biogeme_database = Database("merged_data", biogeme_data)
        logger.info("Define new variables")
        self.define_new_variables(biogeme_database)
        biogeme_data.to_csv(self.biogeme_file_name, index=False)
        logger.info(f"File {self.biogeme_file_name} has been created.")
        return biogeme_database
