"""
PanelStructure: Handles organization and indexing of panel data,
where observations are grouped by individuals.

Michel Bierlaire
Wed Mar 26 19:33:13 2025
"""

import logging

import pandas as pd

from .container import Database

RELEVANT_PREFIX = 'relevant_'

logger = logging.getLogger(__name__)


def observation_suffix(index: int) -> str:
    """Return a zero-padded suffix for observation index (1-based)."""
    return f"__panel__{index + 1:02d}"


class PanelDatabase:
    """Encapsulates a pandas DataFrame for Biogeme, in panel format,"""

    def __init__(self, database: Database, panel_column: str):
        """
        Constructor

        :param database: original database
        :param panel_column: name of the column with the identifier of the individuals.
        :raises BiogemeError: if the dataframe is empty
        """
        self.original_database = database
        # Entries for panel data
        self.panel_column: str = panel_column

    def __str__(self) -> str:
        return f'biogeme panel database {self.database.name}'

    def flatten_database(self, missing_data: float) -> tuple[pd.DataFrame, int]:
        """
        Flatten a long-format dataframe into a wide-format dataframe where each row represents one individual
        and columns represent multiple observations.

        :param missing_data: value to use if data is missing.
        :return: a tuple containing two things:
            - A wide-format DataFrame where each row corresponds to one individual.
            For each variable column in the original DataFrame (excluding `grouping_column`), the output contains
            multiple columns named `columnname_XX`, where `XX` is the zero-padded observation index (starting at 01).
            Additionally, for each observation index, a `relevant_XX` column indicates whether the observation
            is relevant (1) or padded with a missing value (0).

            - The size of the largest group.

        """
        dataframe = self.original_database.dataframe
        logger.info(f'Flattening database [{dataframe.shape}].')

        grouping_column = self.panel_column
        if dataframe.empty:
            return dataframe.copy(), 0

        # Find non-ID columns
        value_columns = [col for col in dataframe.columns if col != grouping_column]

        # Identify columns with constant values within each group
        constant_columns = []
        for col in value_columns:
            if dataframe.groupby(grouping_column)[col].nunique().max() == 1:
                constant_columns.append(col)

        # Determine max number of observations per individual
        largest_group = dataframe.groupby(grouping_column).size().max()

        # Prepare list to collect rows
        flattened_rows = []

        for individual_id, group in dataframe.groupby(grouping_column):
            group = group.reset_index(drop=True)
            obs_count = len(group)
            row = {grouping_column: individual_id}
            for col in constant_columns:
                row[col] = group.at[0, col]

            for obs_index in range(largest_group):
                suffix = observation_suffix(obs_index)
                row[f"{RELEVANT_PREFIX}{suffix}"] = 1 if obs_index < obs_count else 0
                is_valid = obs_index < obs_count
                for col in value_columns:
                    value = group.at[obs_index, col] if is_valid else missing_data
                    row[f"{col}{suffix}"] = value
            flattened_rows.append(row)

        flat_dataframe = pd.DataFrame(flattened_rows)
        logger.info(f'Database flattened [{flat_dataframe.shape}]')

        return flat_dataframe, largest_group
