"""
DataContainer: Responsible for holding and safely manipulating
the Biogeme dataset stored as a Pandas DataFrame.

Michel Bierlaire
Wed Mar 26 19:30:57 2025
"""

from __future__ import annotations

from collections.abc import Callable


from biogeme.floating_point import PANDAS_FLOAT
from .panel import flatten_dataframe
from .sampling import sample_with_replacement
import logging
import pandas as pd
from biogeme.exceptions import BiogemeError
import jax.numpy as jnp

from biogeme.floating_point import JAX_FLOAT
from biogeme.segmentation import (
    DiscreteSegmentationTuple,
    verify_segmentation,
    generate_segmentation,
)

from biogeme.expressions import Expression, Variable

logger = logging.getLogger(__name__)
"""Logger that controls the output of
        messages to the screen and log file.
        """


class Database:
    """Encapsulates a pandas DataFrame for Biogeme, providing safe access
    and basic operations such as checking for emptiness, scaling,
    and column manipulation.
    """

    def __init__(self, name: str, dataframe: pd.DataFrame):
        """
        Constructor

        :param name: name of the database
        :param dataframe: the data in pandas format
        :raises BiogemeError: if the dataframe is empty
        """
        self.name = name
        if dataframe.empty:
            raise BiogemeError('Database has no entry')
        try:
            self._df = dataframe.astype(PANDAS_FLOAT)
        except ValueError as e:
            raise BiogemeError(f'Data type conversion failed: {e}')
        self.number_of_excluded_data = 0

        self._listeners = []  # Called when the database is updated

        # Entries for panel data
        self.panel_column: str | None = None

    def register_listener(self, callback: Callable[[pd.Index], None]):
        self._listeners.append(callback)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a reference to the internal DataFrame."""
        return self._df

    def bootstrap_sample(self):
        """Returns a bootstrap sample of the data."""
        df = sample_with_replacement(self._df)
        return Database(f'{self.name}_bootstrap', df)

    @property
    def data_jax(self) -> jnp.ndarray:
        """Returns the data as a biogeme_jax object"""
        return jnp.asarray(self._df.to_numpy(), dtype=JAX_FLOAT)

    def is_empty(self) -> bool:
        """Returns True if the data container is empty"""
        return self._df.empty

    def num_rows(self) -> int:
        """Returns the number of rows in the dataset"""
        return self._df.shape[0]

    def num_columns(self) -> int:
        """Returns the number of columns in the dataset"""
        return self._df.shape[1]

    @property
    def sample_size(self) -> int:
        """Returns the size of the sample. Panel case will be implemented later."""
        return self.num_rows()

    def column_exists(self, column: str) -> bool:
        """Check if a column exists in the data"""
        return column in self._df.columns

    def scale_column(self, column: str, scale: float):
        """Scales all values in a given column

        :param column: name of the column to scale
        :param scale: scalar to multiply the column values by
        :raises BiogemeError: if the column is not found
        """
        if column not in self._df:
            raise BiogemeError(f'Column {column} not found in the database.')
        self._df[column] *= scale

    def add_column(self, column: str, values: pd.Series):
        """Adds a new column to the dataset

        :param column: name of the new column
        :param values: a pandas Series of same length as data
        :raises ValueError: if column already exists or lengths mismatch
        """
        if column in self._df.columns:
            raise ValueError(f'Column "{column}" already exists.')

        if len(values) != self.num_rows():
            raise ValueError(
                f'Length mismatch: column has {len(values)} values, expected {self.num_rows()}.'
            )

        self._df[column] = values

    def remove_rows(self, condition: pd.Series):
        """Removes all rows where the condition is True

        :param condition: Boolean Series of same length as the data
        """

        self._df = self._df[~condition].reset_index(drop=True)
        condition_index = condition[condition].index
        for callback in self._listeners:
            callback(condition_index)

    def remove(self, exclude_condition: Expression):
        """
        Removes rows from the database that satisfy a given condition.

        This method evaluates a Biogeme expression row by row on the database.
        All rows where the expression evaluates to a truthy value are removed.

        :param exclude_condition: A Biogeme expression that returns a boolean-like value
                                  for each row in the dataset. Rows where the result is
                                  True (nonzero) will be excluded.
        """
        from biogeme.calculator import evaluate_simple_expression_per_row

        condition = evaluate_simple_expression_per_row(
            expression=exclude_condition, database=self
        )
        series = pd.Series(condition != 0.0)
        self.number_of_excluded_data = series.sum()
        self.remove_rows(series)

    def define_variable(self, name: str, expression: Expression) -> Variable:
        """
        This method evaluates a Biogeme expression row by row on the database
        and creates a new column in the internal DataFrame with the results.

        :param name: Name of the new column to be added.
        :param expression: Biogeme expression to evaluate for each row.
        """
        from biogeme.calculator import evaluate_simple_expression_per_row

        new_values = evaluate_simple_expression_per_row(
            expression=expression,
            database=self,
        )
        self.dataframe[name] = pd.Series(new_values, dtype=PANDAS_FLOAT)
        return Variable(name)

    def remove_column(self, column: str):
        """Removes a column from the dataset"""
        if column in self._df.columns:
            self._df.drop(columns=[column], inplace=True)

    def get_column(self, column: str) -> pd.Series:
        """Returns the values of a column"""
        if column not in self._df.columns:
            raise BiogemeError(f'Column "{column}" not found.')
        return self._df[column]

    def generate_segmentation(
        self,
        variable: Variable | str,
        mapping: dict[int, str] | None = None,
        reference: str | None = None,
    ) -> DiscreteSegmentationTuple:
        """Generate a segmentation tuple for a variable.

        :param variable: Variable object or name of the variable
        :param mapping: mapping associating values of the variable to
            names. If incomplete, default names are provided.
        :param reference: name of the reference category. If None, an
            arbitrary category is selected as reference.

        """
        return generate_segmentation(
            dataframe=self.dataframe,
            variable=variable,
            mapping=mapping,
            reference=reference,
        )

    def verify_segmentation(self, segmentation: DiscreteSegmentationTuple) -> None:
        """Verifies if the definition of the segmentation is consistent with the data

        :param segmentation: definition of the segmentation
        :raise BiogemeError: if the segmentation is not consistent with the data.
        """
        verify_segmentation(dataframe=self.dataframe, segmentation=segmentation)

    def extract_slice(self, indices: pd.Index) -> Database:
        """
        Create a new Database instance containing only a subset of the data.

        This is useful to maintain consistency across estimation and validation datasets by slicing
        the original draws array according to the provided indices.

        :param indices: The indices used to extract the subset of draws.
        :return: A new Database instance containing the sliced draws.
        """
        sliced_database = Database(
            name=f'sliced {self.name}', dataframe=self.dataframe[indices]
        )
        return sliced_database

    def panel(self, column_name: str):
        """Defines the data as panel data

        :param column_name: name of the columns that identifies individuals.
        """
        self.panel_column = column_name

    def flatten_database(self, missing_data: float) -> tuple[pd.DataFrame, int]:
        if self.panel_column is None:
            raise BiogemeError('The panel column has not been defined.')
        return flatten_dataframe(
            dataframe=self.dataframe,
            grouping_column=self.panel_column,
            missing_data=missing_data,
        )
