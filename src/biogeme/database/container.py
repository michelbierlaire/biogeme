"""
DataContainer: Responsible for holding and safely manipulating
the Biogeme dataset stored as a Pandas DataFrame.

Michel Bierlaire
Wed Mar 26 19:30:57 2025
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from biogeme.deprecated import deprecated
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    ExpressionOrNumeric,
    Variable,
    validate_and_convert,
)
from biogeme.floating_point import JAX_FLOAT, PANDAS_FLOAT
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.segmentation import (
    DiscreteSegmentationTuple,
    generate_segmentation,
    verify_segmentation,
)
from jax import numpy as jnp

from .sampling import sample_with_replacement

logger = logging.getLogger(__name__)
"""Logger that controls the output of
        messages to the screen and log file.
        """


class Database:
    """Encapsulates a pandas DataFrame for Biogeme, providing safe access
    and basic operations such as checking for emptiness, scaling,
    and column manipulation.
    """

    def __init__(self, name: str, dataframe: pd.DataFrame, use_jit: bool = True):
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
        self.use_jit = use_jit
        self.number_of_excluded_data = 0

        self._listeners = []  # Called when the database is updated

        self.panel_column: str | None = None

    @classmethod
    def dummy_database(
        cls,
    ) -> Database:
        df = pd.DataFrame({'x': [0]})  # single-row dummy input
        return Database('dummy', df)

    def __str__(self) -> str:
        return f'biogeme database {self.name}'

    def register_listener(self, callback: Callable[[pd.Index], None]):
        self._listeners.append(callback)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a reference to the internal DataFrame."""
        return self._df

    def get_copy(self, name_of_copy: str | None = None) -> Database:
        """Returns a copy of the database"""
        the_name = f'{self.name}_copy' if name_of_copy is None else name_of_copy
        return Database(the_name, self.dataframe.copy())

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

    def add_column(self, column: str, values: pd.Series) -> None:
        """Adds a new column to the dataset

        :param column: name of the new column
        :param values: a pandas Series of same length as data
        :raises ValueError: if column already exists or lengths mismatch
        """
        if column in self._df.columns:
            raise ValueError(f'Column "{column}" already exists.')

        if len(values) != self.num_rows():
            raise ValueError(
                f'Length mismatch: column has {len(values)} values, '
                f'expected {self.num_rows()}.'
            )

        self._df[column] = values

    def remove_rows(self, condition: pd.Series):
        """Removes all rows where the condition is True

        :param condition: Boolean Series of same length as the data
        """
        # Build a boolean mask aligned to the current DataFrame index without
        # triggering pandas' future warning about silent downcasting on fillna.
        cond = pd.Series(condition)
        # Align to index first
        cond = cond.reindex(self._df.index)
        # Ensure we are not on object dtype before filling NAs:
        # Prefer pandas' nullable boolean, then downcast to plain bool.
        try:
            cond = cond.astype('boolean')  # BoolDtype with NA support
        except (TypeError, ValueError):
            # Fallbacks if values are heterogeneous: try to infer objects
            # and coerce typical truthy patterns; final fallback: nonzero test.
            cond = cond.infer_objects(copy=False)
            if cond.dtype == object:
                # Map common textual/numeric truthy/falsey to booleans, leave others as NA
                _TRUE = {True, 1, 1.0, 'True', 'true', 'TRUE'}
                _FALSE = {False, 0, 0.0, 'False', 'false', 'FALSE', ''}
                cond = cond.map(
                    lambda v: True if v in _TRUE else (False if v in _FALSE else pd.NA)
                )
                cond = cond.astype('boolean')
            else:
                cond = cond != 0
                cond = cond.astype('boolean')
        # Now safely fill NA and convert to plain bool
        cond = cond.fillna(False).astype(bool)
        if len(cond) != len(self._df):
            raise ValueError(
                f'Condition length {len(cond)} != dataframe length {len(self._df)}'
            )
        self._df = self._df.loc[~cond].reset_index(drop=True)
        condition_index = cond[cond].index
        for callback in self._listeners:
            callback(condition_index)

    def reset_indices(self) -> None:
        self._df = self._df.reset_index(drop=True)

    def remove(self, exclude_condition: ExpressionOrNumeric):
        """
        Removes rows from the database that satisfy a given condition.

        This method evaluates a Biogeme expression row by row on the database.
        All rows where the expression evaluates to a truthy value are removed.

        :param exclude_condition: A Biogeme expression that returns a boolean-like value
                                  for each row in the dataset. Rows where the result is
                                  True (nonzero) will be excluded.
        """
        from biogeme.jax_calculator import evaluate_simple_expression_per_row

        exclude_condition: Expression = validate_and_convert(exclude_condition)
        condition = evaluate_simple_expression_per_row(
            expression=exclude_condition,
            database=self,
            numerically_safe=True,
            second_derivatives_mode=SecondDerivativesMode.NEVER,
            use_jit=self.use_jit,
        )
        series = pd.Series(condition != 0.0)
        self.number_of_excluded_data = int(series.sum())
        self.remove_rows(series)

    def define_variable(self, name: str, expression: Expression) -> Variable:
        """
        This method evaluates a Biogeme expression row by row on the database
        and creates a new column in the internal DataFrame with the results.

        :param name: Name of the new column to be added.
        :param expression: Biogeme expression to evaluate for each row.
        """
        if name in self.dataframe.columns:
            error_msg = f'Variable {name} already exists'
            raise ValueError(error_msg)
        if self.dataframe.empty:
            error_msg = 'Empty database.'
            raise BiogemeError(error_msg)

        from biogeme.jax_calculator import evaluate_simple_expression_per_row

        new_values = evaluate_simple_expression_per_row(
            expression=expression,
            database=self,
            numerically_safe=True,
            second_derivatives_mode=SecondDerivativesMode.NEVER,
            use_jit=self.use_jit,
        )
        if np.isnan(new_values).any():
            num_total = len(new_values)
            num_nan = np.isnan(new_values).sum()
            nan_indices = np.where(np.isnan(new_values))[0].tolist()

            message = f"The evaluated values for '{name}' contain NaN entries.\n"
            message += f'Total values: {num_total}, NaN values: {num_nan}.\n'

            if num_nan == num_total:
                message += 'All values are NaN.'
            else:
                message += f'Indices with NaN: {nan_indices}'

            raise BiogemeError(message)
        self.dataframe[name] = pd.Series(
            new_values, index=self.dataframe.index, dtype=PANDAS_FLOAT
        )
        return Variable(name)

    @deprecated(new_func=define_variable)
    def DefineVariable(self, name: str, expression: Expression) -> Variable:
        """
        This method evaluates a Biogeme expression row by row on the database
        and creates a new column in the internal DataFrame with the results.

        :param name: Name of the new column to be added.
        :param expression: Biogeme expression to evaluate for each row.
        """
        pass

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

    def panel(self, column_name: str):
        self.panel_column = column_name

    def verify_segmentation(self, segmentation: DiscreteSegmentationTuple) -> None:
        """Verifies if the definition of the segmentation is consistent with the data

        :param segmentation: definition of the segmentation
        :raise BiogemeError: if the segmentation is not consistent with the data.
        """
        verify_segmentation(dataframe=self.dataframe, segmentation=segmentation)

    def extract_slice(self, indices: pd.Index) -> Database:
        """
        Create a new Database instance containing only a subset of the data.

        This is useful to maintain consistency across estimation and validation datasets
        by slicing the original draws array according to the provided indices.

        :param indices: The indices used to extract the subset of draws.
        :return: A new Database instance containing the sliced draws.
        """
        sliced_database = Database(
            name=f'sliced {self.name}', dataframe=self.dataframe[indices]
        )
        return sliced_database

    def suggest_scaling(
        self, columns: list[str] | None = None, report_all: bool = False
    ) -> pd.DataFrame:
        """Suggest a scaling of the variables in the database.

        For each column, :math:`\\delta` is the difference between the
        largest and the smallest value, or one if the difference is
        smaller than one. The level of magnitude is evaluated as a
        power of 10. The suggested scale is the inverse of this value.

        .. math:: s = \\frac{1}{10^{|\\log_{10} \\delta|}}

        where :math:`|x|` is the integer closest to :math:`x`.

        :param columns: list of columns to be considered.
                        If None, all of them will be considered.

        :param report_all: if False, remove entries where the suggested
            scale is 1, 0.1 or 10

        :return: A Pandas dataframe where each row contains the name
                 of the variable and the suggested scale s. Ideally,
                 the column should be multiplied by s.

        :raise BiogemeError: if a variable in ``columns`` is unknown.
        """
        if columns is None:
            columns = self.dataframe.columns
        else:
            for c in columns:
                if c not in self.dataframe:
                    error_msg = f'Variable {c} not found.'
                    raise BiogemeError(error_msg)

        largest_value = [
            max(np.abs(self.dataframe[col].max()), np.abs(self.dataframe[col].min()))
            for col in columns
        ]
        res = [
            [col, 1 / 10 ** np.round(np.log10(max(1.0, lv))), lv]
            for col, lv in zip(columns, largest_value)
        ]
        df = pd.DataFrame(res, columns=['Column', 'Scale', 'Largest'])
        if not report_all:
            # Remove entries where the suggested scale is 1, 0.1 or 10
            remove = (df.Scale == 1) | (df.Scale == 0.1) | (df.Scale == 10)
            df.drop(df[remove].index, inplace=True)
        return df

    def is_panel(self) -> bool:
        return self.panel_column is not None

    def extract_rows(self, rows: list[int]) -> Database:
        """Extracts selected rows fronm the database.

        :param rows: list of rows to extract
        :return: the new database with the selected rows.
        """
        selected_rows = self.dataframe.iloc[rows]
        new_name = f'{self.name}_{rows}'
        return Database(name=new_name, dataframe=selected_rows)
