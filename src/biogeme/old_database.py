"""Implementation of the class Database, wrapping a pandas dataframe
for specific services to Biogeme

:author: Michel Bierlaire

:date: Tue Mar 26 16:42:54 2019

"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

import biogeme.filenames as bf
import biogeme.tools.database
from biogeme.deprecated import deprecated
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Variable,
    Expression,
    validate_and_convert,
)
from biogeme.segmentation import DiscreteSegmentationTuple

if TYPE_CHECKING:
    from biogeme.expressions import ExpressionOrNumeric


class EstimationValidation(NamedTuple):
    estimation: pd.DataFrame
    validation: pd.DataFrame


logger = logging.getLogger(__name__)
"""Logger that controls the output of
        messages to the screen and log file.
        """


class Database:
    """Class that contains and prepare the database."""

    def __init__(self, pandas_database: pd.DataFrame):
        """Constructor

        :param pandas_database: data stored in a pandas data frame.
        :type pandas_database: pandas.DataFrame

        :raise BiogemeError: if the audit function detects errors.
        :raise BiogemeError: if the database is empty.
        """

        if len(pandas_database.index) == 0:
            error_msg = 'Database has no entry'
            raise BiogemeError(error_msg)

        self.data = pandas_database  #: Pandas data frame containing the data.

        self.variables = None
        """names of the headers of the database so that they can be used as
        an object of type biogeme.expressions.Expression. Initialized
        by _generateHeaders()
        """

        self._generate_headers()

        self.excluded_data = 0
        """Number of observations removed by the function
        :meth:`biogeme.Database.remove`
        """

        self.panel_column = None
        """Name of the column identifying the individuals in a panel
        data context. None if data is not panel.
        """

        self.individual_map = None
        """map identifying the range of observations for each individual in a
        panel data context. None if data is not panel.
        """

        self.fullIndividualMap = None
        """complete map identifying the range of observations for each
        individual in a panel data context. None if data is not
        panel. Useful when batches of the sample are used to
        approximate the log likelihood function.
        """

        self._avail = None  #: Availability expression to check

        self._choice = None  #: Choice expression to check

        self._expression = None  #: Expression to check

        list_of_errors, _ = self._audit()
        # For now, the audit issues only errors. If warnings are
        # triggered in the future, the nexrt lines should be
        # uncommented.
        # if listOfWarnings:
        #    logger.warning('\n'.join(listOfWarnings))
        if list_of_errors:
            logger.warning('\n'.join(list_of_errors))
            raise BiogemeError('\n'.join(list_of_errors))

    def _generate_headers(self) -> None:
        """Record the names of the headers
        of the database so that they can be used as an object of type
        biogeme.expressions.Expression
        """
        self.variables = {col: Variable(col) for col in self.data.columns}

    def values_from_database(self, expression: Expression) -> pd.Series:
        """Evaluates an expression for each entry of the database.

        :param expression: expression to evaluate
        :type expression:  biogeme.expressions.Expression.

        :return: numpy series, long as the number of entries
                 in the database, containing the calculated quantities.
        :rtype: numpy.Series

        :raise BiogemeError: if the database is empty.
        """

        if len(self.data.index) == 0:
            error_msg = 'Database has no entry'
            raise BiogemeError(error_msg)

        return expression.get_value_c(database=self, prepare_ids=True)

    def check_availability_of_chosen_alt(
        self, avail: dict[int, Expression], choice: Expression
    ) -> pd.Series:
        """Check if the chosen alternative is available for each entry
        in the database.

        :param avail: list of expressions to evaluate the
                      availability conditions for each alternative.
        :type avail: list of biogeme.expressions.Expression
        :param choice: expression for the chosen alternative.
        :type choice: biogeme.expressions.Expression

        :return: numpy series of bool, long as the number of entries
                 in the database, containing True is the chosen alternative is
                 available, False otherwise.
        :rtype: numpy.Series

        :raise BiogemeError: if the chosen alternative does not appear
            in the availability dict
        :raise BiogemeError: if the database is empty.
        """
        self._avail = avail
        self._choice = choice

        if len(self.data.index) == 0:
            error_msg = 'Database has no entry'
            raise BiogemeError(error_msg)

        choice_array = choice.get_value_c(
            database=self, aggregation=False, prepare_ids=True
        )
        calculated_avail = {}
        for key, expression in avail.items():
            calculated_avail[key] = expression.get_value_c(
                database=self, aggregation=False, prepare_ids=True
            )
        try:
            avail_chosen = np.array(
                [calculated_avail[c][i] for i, c in enumerate(choice_array)]
            )
            return avail_chosen != 0
        except KeyError as exc:
            for c in choice_array:
                if c not in calculated_avail:
                    err_msg = (
                        f'Chosen alternative {c} does not appear in '
                        f'availability dict: {calculated_avail.keys()}'
                    )
                    raise BiogemeError(err_msg) from exc

    def choice_availability_statistics(
        self, avail: dict[int, Expression], choice: Expression
    ) -> dict[int, tuple[int, int]]:
        """Calculates the number of time an alternative is chosen and available

        :param avail: list of expressions to evaluate the
                      availability conditions for each alternative.
        :type avail: list of biogeme.expressions.Expression
        :param choice: expression for the chosen alternative.
        :type choice: biogeme.expressions.Expression

        :return: for each alternative, a tuple containing the number of time
            it is chosen, and the number of time it is available.
        :rtype: dict(int: (int, int))

        :raise BiogemeError: if the database is empty.
        """
        if len(self.data.index) == 0:
            error_msg = 'Database has no entry'
            raise BiogemeError(error_msg)

        self._avail = avail
        self._choice = choice

        choice_array = choice.get_value_c(
            database=self,
            aggregation=False,
            prepare_ids=True,
        )
        unique = np.unique(choice_array, return_counts=True)
        choice_stat = {alt: int(unique[1][i]) for i, alt in enumerate(list(unique[0]))}
        calculated_avail = {}
        for key, expression in avail.items():
            calculated_avail[key] = expression.get_value_c(
                database=self,
                aggregation=False,
                prepare_ids=True,
            )
        avail_stat = {k: sum(a) for k, a in calculated_avail.items()}
        the_results = {alt: (c, avail_stat[alt]) for alt, c in choice_stat.items()}
        return the_results

    def scale_column(self, column: str, scale: float):
        """Multiply an entire column by a scale value

        :param column: name of the column
        :type column: string
        :param scale: value of the scale. All values of the column will
              be multiplied by that scale.
        :type scale: float

        """
        self.data[column] *= scale

    def suggest_scaling(
        self, columns: list[str] | None = None, report_all: bool = False
    ):
        """Suggest a scaling of the variables in the database.

        For each column, :math:`\\delta` is the difference between the
        largest and the smallest value, or one if the difference is
        smaller than one. The level of magnitude is evaluated as a
        power of 10. The suggested scale is the inverse of this value.

        .. math:: s = \\frac{1}{10^{|\\log_{10} \\delta|}}

        where :math:`|x|` is the integer closest to :math:`x`.

        :param columns: list of columns to be considered.
                        If None, all of them will be considered.
        :type columns: list(str)

        :param report_all: if False, remove entries where the suggested
            scale is 1, 0.1 or 10
        :type report_all: bool

        :return: A Pandas dataframe where each row contains the name
                 of the variable and the suggested scale s. Ideally,
                 the column should be multiplied by s.

        :rtype: pandas.DataFrame

        :raise BiogemeError: if a variable in ``columns`` is unknown.
        """
        if columns is None:
            columns = self.data.columns
        else:
            for c in columns:
                if c not in self.data:
                    error_msg = f'Variable {c} not found.'
                    raise BiogemeError(error_msg)

        largest_value = [
            max(np.abs(self.data[col].max()), np.abs(self.data[col].min()))
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

    def sample_with_replacement(self, size: int | None = None) -> pd.DataFrame:
        """Extract a random sample from the database, with replacement.

        Useful for bootstrapping.

        :param size: size of the sample. If None, a sample of
               the same size as the database will be generated.
               Default: None.
        :type size: int

        :return: pandas dataframe with the sample.
        :rtype: pandas.DataFrame

        """
        if size is None:
            size = len(self.data)
        sample = self.data.iloc[np.random.randint(0, len(self.data), size=size)]
        return sample

    def sample_individual_map_with_replacement(
        self, size: int | None = None
    ) -> pd.DataFrame:
        """Extract a random sample of the individual map
        from a panel data database, with replacement.

        Useful for bootstrapping.

        :param size: size of the sample. If None, a sample of
                   the same size as the database will be generated.
                   Default: None.
        :type size: int

        :return: pandas dataframe with the sample.
        :rtype: pandas.DataFrame

        :raise BiogemeError: if the database in not in panel mode.
        """
        if not self.is_panel():
            error_msg = (
                'Function sampleIndividualMapWithReplacement'
                ' is available only on panel data.'
            )
            raise BiogemeError(error_msg)

        if size is None:
            size = len(self.individual_map)
        sample = self.individual_map.iloc[
            np.random.randint(0, len(self.individual_map), size=size)
        ]
        return sample

    def add_column(self, expression: Expression, column: str) -> pd.Series:
        """Add a new column in the database, calculated from an expression.

        :param expression:  expression to evaluate
        :type expression: biogeme.expressions.Expression

        :param column: name of the column to add
        :type column: string

        :return: the added column
        :rtype: numpy.Series

        :raises ValueError: if the column name already exists.
        :raise BiogemeError: if the database is empty.

        """
        if len(self.data.index) == 0:
            error_msg = 'Database has no entry'
            raise BiogemeError(error_msg)

        if column in self.data.columns:
            raise ValueError(
                f'Column {column} already exists in the database {self.name}'
            )

        self._expression = expression
        new_column = self._expression.get_value_c(
            database=self, aggregation=False, prepare_ids=True
        )
        self.data[column] = new_column
        self.variables[column] = Variable(column)
        return self.data[column]

    def define_variable(self, name: str, expression: Expression) -> Variable:
        """Insert a new column in the database and define it as a variable."""
        self.add_column(expression, name)
        return Variable(name)

    def remove(self, expression: ExpressionOrNumeric):
        """Removes from the database all entries such that the value
        of the expression is not 0.

        :param expression: expression to evaluate
        :type expression: biogeme.expressions.Expression

        """
        column_name = '__bioRemove__'
        expression = validate_and_convert(expression)
        self.add_column(expression, column_name)
        self.excluded_data = len(self.data[self.data[column_name] != 0].index)
        self.data.drop(self.data[self.data[column_name] != 0].index, inplace=True)
        self.data.drop(columns=[column_name], inplace=True)

    def check_segmentation(
        self, segmentation_tuple: DiscreteSegmentationTuple
    ) -> dict[str, int]:
        """Check that the segmentation covers the complete database

        :param segmentation_tuple: object describing the segmentation
        :type segmentation_tuple: biogeme.segmentation.DiscreteSegmentationTuple

        :return: number of observations per segment.
        :rtype: dict(str: int)
        """

        all_values = self.data[segmentation_tuple.variable.name].value_counts()
        # Check if all values in the segmentation are in the database
        for value, name in segmentation_tuple.mapping.items():
            if value not in all_values:
                error_msg = (
                    f'Variable {segmentation_tuple.variable.name} does not '
                    f'take the value {value} representing segment "{name}"'
                )
                raise BiogemeError(error_msg)
        for value, count in all_values.items():
            if value not in segmentation_tuple.mapping:
                error_msg = (
                    f'Variable {segmentation_tuple.variable.name} '
                    f'takes the value {value} [{count} times], and it does not '
                    f'define any segment.'
                )
                raise BiogemeError(error_msg)

        named_values = {}
        for value, name in segmentation_tuple.mapping.items():
            named_values[name] = all_values[value]
        return named_values

    def dump_on_file(self) -> str:
        """Dumps the database in a CSV formatted file.

        :return:  name of the file
        :rtype: string
        """
        the_name = f'{self.name}_dumped'
        data_file_name = bf.get_new_file_name(the_name, 'dat')
        self.data.to_csv(data_file_name, sep='\t', index_label='__rowId')
        logger.info(f'File {data_file_name} has been created')
        return data_file_name

    def get_number_of_observations(self) -> int:
        """
        Reports the number of observations in the database.

        Note that it returns the same value, irrespectively
        if the database contains panel data or not.

        :return: Number of observations.
        :rtype: int

        See also:  getSampleSize()
        """
        return self.data.shape[0]

    def get_sample_size(self) -> int:
        """Reports the size of the sample.

        If the data is cross-sectional, it is the number of
        observations in the database. If the data is panel, it is the
        number of individuals.

        :return: Sample size.
        :rtype: int

        See also: getNumberOfObservations()

        """
        if self.is_panel():
            return self.individual_map.shape[0]

        return self.data.shape[0]

    def split(
        self, slices: int, groups: str | None = None
    ) -> list[EstimationValidation]:
        """Prepare estimation and validation sets for validation.

        :param slices: number of slices
        :type slices: int

        :param groups: name of the column that defines the ID of the
            groups. Data belonging to the same groups will be maintained
            together.
        :type groups: str

        :return: list of estimation and validation data sets
        :rtype: list(tuple(pandas.DataFrame, pandas.DataFrame))

        :raise BiogemeError: if the number of slices is less than two

        """
        if slices < 2:
            error_msg = (
                f'The number of slices is {slices}. It must be greater '
                f'or equal to 2.'
            )
            raise BiogemeError(error_msg)

        if groups is not None and self.is_panel():
            if groups != self.panel_column:
                error_msg = (
                    f'The data is already organized by groups on '
                    f'{self.panel_column}. The grouping by {groups} '
                    f'cannot be done.'
                )
                raise BiogemeError(error_msg)

        if self.is_panel():
            groups = self.panel_column

        if groups is None:
            shuffled = self.data.sample(frac=1)
            the_slices = np.array_split(shuffled, slices)
        else:
            ids = self.data[groups].unique()
            np.random.shuffle(ids)
            the_slices_ids = np.array_split(ids, slices)
            the_slices = [
                self.data[self.data[groups].isin(ids)] for ids in the_slices_ids
            ]
        estimation_sets = []
        validation_sets = []
        for i, v in enumerate(the_slices):
            estimation_sets.append(pd.concat(the_slices[:i] + the_slices[i + 1 :]))
            validation_sets.append(v)
        return [
            EstimationValidation(estimation=e, validation=v)
            for e, v in zip(estimation_sets, validation_sets)
        ]

    def is_panel(self) -> bool:
        """Tells if the data is panel or not.

        :return: True if the data is panel.
        :rtype: bool
        """
        return self.panel_column is not None

    def panel(self, column_name: str):
        """Defines the data as panel data

        :param column_name: name of the columns that identifies individuals.
        :type column_name: string

        :raise BiogemeError: if the data are not sorted properly, that
            is if the data for the one individuals are not consecutive.

        """

        self.panel_column = column_name

        # Check if the data is organized in consecutive entries
        # Number of groups of data
        n_groups = biogeme.tools.count_number_of_groups(self.data, self.panel_column)
        sorted_data = self.data.sort_values(by=[self.panel_column])
        n_individuals = biogeme.tools.count_number_of_groups(
            sorted_data, self.panel_column
        )
        if n_groups != n_individuals:
            the_error = (
                f'The data must be sorted so that the data'
                f' for the same individual are consecutive.'
                f' There are {n_individuals} individuals '
                f'in the sample, and {n_groups} groups of '
                f'data for column {self.panel_column}.'
            )
            raise BiogemeError(the_error)

        self.build_panel_map()

    def build_panel_map(self) -> None:
        """Sorts the data so that the observations for each individuals are
        contiguous, and builds a map that identifies the range of indices of
        the observations of each individuals.
        """
        if self.panel_column is not None:
            self.data = self.data.sort_values(by=self.panel_column)
            # It is necessary to renumber the row to reflect the new ordering
            self.data.index = range(len(self.data.index))
            local_map = {}
            individuals = self.data[self.panel_column].unique()
            for i in individuals:
                indices = self.data.loc[self.data[self.panel_column] == i].index
                local_map[i] = [min(indices), max(indices)]
            self.individual_map = pd.DataFrame(local_map).T
            self.fullIndividualMap = self.individual_map

    @deprecated(build_panel_map)
    def buildPanelMap(self) -> None:
        pass

    def count(self, column_name: str, value: float) -> int:
        """Counts the number of observations that have a specific value in a
        given column.

        :param column_name: name of the column.
        :type column_name: string
        :param value: value that is searched.
        :type value: float

        :return: Number of times that the value appears in the column.
        :rtype: int
        """
        return self.data[self.data[column_name] == value].count()[column_name]

    def generate_flat_panel_dataframe(
        self, save_on_file: bool = False, identical_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Generate a flat version of the panel data

        :param save_on_file: if True, the flat database is saved on file.
        :type save_on_file: bool

        :param identical_columns: tuple of columns that contain the
            same values for all observations of the same
            individual. Default: empty list.

        :type identical_columns: tuple(str)

        :return: the flatten database, in Pandas format
        :rtype: pandas.DataFrame

        :raise BiogemeError: if the database in not panel

        """
        if not self.is_panel():
            error_msg = 'This function can only be called for panel data'
            raise BiogemeError(error_msg)
        flat_data = biogeme.tools.database.flatten_database(
            self.data, self.panel_column, identical_columns=identical_columns
        )
        if save_on_file:
            file_name = f'{self.name}_flatten.csv'
            flat_data.to_csv(file_name)
            logger.info(f'File {file_name} has been created.')
        return flat_data

    @deprecated(generate_flat_panel_dataframe)
    def generateFlatPanelDataframe(
        self, save_on_file: bool = False, identical_columns: list[str] | None = None
    ) -> pd.DataFrame:
        pass

    def __str__(self) -> str:
        """Allows to print the database"""
        result = f'biogeme database {self.name}:\n{self.data}'
        if self.is_panel():
            result += f'\nPanel data\n{self.individual_map}'
        return result

    def verify_segmentation(self, segmentation: DiscreteSegmentationTuple) -> None:
        """Verifies if the definition of the segmentation is consistent with the data

        :param segmentation: definition of the segmentation
        :type segmentation: DiscreteSegmentationTuple

        :raise BiogemeError: if the segmentation is not consistent with the data.
        """

        variable = (
            segmentation.variable
            if isinstance(segmentation.variable, Variable)
            else Variable(segmentation.variable)
        )

        # Check if the variable is in the database.
        if variable.name not in self.data.columns:
            error_msg = f'Unknown variable {variable.name}'
            raise BiogemeError(error_msg)

        # Extract all unique values from the data base.
        unique_values = set(self.data[variable.name].unique())
        segmentation_values = set(segmentation.mapping.keys())

        in_data_not_in_segmentation = unique_values - segmentation_values
        in_segmentation_not_in_data = segmentation_values - unique_values

        error_msg_1 = (
            (
                f'The following entries are missing in the segmentation: '
                f'{in_data_not_in_segmentation}.'
            )
            if in_data_not_in_segmentation
            else ''
        )

        error_msg_2 = (
            (
                f'Segmentation entries do not exist in the data: '
                f'{in_segmentation_not_in_data}.'
            )
            if in_segmentation_not_in_data
            else ''
        )

        if error_msg_1 or error_msg_2:
            raise BiogemeError(f'{error_msg_1} {error_msg_2}')

    def extract_rows(self, a_range: Iterable[int]) -> Database:
        """
        Create a database object using only some rows

        :param a_range: specify the desired range of rows.
        :return: the reduced dataabse
        """

        # Validate the provided range
        max_index = len(self.data) - 1
        if any(i < 0 or i > max_index for i in a_range):
            raise IndexError(
                'One or more indices in a_range are out of the valid range.'
            )
        reduced_data_frame = self.data.iloc[list(a_range)]

        return Database(name=f'{self.name}_reduced', pandas_database=reduced_data_frame)

    def generate_segmentation(
        self,
        variable: Variable | str,
        mapping: dict[int, str] | None = None,
        reference: str | None = None,
    ) -> DiscreteSegmentationTuple:
        """Generate a segmentation tuple for a variable.

        :param variable: Variable object or name of the variable
        :type variable: biogeme.expressions.Variable or string

        :param mapping: mapping associating values of the variable to
            names. If incomplete, default names are provided.
        :type mapping: dict(int: str)

        :param reference: name of the reference category. If None, an
            arbitrary category is selected as reference.  :type:
        :type reference: str


        """

        the_variable = (
            variable if isinstance(variable, Variable) else Variable(variable)
        )

        # Check if the variable is in the database.
        if the_variable.name not in self.data.columns:
            error_msg = f'Unknown the_variable {the_variable.name}'
            raise BiogemeError(error_msg)

        # Extract all unique values from the data base.
        unique_values = set(self.data[the_variable.name].unique())

        if len(unique_values) >= 10:
            warning_msg = (
                f'Variable {the_variable.name} takes a total of '
                f'{len(unique_values)} different values in the database. It is '
                f'likely to be too large for a discrete segmentation.'
            )
            logger.warning(warning_msg)

        # Check that the provided mapping is consistent with the data
        values_not_in_data = [
            value for value in mapping.keys() if value not in unique_values
        ]

        if values_not_in_data:
            error_msg = (
                f'The following values in the mapping do not exist in the data for '
                f'variable {the_variable.name}: {values_not_in_data}'
            )
            raise BiogemeError(error_msg)

        the_mapping = {value: f'{the_variable.name}_{value}' for value in unique_values}

        if mapping is not None:
            the_mapping.update(mapping)

        if reference is not None and reference not in mapping.values():
            error_msg = (
                f'Level {reference} of variable {the_variable.name} does not '
                'appear in the mapping: {mapping.values()}'
            )
            raise BiogemeError(error_msg)

        return DiscreteSegmentationTuple(
            variable=the_variable,
            mapping=the_mapping,
            reference=reference,
        )

    def mdcev_count(self, list_of_columns: list[str], new_column: str) -> None:
        """For the MDCEV models, we calculate the number of
            alternatives that are chosen, that is the number of
            columns with a non zero entry.

        :param list_of_columns: list of columns containing the quantity of each good.
        :param new_column: name of the new column where the result is stored
        """
        self.data[new_column] = self.data[list_of_columns].apply(
            lambda x: (x != 0).sum(), axis=1
        )

    def mdcev_row_split(self, a_range: Iterable[int] | None = None) -> list[Database]:
        """
        For the MDCEV model, we generate a list of Database objects, each of them associated with a different row of
        the database,

        :param a_range: specify the desired range of rows.
        :return: list of rows, each in a Database format
        """
        if a_range is None:
            the_range = range(len(self.data))
        else:
            # Validate the provided range
            max_index = len(self.data) - 1
            if any(i < 0 or i > max_index for i in a_range):
                raise IndexError(
                    'One or more indices in a_range are out of the valid range.'
                )
            the_range = a_range

        rows_of_database = [
            Database(name=f'row_{i}', pandas_database=self.data.iloc[[i]])
            for i in the_range
        ]
        return rows_of_database

    @deprecated(new_func=description_of_native_draws)
    def descriptionOfNativeDraws():
        pass
