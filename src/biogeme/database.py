"""Implementation of the class Database, wrapping a pandas dataframe
for specific services to Biogeme

:author: Michel Bierlaire

:date: Tue Mar 26 16:42:54 2019

"""

import logging
from typing import NamedTuple
import numpy as np
import pandas as pd

from biogeme.segmentation import DiscreteSegmentationTuple
import biogeme.exceptions as excep
import biogeme.filenames as bf
from biogeme import tools
from biogeme import draws

from biogeme.expressions import Variable, is_numeric, Numeric


class EstimationValidation(NamedTuple):
    estimation: pd.DataFrame
    validation: pd.DataFrame


logger = logging.getLogger(__name__)
"""Logger that controls the output of
        messages to the screen and log file.
        """


class Database:
    """Class that contains and prepare the database."""

    # @staticmethod
    def uniform_antithetic(sample_size, number_of_draws):
        return draws.getAntithetic(draws.getUniform, sample_size, number_of_draws)

    # @staticmethod
    def halton2(sample_size, number_of_draws):
        return draws.getHaltonDraws(sample_size, number_of_draws, base=2, skip=10)

    # @staticmethod
    def halton3(sample_size, number_of_draws):
        return draws.getHaltonDraws(sample_size, number_of_draws, base=3, skip=10)

    # @staticmethod
    def halton5(sample_size, number_of_draws):
        return draws.getHaltonDraws(sample_size, number_of_draws, base=5, skip=10)

    # @staticmethod
    def MLHS_anti(sample_size, number_of_draws):
        return draws.getAntithetic(
            draws.getLatinHypercubeDraws, sample_size, number_of_draws
        )

    # @staticmethod
    def symm_uniform(sample_size, number_of_draws):
        return draws.getUniform(sample_size, number_of_draws, symmetric=True)

    # @staticmethod
    def symm_uniform_antithetic(sample_size, number_of_draws):
        R = int(number_of_draws / 2)
        localDraws = Database.symm_uniform(sample_size, R)
        return np.concatenate((localDraws, -localDraws), axis=1)

    # @staticmethod
    def symm_halton2(sample_size, number_of_draws):
        return draws.getHaltonDraws(
            sample_size, number_of_draws, symmetric=True, base=2, skip=10
        )

    # @staticmethod
    def symm_halton3(sample_size, number_of_draws):
        return draws.getHaltonDraws(
            sample_size, number_of_draws, symmetric=True, base=3, skip=10
        )

    # @staticmethod
    def symm_halton5(sample_size, number_of_draws):
        return draws.getHaltonDraws(
            sample_size, number_of_draws, symmetric=True, base=5, skip=10
        )

    # @staticmethod
    def symm_MLHS(sample_size, number_of_draws):
        return draws.getLatinHypercubeDraws(
            sample_size, number_of_draws, symmetric=True
        )

    # @staticmethod
    def symm_MLHS_anti(sample_size, number_of_draws):
        R = int(number_of_draws / 2)
        localDraws = Database.symm_MLHS(sample_size, R)
        return np.concatenate((localDraws, -localDraws), axis=1)

    # @staticmethod
    def normal_antithetic(sample_size, number_of_draws):
        return draws.getNormalWichuraDraws(
            sample_size=sample_size,
            number_of_draws=number_of_draws,
            antithetic=True,
        )

    # @staticmethod
    def normal_halton2(sample_size, number_of_draws):
        unif = draws.getHaltonDraws(sample_size, number_of_draws, base=2, skip=10)
        return draws.getNormalWichuraDraws(
            sample_size,
            number_of_draws,
            uniformNumbers=unif,
            antithetic=False,
        )

    # @staticmethod
    def normal_halton3(sample_size, number_of_draws):
        unif = draws.getHaltonDraws(sample_size, number_of_draws, base=2, skip=10)
        return draws.getNormalWichuraDraws(
            sample_size,
            number_of_draws,
            uniformNumbers=unif,
            antithetic=False,
        )

    # @staticmethod
    def normal_halton5(sample_size, number_of_draws):
        unif = draws.getHaltonDraws(sample_size, number_of_draws, base=2, skip=10)
        return draws.getNormalWichuraDraws(
            sample_size,
            number_of_draws,
            uniformNumbers=unif,
            antithetic=False,
        )

    # @staticmethod
    def normal_MLHS(sample_size, number_of_draws):
        unif = draws.getLatinHypercubeDraws(sample_size, number_of_draws)
        return draws.getNormalWichuraDraws(
            sample_size,
            number_of_draws,
            uniformNumbers=unif,
            antithetic=False,
        )

    # @staticmethod
    def normal_MLHS_anti(sample_size, number_of_draws):
        unif = draws.getLatinHypercubeDraws(sample_size, int(number_of_draws / 2.0))
        return draws.getNormalWichuraDraws(
            sample_size, number_of_draws, uniformNumbers=unif, antithetic=True
        )

    # Dictionary containing native random number generators. Class attribute
    nativeRandomNumberGenerators = {
        'UNIFORM': (draws.getUniform, 'Uniform U[0, 1]'),
        'UNIFORM_ANTI': (uniform_antithetic, 'Antithetic uniform U[0, 1]'),
        'UNIFORM_HALTON2': (
            halton2,
            'Halton draws with base 2, skipping the first 10',
        ),
        'UNIFORM_HALTON3': (
            halton3,
            'Halton draws with base 3, skipping the first 10',
        ),
        'UNIFORM_HALTON5': (
            halton5,
            'Halton draws with base 5, skipping the first 10',
        ),
        'UNIFORM_MLHS': (
            draws.getLatinHypercubeDraws,
            'Modified Latin Hypercube Sampling on [0, 1]',
        ),
        'UNIFORM_MLHS_ANTI': (
            MLHS_anti,
            'Antithetic Modified Latin Hypercube Sampling on [0, 1]',
        ),
        'UNIFORMSYM': (symm_uniform, 'Uniform U[-1, 1]'),
        'UNIFORMSYM_ANTI': (
            symm_uniform_antithetic,
            'Antithetic uniform U[-1, 1]',
        ),
        'UNIFORMSYM_HALTON2': (
            symm_halton2,
            'Halton draws on [-1, 1] with base 2, skipping the first 10',
        ),
        'UNIFORMSYM_HALTON3': (
            symm_halton3,
            'Halton draws on [-1, 1] with base 3, skipping the first 10',
        ),
        'UNIFORMSYM_HALTON5': (
            symm_halton5,
            'Halton draws on [-1, 1] with base 5, skipping the first 10',
        ),
        'UNIFORMSYM_MLHS': (
            symm_MLHS,
            'Modified Latin Hypercube Sampling on [-1, 1]',
        ),
        'UNIFORMSYM_MLHS_ANTI': (
            symm_MLHS_anti,
            'Antithetic Modified Latin Hypercube Sampling on [-1, 1]',
        ),
        'NORMAL': (draws.getNormalWichuraDraws, 'Normal N(0, 1) draws'),
        'NORMAL_ANTI': (normal_antithetic, 'Antithetic normal draws'),
        'NORMAL_HALTON2': (
            normal_halton2,
            'Normal draws from Halton base 2 sequence',
        ),
        'NORMAL_HALTON3': (
            normal_halton3,
            'Normal draws from Halton base 3 sequence',
        ),
        'NORMAL_HALTON5': (
            normal_halton5,
            'Normal draws from Halton base 5 sequence',
        ),
        'NORMAL_MLHS': (
            normal_MLHS,
            'Normal draws from Modified Latin Hypercube Sampling',
        ),
        'NORMAL_MLHS_ANTI': (
            normal_MLHS_anti,
            ('Antithetic normal draws from Modified Latin Hypercube Sampling'),
        ),
    }

    # This statement does not work for versions of python before 3.10
    # @staticmethod
    def descriptionOfNativeDraws():
        """Describe the draws available draws with Biogeme

        :return: dict, where the keys are the names of the draws,
                 and the value their description

        Example of output::

            {'UNIFORM: Uniform U[0, 1]',
             'UNIFORM_ANTI: Antithetic uniform U[0, 1]'],
             'NORMAL: Normal N(0, 1) draws'}

        :rtype: dict

        """
        return {
            key: tuple[1]
            for key, tuple in Database.nativeRandomNumberGenerators.items()
        }

    def __init__(self, name, pandasDatabase):
        """Constructor

        :param name: name of the database.
        :type name: string

        :param pandasDatabase: data stored in a pandas data frame.
        :type pandasDatabase: pandas.DataFrame

        :raise BiogemeError: if the audit function detects errors.
        :raise BiogemeError: if the database is empty.
        """

        self.name = name
        """ Name of the database. Used mainly for the file name when
        dumping data.
        """

        if len(pandasDatabase.index) == 0:
            error_msg = 'Database has no entry'
            raise excep.BiogemeError(error_msg)

        self.data = pandasDatabase  #: Pandas data frame containing the data.

        self.fullData = pandasDatabase
        """Pandas data frame containing the full data. Useful when batches of
        the sample are used for approximating the log likelihood.
        """

        self.variables = None
        """names of the headers of the database so that they can be used as
        an object of type biogeme.expressions.Expression. Initialized
        by _generateHeaders()
        """

        self._generateHeaders()

        self.excludedData = 0
        """Number of observations removed by the function
        :meth:`biogeme.Database.remove`
        """

        self.panelColumn = None
        """Name of the column identifying the individuals in a panel
        data context. None if data is not panel.
        """

        self.individualMap = None
        """map identifying the range of observations for each individual in a
        panel data context. None if data is not panel.
        """

        self.fullIndividualMap = None
        """complete map identifying the range of observations for each
        individual in a panel data context. None if data is not
        panel. Useful when batches of the sample are used to
        approximate the log likelihood function.
        """

        self.userRandomNumberGenerators = {}
        """Dictionary containing user defined random number
        generators. Defined by the function
        Database.setRandomNumberGenerators that checks that reserved
        keywords are not used. The element of the dictionary is a
        tuple with two elements: (0) the function generating the
        draws, and (1) a string describing the type of draws
        """

        self.number_of_draws = 0
        """Number of draws generated by the function Database.generateDraws.
        Value 0 if this function is not called.
        """

        self.typesOfDraws = {}  #: Types of draws for Monte Carlo integration

        self.theDraws = None  #: Draws for Monte-Carlo integration

        self._avail = None  #: Availability expression to check

        self._choice = None  #: Choice expression to check

        self._expression = None  #: Expression to check

        listOfErrors, _ = self._audit()
        # For now, the audit issues only errors. If warnings are
        # triggered in the future, the nexrt lines should be
        # uncommented.
        # if listOfWarnings:
        #    logger.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            logger.warning('\n'.join(listOfErrors))
            raise excep.BiogemeError('\n'.join(listOfErrors))

    def _audit(self):
        """Performs a series of checks and reports warnings and errors.
          - Check if there are non numerical entries.
          - Check if there are NaN (not a number) entries.
          - Check if there are strings.
          - Check if the numbering of individuals are contiguous
            (panel data only).

        :return: A tuple of two lists with the results of the diagnostic:
            listOfErrors, listOfWarnings
        :rtype: tuple(list(str), list(str))
        """
        listOfErrors = []
        listOfWarnings = []
        for col, dtype in self.data.dtypes.items():
            if not np.issubdtype(dtype, np.number):
                theError = f'Column {col} in the database does contain {dtype}'
                listOfErrors.append(theError)

        if self.data.isnull().values.any():
            theError = (
                'The database contains NaN value(s). '
                'Detect where they are using the function isnan()'
            )
            listOfErrors.append(theError)

        return listOfErrors, listOfWarnings

    def _generateHeaders(self):
        """Record the names of the headers
        of the database so that they can be used as an object of type
        biogeme.expressions.Expression
        """
        self.variables = {col: Variable(col) for col in self.data.columns}

    def valuesFromDatabase(self, expression):
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
            raise excep.BiogemeError(error_msg)

        return expression.getValue_c(database=self, prepareIds=True)

    def checkAvailabilityOfChosenAlt(self, avail, choice):
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
            raise excep.BiogemeError(error_msg)

        choice_array = choice.getValue_c(
            database=self, aggregation=False, prepareIds=True
        )
        calculated_avail = {}
        for key, expression in avail.items():
            calculated_avail[key] = expression.getValue_c(
                database=self, aggregation=False, prepareIds=True
            )
        try:
            avail_chosen = np.array(
                [calculated_avail[c][i] for i, c in enumerate(choice_array)]
            )
        except KeyError as exc:
            for c in choice_array:
                err_msg = ''
                if c not in calculated_avail:
                    err_msg = (
                        f'Chosen alternative {c} does not appear in '
                        f'availability dict: {calculated_avail.keys()}'
                    )
                    raise excep.BiogemeError(err_msg) from exc
        return avail_chosen != 0

    def choiceAvailabilityStatistics(self, avail, choice):
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
            raise excep.BiogemeError(error_msg)

        self._avail = avail
        self._choice = choice

        choice_array = choice.getValue_c(
            database=self,
            aggregation=False,
            prepareIds=True,
        )
        unique = np.unique(choice_array, return_counts=True)
        choice_stat = {alt: unique[1][i] for i, alt in enumerate(unique[0])}
        calculated_avail = {}
        for key, expression in avail.items():
            calculated_avail[key] = expression.getValue_c(
                database=self,
                aggregation=False,
                prepareIds=True,
            )
        avail_stat = {k: sum(a) for k, a in calculated_avail.items()}
        theResults = {alt: (c, avail_stat[alt]) for alt, c in choice_stat.items()}
        return theResults

    def scaleColumn(self, column, scale):
        """Multiply an entire column by a scale value

        :param column: name of the column
        :type column: string
        :param scale: value of the scale. All values of the column will
              be multiplied by that scale.
        :type scale: float

        """
        self.data[column] = self.data[column] * scale

    def suggestScaling(self, columns=None, reportAll=False):
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

        :param reportAll: if False, remove entries where the suggested
            scale is 1, 0.1 or 10
        :type reportAll: bool

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
                    errorMsg = f'Variable {c} not found.'
                    raise excep.BiogemeError(errorMsg)

        largestValue = [
            max(np.abs(self.data[col].max()), np.abs(self.data[col].min()))
            for col in columns
        ]
        res = [
            [col, 1 / 10 ** np.round(np.log10(max(1.0, lv))), lv]
            for col, lv in zip(columns, largestValue)
        ]
        df = pd.DataFrame(res, columns=['Column', 'Scale', 'Largest'])
        if not reportAll:
            # Remove entries where the suggested scale is 1, 0.1 or 10
            remove = (df.Scale == 1) | (df.Scale == 0.1) | (df.Scale == 10)
            df.drop(df[remove].index, inplace=True)
        return df

    def sampleWithReplacement(self, size=None):
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

    def sampleIndividualMapWithReplacement(self, size=None):
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
        if not self.isPanel():
            errorMsg = (
                'Function sampleIndividualMapWithReplacement'
                ' is available only on panel data.'
            )
            raise excep.BiogemeError(errorMsg)

        if size is None:
            size = len(self.individualMap)
        sample = self.individualMap.iloc[
            np.random.randint(0, len(self.individualMap), size=size)
        ]
        return sample

    #####
    # This has to be reimplemented in a cleaner way
    ####
    #    def sampleWithoutReplacement(
    #        self, samplingRate, columnWithSamplingWeights=None
    #    ):
    #        """Replace the data set by a sample for stochastic algorithms
    #
    #        :param samplingRate: the proportion of data to include in the sample.
    #        :type samplingRate: float
    #        :param columnWithSamplingWeights: name of the column with
    #              the sampling weights. If None, each row has equal probability.
    #        :type columnWithSamplingWeights: string
    #
    #        :raise BiogemeError: if the structure of the database has been modified
    #            since last sample.
    #        """
    #        if self.isPanel():
    #            if self.fullIndividualMap is None:
    #                self.fullIndividualMap = self.individualMap
    #            # Check if the structure has not been modified since
    #            # last sample
    #            if set(self.fullIndividualMap.columns) != set(
    #                self.individualMap.columns
    #            ):
    #                message = (
    #                    'The structure of the database has been '
    #                    'modified since last sample. '
    #                )
    #                left = set(self.fullIndividualMap.columns).difference(
    #                    set(self.individualMap.columns)
    #                )
    #                if left:
    #                    message += f' Columns that disappeared: {left}'
    #                right = set(self.individualMap.columns).difference(
    #                    set(self.fullIndividualMap.columns)
    #                )
    #                if right:
    #                    message += f' Columns that were added: {right}'
    #                raise excep.BiogemeError(message)
    #
    #            self.individualMap = self.fullIndividualMap.sample(
    #                frac=samplingRate, weights=columnWithSamplingWeights
    #            )
    #        else:
    #            # Cross sectional data
    #            if self.fullData is None:
    #                self.fullData = self.data
    #            else:
    #                # Check if the structure has not been modified since
    #                # last sample
    #                if set(self.fullData.columns) != set(self.data.columns):
    #                    message = (
    #                        'The structure of the database has been modified '
    #                        'since last sample. '
    #                    )
    #                    left = set(self.fullData.columns).difference(
    #                        set(self.data.columns)
    #                    )
    #                    if left:
    #                        message += f' Columns that disappeared: {left}'
    #                    right = set(self.data.columns).difference(
    #                        set(self.fullData.columns)
    #                    )
    #                    if right:
    #                        message += f' Columns that were added: {right}'
    #                    raise excep.BiogemeError(message)
    #
    #            self.data = self.fullData.sample(
    #                frac=samplingRate, weights=columnWithSamplingWeights
    #            )

    #    def useFullSample(self):
    #        """Re-establish the full sample for calculation of the likelihood"""
    #        if self.isPanel():
    #            if self.fullIndividualMap is None:
    #                raise excep.BiogemeError(
    #                    'Full panel data set has not been saved.'
    #                )
    #            self.individualMap = self.fullIndividualMap
    #        else:
    #            if self.fullData is None:
    #                raise excep.BiogemeError('Full data set has not been saved.')
    #            self.data = self.fullData

    def addColumn(self, expression, column):
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
            raise excep.BiogemeError(error_msg)

        if column in self.data.columns:
            raise ValueError(
                f'Column {column} already exists in the database {self.name}'
            )

        self._expression = expression
        new_column = self._expression.getValue_c(
            database=self, aggregation=False, prepareIds=True
        )
        self.data[column] = new_column
        self.variables[column] = Variable(column)
        return self.data[column]

    def DefineVariable(self, name, expression):
        """Insert a new column in the database and define it as a variable."""
        self.addColumn(expression, name)
        return Variable(name)

    def remove(self, expression):
        """Removes from the database all entries such that the value
        of the expression is not 0.

        :param expression: expression to evaluate
        :type expression: biogeme.expressions.Expression

        """
        columnName = '__bioRemove__'
        if is_numeric(expression):
            self.addColumn(Numeric(expression), columnName)
        else:
            self.addColumn(expression, columnName)
        self.excludedData = len(self.data[self.data[columnName] != 0].index)
        self.data.drop(self.data[self.data[columnName] != 0].index, inplace=True)
        self.data.drop(columns=[columnName], inplace=True)

    def check_segmentation(self, segmentation_tuple):
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
                raise excep.BiogemeError(error_msg)
        for value, count in all_values.items():
            if value not in segmentation_tuple.mapping:
                error_msg = (
                    f'Variable {segmentation_tuple.variable.name} '
                    f'takes the value {value} [{count} times], and it does not '
                    f'define any segment.'
                )
                raise excep.BiogemeError(error_msg)

        named_values = {}
        for value, name in segmentation_tuple.mapping.items():
            named_values[name] = all_values[value]
        return named_values

    def dumpOnFile(self):
        """Dumps the database in a CSV formatted file.

        :return:  name of the file
        :rtype: string
        """
        theName = f'{self.name}_dumped'
        dataFileName = bf.get_new_file_name(theName, 'dat')
        self.data.to_csv(dataFileName, sep='\t', index_label='__rowId')
        logger.info(f'File {dataFileName} has been created')
        return dataFileName

    def setRandomNumberGenerators(self, rng):
        """Defines user-defined random numbers generators.

        :param rng: a dictionary of generators. The keys of the dictionary
           characterize the name of the generators, and must be
           different from the pre-defined generators in Biogeme
           (see :func:`~biogeme.database.Database.generateDraws` for the list).
           The elements of the
           dictionary are functions that take two arguments: the
           number of series to generate (typically, the size of the
           database), and the number of draws per series.
        :type rng: dict

        Example::

            def logNormalDraws(sample_size, number_of_draws):
                return np.exp(np.random.randn(sample_size, number_of_draws))

            def exponentialDraws(sample_size, number_of_draws):
                return -1.0 * np.log(np.random.rand(sample_size, number_of_draws))

            # We associate these functions with a name
            dict = {'LOGNORMAL':(logNormalDraws,
                                 'Draws from lognormal distribution'),
                    'EXP':(exponentialDraws,
                           'Draws from exponential distributions')}
            myData.setRandomNumberGenerators(dict)

        :raise ValueError: if a reserved keyword is used for a
             user-defined draws.

        """
        for k in self.nativeRandomNumberGenerators:
            if k in rng:
                errorMsg = (
                    f'{k} is a reserved keyword for draws'
                    f' and cannot be used for user-defined '
                    f'generators'
                )
                raise ValueError(errorMsg)

        self.userRandomNumberGenerators = rng

    def generateDraws(self, types, names, number_of_draws):
        """Generate draws for each variable.


        :param types: A dict indexed by the names of the variables,
                      describing the types of draws. Each of them can
                      be a native type or any type defined by the
                      function
                      :func:`~biogeme.database.Database.setRandomNumberGenerators`.

                      Native types:

                      - ``'UNIFORM'``: Uniform U[0, 1],
                      - ``'UNIFORM_ANTI``: Antithetic uniform U[0, 1]',
                      - ``'UNIFORM_HALTON2'``: Halton draws with base 2,
                        skipping the first 10,
                      - ``'UNIFORM_HALTON3'``: Halton draws with base 3,
                        skipping the first 10,
                      - ``'UNIFORM_HALTON5'``: Halton draws with base 5,
                        skipping  the first 10,
                      - ``'UNIFORM_MLHS'``: Modified Latin Hypercube
                        Sampling on [0, 1],
                      - ``'UNIFORM_MLHS_ANTI'``: Antithetic Modified
                        Latin Hypercube Sampling on [0, 1],
                      - ``'UNIFORMSYM'``: Uniform U[-1, 1],
                      - ``'UNIFORMSYM_ANTI'``: Antithetic uniform U[-1, 1],
                      - ``'UNIFORMSYM_HALTON2'``: Halton draws on [-1, 1]
                        with base 2, skipping the first 10,
                      - ``'UNIFORMSYM_HALTON3'``: Halton draws on [-1, 1]
                        with base 3, skipping the first 10,
                      - ``'UNIFORMSYM_HALTON5'``: Halton draws on [-1, 1]
                        with base 5, skipping the first 10,
                      - ``'UNIFORMSYM_MLHS'``: Modified Latin Hypercube
                        Sampling on [-1, 1],
                      - ``'UNIFORMSYM_MLHS_ANTI'``: Antithetic Modified
                        Latin Hypercube Sampling on [-1, 1],
                      - ``'NORMAL'``: Normal N(0, 1) draws,
                      - ``'NORMAL_ANTI'``: Antithetic normal draws,
                      - ``'NORMAL_HALTON2'``: Normal draws from Halton
                        base 2 sequence,
                      - ``'NORMAL_HALTON3'``: Normal draws from Halton
                        base 3 sequence,
                      - ``'NORMAL_HALTON5'``: Normal draws from Halton
                        base 5 sequence,
                      - ``'NORMAL_MLHS'``: Normal draws from Modified
                        Latin Hypercube Sampling,
                      - ``'NORMAL_MLHS_ANTI'``: Antithetic normal draws
                        from Modified Latin Hypercube Sampling]

                      For an updated description of the native types, call the function
                      :func:`~biogeme.database.Database.descriptionOfNativeDraws`.



        :type types: dict

        :param names: the list of names of the variables that require draws
            to be generated.
        :type names: list of strings

        :param number_of_draws: number of draws to generate.
        :type number_of_draws: int

        :return: a 3-dimensional table with draws. The 3 dimensions are

              1. number of individuals
              2. number of draws
              3. number of variables

        :rtype: numpy.array

        Example::

              types = {'randomDraws1': 'NORMAL_MLHS_ANTI',
                       'randomDraws2': 'UNIFORM_MLHS_ANTI',
                       'randomDraws3': 'UNIFORMSYM_MLHS_ANTI'}
              theDrawsTable = myData.generateDraws(types,
                  ['randomDraws1', 'randomDraws2', 'randomDraws3'], 10)


        :raise BiogemeError: if a type of draw is unknown.

        :raise BiogemeError: if the output of the draw generator does not
            have the requested dimensions.

        """
        self.number_of_draws = number_of_draws
        # Dimensions of the draw table:
        # 1. number of variables
        # 2. number of individuals
        # 3. number of draws
        listOfDraws = [None] * len(names)
        for i, v in enumerate(names):
            name = v
            drawType = types[name]
            self.typesOfDraws[name] = drawType
            theGenerator = self.nativeRandomNumberGenerators.get(drawType)
            if theGenerator is None:
                theGenerator = self.userRandomNumberGenerators.get(drawType)
                if theGenerator is None:
                    native = self.nativeRandomNumberGenerators
                    user = self.userRandomNumberGenerators
                    errorMsg = (
                        f'Unknown type of draws for '
                        f'variable {name}: {drawType}. '
                        f'Native types: {native}. '
                        f'User defined: {user}'
                    )
                    raise excep.BiogemeError(errorMsg)
            listOfDraws[i] = theGenerator[0](self.getSampleSize(), number_of_draws)
            if listOfDraws[i].shape != (self.getSampleSize(), number_of_draws):
                errorMsg = (
                    f'The draw generator for {name} must'
                    f' generate a numpy array of dimensions'
                    f' ({self.getSampleSize()}, {number_of_draws})'
                    f' instead of {listOfDraws[i].shape}'
                )
                raise excep.BiogemeError(errorMsg)

        self.theDraws = np.array(listOfDraws)
        # Draws as a three-dimensional numpy series. The dimensions
        # are organized to be more suited for calculation.
        # 1. number of individuals
        # 2. number of draws
        # 3. number of variables
        self.theDraws = np.moveaxis(self.theDraws, 0, -1)
        return self.theDraws

    def getNumberOfObservations(self):
        """
        Reports the number of observations in the database.

        Note that it returns the same value, irrespectively
        if the database contains panel data or not.

        :return: Number of observations.
        :rtype: int

        See also:  getSampleSize()
        """
        return self.data.shape[0]

    def getSampleSize(self):
        """Reports the size of the sample.

        If the data is cross-sectional, it is the number of
        observations in the database. If the data is panel, it is the
        number of individuals.

        :return: Sample size.
        :rtype: int

        See also: getNumberOfObservations()

        """
        if self.isPanel():
            return self.individualMap.shape[0]

        return self.data.shape[0]

    def split(self, slices, groups=None):
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
            raise excep.BiogemeError(error_msg)

        if groups is not None and self.isPanel():
            if groups != self.panelColumn:
                error_msg = (
                    f'The data is already organized by groups on '
                    f'{self.panelColumn}. The grouping by {groups} '
                    f'cannot be done.'
                )
                raise excep.BiogemeError(error_msg)

        if self.isPanel():
            groups = self.panelColumn

        if groups is None:
            shuffled = self.data.sample(frac=1)
            theSlices = np.array_split(shuffled, slices)
        else:
            ids = self.data[groups].unique()
            np.random.shuffle(ids)
            the_slices_ids = np.array_split(ids, slices)
            theSlices = [
                self.data[self.data[groups].isin(ids)] for ids in the_slices_ids
            ]
        estimationSets = []
        validationSets = []
        for i, v in enumerate(theSlices):
            estimationSets.append(pd.concat(theSlices[:i] + theSlices[i + 1 :]))
            validationSets.append(v)
        return [
            EstimationValidation(estimation=e, validation=v)
            for e, v in zip(estimationSets, validationSets)
        ]

    def isPanel(self):
        """Tells if the data is panel or not.

        :return: True if the data is panel.
        :rtype: bool
        """
        return self.panelColumn is not None

    def panel(self, columnName):
        """Defines the data as panel data

        :param columnName: name of the columns that identifies individuals.
        :type columnName: string

        :raise BiogemeError: if the data are not sorted properly, that
            is if the data for the one individuals are not consecutive.

        """

        self.panelColumn = columnName

        # Check if the data is organized in consecutive entries
        # Number of groups of data
        nGroups = tools.countNumberOfGroups(self.data, self.panelColumn)
        sortedData = self.data.sort_values(by=[self.panelColumn])
        nIndividuals = tools.countNumberOfGroups(sortedData, self.panelColumn)
        if nGroups != nIndividuals:
            theError = (
                f'The data must be sorted so that the data'
                f' for the same individual are consecutive.'
                f' There are {nIndividuals} individuals '
                f'in the sample, and {nGroups} groups of '
                f'data for column {self.panelColumn}.'
            )
            raise excep.BiogemeError(theError)

        self.buildPanelMap()

    def buildPanelMap(self):
        """Sorts the data so that the observations for each individuals are
        contiguous, and builds a map that identifies the range of indices of
        the observations of each individuals.
        """
        if self.panelColumn is not None:
            self.data = self.data.sort_values(by=self.panelColumn)
            # It is necessary to renumber the row to reflect the new ordering
            self.data.index = range(len(self.data.index))
            local_map = {}
            individuals = self.data[self.panelColumn].unique()
            for i in individuals:
                indices = self.data.loc[self.data[self.panelColumn] == i].index
                local_map[i] = [min(indices), max(indices)]
            self.individualMap = pd.DataFrame(local_map).T
            self.fullIndividualMap = self.individualMap

    def count(self, columnName, value):
        """Counts the number of observations that have a specific value in a
        given column.

        :param columnName: name of the column.
        :type columnName: string
        :param value: value that is seeked.
        :type value: float

        :return: Number of times that the value appears in the column.
        :rtype: int
        """
        return self.data[self.data[columnName] == value].count()[columnName]

    def generateFlatPanelDataframe(self, saveOnFile=None, identical_columns=tuple()):
        """Generate a flat version of the panel data

        :param saveOnFile: if True, the flat database is saved on file.
        :type saveOnFile: bool

        :param identical_columns: tuple of columns that contain the
            same values for all observations of the same
            individual. Default: empty list.

        :type identical_columns: tuple(str)

        :return: the flatten database, in Pandas format
        :rtype: pandas.DataFrame

        :raise BiogemeError: if the database in not panel

        """
        if not self.isPanel():
            error_msg = 'This function can only be called for panel data'
            raise excep.BiogemeError(error_msg)
        flat_data = tools.flatten_database(
            self.data, self.panelColumn, identical_columns=identical_columns
        )
        if saveOnFile:
            file_name = f'{self.name}_flatten.csv'
            flat_data.to_csv(file_name)
            logger.info(f'File {file_name} has been created.')
        return flat_data

    def __str__(self):
        """Allows to print the dabase"""
        result = f'biogeme database {self.name}:\n{self.data}'
        if self.isPanel():
            result += f'\nPanel data\n{self.individualMap}'
        return result

    def verify_segmentation(self, segmentation):
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
            raise excep.BiogemeError(error_msg)

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
            raise excep.BiogemeError(f'{error_msg_1} {error_msg_2}')

    def generate_segmentation(self, variable, mapping=None, reference=None):
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
            raise excep.BiogemeError(error_msg)

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
            raise excep.BiogemeError(error_msg)

        the_mapping = {value: f'{the_variable.name}_{value}' for value in unique_values}

        if mapping is not None:
            the_mapping.update(mapping)

        if reference is not None and reference not in mapping.values():
            error_msg = (
                f'Level {reference} of variable {the_variable.name} does not '
                'appear in the mapping: {mapping.values()}'
            )
            raise excep.BiogemeError(error_msg)

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
