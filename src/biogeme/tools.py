"""Implements some useful functions

:author: Michel Bierlaire

:date: Sun Apr 14 10:46:10 2019

"""

import sys
from datetime import timedelta
import logging
from itertools import product
from os import path
import shutil
from collections import defaultdict
import types
from typing import NamedTuple, Callable, Any, Optional, Type
import tempfile
import uuid
import numpy as np
import pandas as pd
from scipy.stats import chi2
import biogeme.exceptions as excep

logger = logging.getLogger(__name__)


class LRTuple(NamedTuple):
    """Tuple for the likelihood ratio test"""

    message: str
    statistic: float
    threshold: float


def findiff_g(
    the_function: Callable[[np.ndarray], tuple[float, ...]], x: np.ndarray
) -> np.ndarray:
    """Calculates the gradient of a function :math:`f` using finite differences

    :param the_function: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`. The other elements are not
                        used.

    :param x: argument of the function

    :return: numpy vector, same dimension as x, containing the gradient
       calculated by finite differences.
    """
    x = x.astype(float)
    tau = 0.0000001
    n = len(x)
    g = np.zeros(n)
    f = the_function(x)[0]
    for i in range(n):
        xi = x.item(i)
        xp = x.copy()
        if abs(xi) >= 1:
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        xp[i] = xi + s
        fp = the_function(xp)[0]
        g[i] = (fp - f) / s
    return g


def findiff_H(
    the_function: Callable[[np.ndarray], tuple[float, np.ndarray, Any]], x: np.ndarray
) -> np.ndarray:
    """Calculates the hessian of a function :math:`f` using finite differences

    :param the_function: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`, and the second is the
                        gradient of the function.  The other elements
                        are not used.

    :param x: argument of the function
    :return: numpy matrix containing the hessian calculated by
             finite differences.
    """
    tau = 1.0e-7
    n = len(x)
    H = np.zeros((n, n))
    g = the_function(x)[1]
    eye = np.eye(n, n)
    for i in range(n):
        xi = x.item(i)
        if abs(xi) >= 1:
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        ei = eye[i]
        gp = the_function(x + s * ei)[1]
        H[:, i] = (gp - g).flatten() / s
    return H


def checkDerivatives(
    the_function: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
    x: np.ndarray,
    names: Optional[list[str]] = None,
    logg: Optional[bool] = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Verifies the analytical derivatives of a function by comparing
    them with finite difference approximations.

    :param the_function: A function object that takes a vector as an argument,
        and returns a tuple:

        - The first element of the tuple is the value of the
          function :math:`f`,
        - the second is the gradient of the function,
        - the third is the hessian.

    :param x: arguments of the function

    :param names: the names of the entries of x (for reporting).

    :param logg: if True, messages will be displayed.

    :return: tuple f, g, h, gdiff, hdiff where

          - f is the value of the function at x,
          - g is the analytical gradient,
          - h is the analytical hessian,
          - gdiff is the difference between the analytical gradient
            and the finite difference approximation
          - hdiff is the difference between the analytical hessian
            and the finite difference approximation

    """
    x = np.array(x, dtype=float)
    f, g, h = the_function(x)
    g_num = findiff_g(the_function, x)
    gdiff = g - g_num
    if logg:
        if names is None:
            names = [f'x[{i}]' for i in range(len(x))]
        logger.info('x\t\tGradient\tFinDiff\t\tDifference')
        for k, v in enumerate(gdiff):
            logger.info(f'{names[k]:15}\t{g[k]:+E}\t{g_num[k]:+E}\t{v:+E}')

    h_num = findiff_H(the_function, x)
    hdiff = h - h_num
    if logg:
        logger.info('Row\t\tCol\t\tHessian\tFinDiff\t\tDifference')
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logger.info(
                    f'{names[row]:15}\t{names[col]:15}\t{h[row,col]:+E}\t'
                    f'{h_num[row,col]:+E}\t{hdiff[row,col]:+E}'
                )
    return f, g, h, gdiff, hdiff


def get_prime_numbers(n: int) -> list[int]:
    """Get a given number of prime numbers

    :param n: number of primes that are requested

    :return: array with prime numbers

    :raise BiogemeError: if the requested number is non positive or a float

    """
    total = 0
    upper_bound = 100
    if n <= 0:
        raise excep.BiogemeError(f'Incorrect number: {n}')

    while total < n:
        upper_bound *= 10
        primes = calculate_prime_numbers(upper_bound)
        total = len(primes)
    try:
        return primes[0:n]
    except TypeError as e:
        raise excep.BiogemeError(f'Incorrect number: {n}') from e


def calculate_prime_numbers(upper_bound: int) -> list[int]:
    """Calculate prime numbers

    :param upper_bound: prime numbers up to this value will be computed

    :return: array with prime numbers

    :raise BiogemeError: if the upper_bound is incorrectly defined
        (negative number, e.g.)

    >>> tools.calculate_prime_numbers(10)
    [2, 3, 5, 7]

    """
    if upper_bound < 0:
        raise excep.BiogemeError(f'Incorrect value: {upper_bound}')
    try:
        mywork = list(range(0, upper_bound + 1))
    except TypeError as e:
        raise excep.BiogemeError(f'Incorrect value: {upper_bound}') from e

    try:
        largest = int(np.ceil(np.sqrt(float(upper_bound))))
    except ValueError as e:
        raise excep.BiogemeError(f'Incorrect value: {upper_bound}') from e

    # Remove all multiples
    for i in range(2, largest + 1):
        if mywork[i] != 0:
            for k in range(2 * i, upper_bound + 1, i):
                mywork[k] = 0
    # Gather non zero entries, which are the prime numbers
    myprimes = []
    for i in range(1, upper_bound + 1):
        if mywork[i] != 0 and mywork[i] != 1:
            myprimes += [mywork[i]]

    return myprimes


def countNumberOfGroups(df: pd.DataFrame, column: str) -> int:
    """
    This function counts the number of groups of same value in a column.
    For instance: 1,2,2,3,3,3,4,1,1  would give 5.

    Example::

        >>>df = pd.DataFrame({'ID': [1, 1, 2, 3, 3, 1, 2, 3],
                              'value':[1000,
                                       2000,
                                       3000,
                                       4000,
                                       5000,
                                       5000,
                                       10000,
                                       20000]})
        >>>tools.countNumberOfGroups(df,'ID')
        6

        >>>tools.countNumberOfGroups(df,'value')
        7

    """
    df['_biogroups'] = (df[column] != df[column].shift(1)).cumsum()
    result = len(df['_biogroups'].unique())
    df.drop(columns=['_biogroups'], inplace=True)
    return result


def likelihood_ratio_test(
    model1: tuple[float, int],
    model2: tuple[float, int],
    significance_level: float = 0.05,
) -> LRTuple:
    """This function performs a likelihood ratio test between a
    restricted and an unrestricted model.

    :param model1: the final loglikelihood of one model, and the number of
                   estimated parameters.

    :param model2: the final loglikelihood of the other model, and
                   the number of estimated parameters.

    :param significance_level: level of significance of the test. Default: 0.05

    :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

    :raise BiogemeError: if the unrestricted model has a lower log
        likelihood than the restricted model.

    """

    loglike_m1, df_m1 = model1
    loglike_m2, df_m2 = model2
    if loglike_m1 > loglike_m2:
        if df_m1 < df_m2:
            raise excep.BiogemeError(
                f'The unrestricted model {model2} has a '
                f'lower log likelihood than the restricted one {model1}'
            )
        loglike_ur = loglike_m1
        loglike_r = loglike_m2
        df_ur = df_m1
        df_r = df_m2
    else:
        if df_m1 >= df_m2:
            raise excep.BiogemeError(
                f'The unrestricted model {model1} has a '
                f'lower log likelihood than the restricted one {model2}'
            )
        loglike_ur = loglike_m2
        loglike_r = loglike_m1
        df_ur = df_m2
        df_r = df_m1

    stat = -2 * (loglike_r - loglike_ur)
    chi_df = df_ur - df_r
    threshold = chi2.ppf(1 - significance_level, chi_df)
    if stat <= threshold:
        final_msg = f'H0 cannot be rejected at level {100*significance_level:.1f}%'
    else:
        final_msg = f'H0 can be rejected at level {100*significance_level:.1f}%'
    return LRTuple(message=final_msg, statistic=stat, threshold=threshold)


def flatten_database(
    df: pd.DataFrame,
    merge_id: str,
    row_name: str = None,
    identical_columns: list[str] = None,
) -> pd.DataFrame:
    """Combine several rows of a Pandas database into one.
    For instance, consider the following database::

           ID  Age  Cost   Name
        0   1   23    34  Item3
        1   1   23    45  Item4
        2   1   23    12  Item7
        3   2   45    65  Item3
        4   2   45    34  Item7



    If row_name is 'Name', the function generates the same data in the
    following format::

            Age  Item3_Cost  Item4_Cost  Item7_Cost
        ID
        1    23          34        45.0          12
        2    45          65         NaN          34


    If row_name is None, the function generates the same data in the
    following format::

            Age  1_Cost 1_Name  2_Cost 2_Name  3_Cost 3_Name
        ID
        1    23      34  Item3      45  Item4    12.0  Item7
        2    45      65  Item3      34  Item7     NaN    NaN

    :param df: initial data frame
    :type df: pandas.DataFrame

    :param merge_id: name of the column that identifies rows that
        should be merged. In the above example: 'ID'
    :type merge_id: str

    :param row_name: name of the columns that provides the name of the
        rows in the new dataframe. In the example above: 'Name'. If
        None, the rows are numbered sequentially.
    :type row_name: str

    :param identical_columns: name of the columns that contain
        identical values across the rows of a group. In the example
        above: ['Age']. If None, these columns are automatically
        detected. On large database, there may be a performance issue.
    :type identical_columns: list(str)

    :return: reformatted database
    :rtype: pandas.DataFrame

    """
    grouped = df.groupby(by=merge_id)
    all_columns = set(df.columns)

    def are_values_identical(col: pd.Series) -> bool:
        """This function checks if all the values in a column
        are identical

        :param col: the column

        :return: True if all values are identical. False otherwise.
        """

        return (col.iloc[0] == col).all(0)

    def get_varying_cols(g: pd.DataFrame) -> set[str]:
        """This functions returns the name of all columns
        that have constant values within each group of data.

        :param g: group of data

        :return: name of all columns that have constant values
            within each group of data.
        """
        return {colname for colname, col in g.items() if not are_values_identical(col)}

    if identical_columns is None:
        all_varying_cols = grouped.apply(get_varying_cols)
        varying_columns = set.union(*all_varying_cols)
        identical_columns = list(all_columns - varying_columns)
        varying_columns = list(varying_columns)
    else:
        identical_columns = set(identical_columns)
        identical_columns.add(merge_id)
        varying_columns = list(all_columns - identical_columns)

    # Take the first row for columns that are identical
    if identical_columns:
        common_data = df[list(identical_columns)].drop_duplicates(
            merge_id, keep='first'
        )
        common_data.index = common_data[merge_id]
    # Treat the other columns
    grouped_varying = df[[merge_id] + list(varying_columns)].groupby(by=merge_id)

    def treat(x: pd.DataFrame) -> pd.DataFrame:
        """Treat a group of data.

        :param x: group of data

        :return: the same data organized in one row, with proper column names

        :raise BiogemeError:  if there are duplicates in the name of
        the row. Indeed, in that case, they cannot be used to name the
        new columns.
        """
        if not are_values_identical(x[merge_id]):
            err_msg = f'Group has different IDs: {x[merge_id]}. ' f'Rows id: {x.index}'
            raise excep.BiogemeError(err_msg)
        if row_name is not None and not x[row_name].is_unique:
            err_msg = (
                f'Entries in column [{row_name}] are not unique. '
                f'This column cannot be used to name the new '
                f'columns:\n{x[[row_name, merge_id]]}. '
            )
            raise excep.BiogemeError(err_msg)

        the_columns = set(x.columns) - {merge_id}
        if row_name is not None:
            the_columns -= {row_name}
        sorted_list = sorted(list(the_columns))
        first = True
        i = 0
        for _, row in x.iterrows():
            i += 1
            if first:
                all_values = [row[merge_id]]
                all_columns = [merge_id]
                first = False
            name = f'{i}' if row_name is None else row[row_name]
            columns = [f'{name}_{c}' for c in sorted_list]
            all_values.extend([row[c] for c in sorted_list])
            all_columns.extend(columns)
        df = pd.DataFrame([all_values], columns=all_columns)
        return df

    flat_data = grouped_varying.apply(treat)
    flat_data.index = flat_data[merge_id]

    # We remove the column 'merge_id' as it is stored as index.
    if identical_columns:
        return pd.concat([common_data, flat_data], axis='columns').drop(
            columns=[merge_id]
        )
    return flat_data.drop(columns=[merge_id])


class TemporaryFile:
    """Class generating a temporary file, so that the user does not
    bother about its location, or even its name

    Example::

        with TemporaryFile() as filename:
            with open(filename, 'w') as f:
                print('stuff', file=f)
    """

    def __enter__(self, name: str = None) -> str:
        self.dir = tempfile.mkdtemp()
        name = str(uuid.uuid4()) if name is None else name
        return path.join(self.dir, name)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        """Destroys the temporary directory"""
        shutil.rmtree(self.dir)


class ModelNames:
    """Class generating model names from unique configuration string"""

    def __init__(self, prefix: str = 'Model'):
        self.prefix = prefix
        self.dict_of_names = {}
        self.current_number = 0

    def __call__(self, the_id):
        """Get a short model name from a unique ID

        :param the_id: Id of the model
        :type the_id: str (or anything that can be used as a key for a dict)
        """
        the_name = self.dict_of_names.get(the_id)
        if the_name is None:
            the_name = f'{self.prefix}_{self.current_number:06d}'
            self.current_number += 1
            self.dict_of_names[the_id] = the_name
        return the_name


def generate_unique_ids(list_of_ids):
    """If there are duplicates in the list, a new list is generated
    where there are renamed to obtain a list with unique IDs.

    :param list_of_ids: list of ids
    :type list_of_ids: list[str]

    :return: a dict that maps the unique names with the original name
    """
    counts = defaultdict(int)
    for the_id in list_of_ids:
        counts[the_id] += 1

    results = {}
    for name, count in counts.items():
        if count == 1:
            results[name] = name
        else:
            substitutes = [f'{name}_{i}' for i in range(count)]
            for new_name in substitutes:
                results[new_name] = name
    return results


def unique_product(*iterables, max_memory_mb=1024):
    """Generate the Cartesian product of multiple iterables, keeping
    only the unique entries.  Raises a MemoryError if memory usage
    exceeds the specified threshold.

    :param iterables: Variable number of iterables to compute the
        Cartesian product from.
    :type iterables: Iterable

    :param max_memory_mb: Maximum memory usage in megabytes (default: 1024MB).
    :type max_memory_mb: int

    :return: Yields unique entries from the Cartesian product.
    :rtype: Iterator[tuple]

    """

    MB_TO_BYTES = 1024 * 1024
    max_memory_bytes = max_memory_mb * MB_TO_BYTES  # Convert MB to bytes
    seen = set()  # Set to store seen entries
    total_memory = 0  # Track memory usage

    for items in product(*iterables):
        if items not in seen:
            seen.add(items)
            item_size = sum(sys.getsizeof(item) for item in items)
            total_memory += item_size
            if total_memory > max_memory_bytes:
                raise MemoryError(
                    f'Memory usage exceeded the specified threshold: '
                    f'{total_memory/MB_TO_BYTES:.1f} MB > '
                    f'{max_memory_bytes/MB_TO_BYTES} MB.'
                )
            yield items


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta in a "human-readable" way"""

    # Determine the total amount of seconds
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Get the total microseconds remaining
    microseconds = td.microseconds

    # Format based on the most significant unit
    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    if minutes > 0:
        return f'{minutes}m {second}s'
    if seconds > 0:
        return f'{seconds}.{microseconds // 100000:01}s'
    if microseconds >= 1000:
        return f'{microseconds // 1000}ms'  # Convert to milliseconds

    return f'{microseconds}μs'  # Microseconds as is
