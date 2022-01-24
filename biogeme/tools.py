"""Implements some useful functions

:author: Michel Bierlaire

:date: Sun Apr 14 10:46:10 2019

"""

# Too constraining
# pylint: disable=invalid-name, too-many-locals

import numpy as np
import pandas as pd
from scipy.stats import chi2
import biogeme.messaging as msg
import biogeme.exceptions as excep

logger = msg.bioMessage()


def findiff_g(theFunction, x):
    """Calculates the gradient of a function :math`f` using finite differences

    :param theFunction: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`. The other elements are not
                        used.
    :type theFunction: function

    :param x: argument of the function
    :type x: numpy.array

    :return: numpy vector, same dimension as x, containing the gradient
       calculated by finite differences.
    :rtype: numpy.array

    """
    x = x.astype(float)
    tau = 0.0000001
    n = len(x)
    g = np.zeros(n)
    f = theFunction(x)[0]
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
        fp = theFunction(xp)[0]
        g[i] = (fp - f) / s
    return g


def findiff_H(theFunction, x):
    """Calculates the hessian of a function :math:`f` using finite differences

    :param theFunction: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`, and the second is the
                        gradient of the function.  The other elements
                        are not used.

    :type theFunction: function

    :param x: argument of the function
    :type x: numpy.array

    :return: numpy matrix containing the hessian calculated by
             finite differences.
    :rtype: numpy.array

    """
    tau = 1.0e-7
    n = len(x)
    H = np.zeros((n, n))
    g = theFunction(x)[1]
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
        gp = theFunction(x + s * ei)[1]
        H[:, i] = (gp - g).flatten() / s
    return H


def checkDerivatives(theFunction, x, names=None, logg=False):
    """Verifies the analytical derivatives of a function by comparing
    them with finite difference approximations.

    :param theFunction: A function object that takes a vector as an argument,
        and returns a tuple:

        - The first element of the tuple is the value of the
          function :math:`f`,
        - the second is the gradient of the function,
        - the third is the hessian.

    :type theFunction: function

    :param x: arguments of the function
    :type x: numpy.array

    :param names: the names of the entries of x (for reporting).
    :type names: list(string)
    :param logg: if True, messages will be displayed.
    :type logg: bool


    :return: tuple f, g, h, gdiff, hdiff where

          - f is the value of the function at x,
          - g is the analytical gradient,
          - h is the analytical hessian,
          - gdiff is the difference between the analytical gradient
            and the finite difference approximation
          - hdiff is the difference between the analytical hessian
            and the finite difference approximation

    :rtype: float, numpy.array,numpy.array,  numpy.array,numpy.array

    """
    x = np.array(x, dtype=float)
    f, g, h = theFunction(x)
    g_num = findiff_g(theFunction, x)
    gdiff = g - g_num
    if logg:
        if names is None:
            names = [f'x[{i}]' for i in range(len(x))]
        logger.detailed('x\t\tGradient\tFinDiff\t\tDifference')
        for k, v in enumerate(gdiff):
            logger.detailed(f'{names[k]:15}\t{g[k]:+E}\t{g_num[k]:+E}\t{v:+E}')

    h_num = findiff_H(theFunction, x)
    hdiff = h - h_num
    if logg:
        logger.detailed('Row\t\tCol\t\tHessian\tFinDiff\t\tDifference')
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logger.detailed(
                    f'{names[row]:15}\t{names[col]:15}\t{h[row,col]:+E}\t'
                    f'{h_num[row,col]:+E}\t{hdiff[row,col]:+E}'
                )
    return f, g, h, gdiff, hdiff


def getPrimeNumbers(n):
    """Get a given number of prime numbers

    :param n: number of primes that are requested
    :type n: int

    :return: array with prime numbers
    :rtype: list(int)

    :raise biogemeError: if the requested number is non positive or a float

    """
    total = 0
    upperBound = 100
    if n <= 0:
        raise excep.biogemeError(f'Incorrect number: {n}')

    while total < n:
        upperBound *= 10
        primes = calculatePrimeNumbers(upperBound)
        total = len(primes)
    try:
        return primes[0:n]
    except TypeError as e:
        raise excep.biogemeError(f'Incorrect number: {n}') from e


def calculatePrimeNumbers(upperBound):
    """Calculate prime numbers

    :param upperBound: prime numbers up to this value will be computed
    :type upperBound: int

    :return: array with prime numbers
    :rtype: list(int)

    :raise biogemeError: if the upperBound is incorrectly defined
        (negative number, e.g.)

    >>> tools.calculatePrimeNumbers(10)
    [2, 3, 5, 7]

    """
    if upperBound < 0:
        raise excep.biogemeError(f'Incorrect value: {upperBound}')
    try:
        mywork = list(range(0, upperBound + 1))
    except TypeError as e:
        raise excep.biogemeError(f'Incorrect value: {upperBound}') from e

    try:
        largest = int(np.ceil(np.sqrt(float(upperBound))))
    except ValueError as e:
        raise excep.biogemeError(f'Incorrect value: {upperBound}') from e

    # Remove all multiples
    for i in range(2, largest + 1):
        if mywork[i] != 0:
            for k in range(2 * i, upperBound + 1, i):
                mywork[k] = 0
    # Gather non zero entries, which are the prime numbers
    myprimes = []
    for i in range(1, upperBound + 1):
        if mywork[i] != 0 and mywork[i] != 1:
            myprimes += [mywork[i]]

    return myprimes


def countNumberOfGroups(df, column):
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
    return len(df['_biogroups'].unique())


def likelihood_ratio_test(model1, model2, significance_level=0.95):
    """This function performs a likelihood ratio test between a
    restricted and an unrestricted model.

    :param model1: the final loglikelood of one model, and the number of
                   estimated parameters.
    :type model1: tuple(float, int)

    :param model2: the final loglikelood of the other model, and
                   the number of estimated parameters.
    :type model2: tuple(float, int)

    :param significance_level: level of significance of the test. Default: 0.95
    :type significance_level: float

    :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

    :rtype: (str, float, float)

    :raise biogemeError: if the unrestricted model has a lower log
        likelihood than the restricted model.

    """

    loglike_m1, df_m1 = model1
    loglike_m2, df_m2 = model2
    if loglike_m1 > loglike_m2:
        if df_m1 < df_m2:
            raise excep.biogemeError(
                f'The unrestricted model {model2} has a '
                f'lower log likelihood than the restricted one {model1}'
            )
        loglike_ur = loglike_m1
        loglike_r = loglike_m2
        df_ur = df_m1
        df_r = df_m2
    else:
        if df_m1 >= df_m2:
            raise excep.biogemeError(
                f'The unrestricted model {model1} has a '
                f'lower log likelihood than the restricted one {model2}'
            )
        loglike_ur = loglike_m2
        loglike_r = loglike_m1
        df_ur = df_m2
        df_r = df_m1

    stat = -2 * (loglike_r - loglike_ur)
    chi_df = df_ur - df_r
    threshold = chi2.ppf(significance_level, chi_df)
    if stat <= threshold:
        final_msg = f'H0 cannot be rejected at level {significance_level}'
    else:
        final_msg = f'H0 can be rejected at level {significance_level}'
    return final_msg, stat, threshold


def flatten_database(df, merge_id, row_name=None, identical_columns=None):
    """Combine several rows of a Pandas database into one. For instance,
    consider the following database:

       ID  Age  Cost   Name
    0   1   23    34  Item3
    1   1   23    45  Item4
    2   1   23    12  Item7
    3   2   45    65  Item3
    4   2   45    34  Item7

    If row_name is 'Name', the function generates the same data in the
    following format:

        Age  Item3_Cost  Item4_Cost  Item7_Cost
    ID
    1    23          34        45.0          12
    2    45          65         NaN          34

    If row_name is None, the function generates the same data in the
    following format:


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
    if identical_columns is None:

        def are_values_identical(col):
            """This function checks if all the values in a column
                are identical

            :param col: the column
            :type col: pandas.Series

            :return: True if all values are identical. False otherwise.
            :rtype: bool
            """
            
            return (col.iloc[0] == col).all(0)

        def get_varying_cols(g):
            """This functions returns the name of all columns
                that have constant values within each group of data.

            :param g: group of data
            :type g: pandas.DataFrame

            :return: name of all columns that have constant values
                within each group of data.
            :rtype: set(str)
            """
            return {
                colname
                for colname, col in g.iteritems()
                if not are_values_identical(col)
            }

        all_varying_cols = grouped.apply(get_varying_cols)
        varying_columns = set.union(*all_varying_cols)
        identical_columns = list(all_columns - varying_columns)
        varying_columns = list(varying_columns)
    else:
        varying_columns = list(all_columns - set(identical_columns))

    # Take the first row for columns that are identical
    common_data = df[identical_columns].drop_duplicates(merge_id, keep='first')
    common_data.index = common_data[merge_id]
    # Treat the other columns
    grouped_varying = df[[merge_id] + list(varying_columns)].groupby(
        by=merge_id
    )

    def treat(x):
        """Treat a group of data.

        :param x: group of data
        :type x: pandas.DataFrame

        :return: the same data organized in one row, with proper column names
        :rtype: pandas.DataFrame

        :raise: biogemeError if there are duplicates in the name of
        the row. Indeed, in that case, they cannot be used to name the
        new columns.
        """
        if not are_values_identical(x[merge_id]):
            err_msg = (
                f'Group has different IDs: {x[merge_id]}. '
                f'Rows id: {x.index}'
            )
            raise excep.biogemeError(err_msg)
        if row_name is not None and not x[row_name].is_unique:
            err_msg = (
                f'Entries in column [{row_name}] are not unique. '
                f'This column cannot be used to name the new columns:\n{x[[row_name, merge_id]]}. '
            )
            raise excep.biogemeError(err_msg)

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
    return pd.concat(
        [common_data, flat_data], axis='columns'
    ).drop(columns=[merge_id])
