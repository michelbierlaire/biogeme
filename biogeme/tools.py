"""Implements some useful functions

:author: Michel Bierlaire

:date: Sun Apr 14 10:46:10 2019

"""

# Too constraining
# pylint: disable=invalid-name, too-many-locals

import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.integrate import dblquad
import biogeme.messaging as msg
import biogeme.exceptions as excep
from biogeme.expressions import Expression

logger = msg.bioMessage()

LRTuple = namedtuple('LRTuple', 'message statistic threshold')


def findiff_g(theFunction, x):
    """Calculates the gradient of a function :math:`f` using finite differences

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
    result = len(df['_biogroups'].unique())
    df.drop(columns=['_biogroups'], inplace=True)
    return result


def likelihood_ratio_test(model1, model2, significance_level=0.05):
    """This function performs a likelihood ratio test between a
    restricted and an unrestricted model.

    :param model1: the final loglikelihood of one model, and the number of
                   estimated parameters.
    :type model1: tuple(float, int)

    :param model2: the final loglikelihood of the other model, and
                   the number of estimated parameters.
    :type model2: tuple(float, int)

    :param significance_level: level of significance of the test. Default: 0.05
    :type significance_level: float

    :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

    :rtype: LRTuple(str, float, float)

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
    threshold = chi2.ppf(1 - significance_level, chi_df)
    if stat <= threshold:
        final_msg = (
            f'H0 cannot be rejected at level {100*significance_level:.1f}%'
        )
    else:
        final_msg = (
            f'H0 can be rejected at level {100*significance_level:.1f}%'
        )
    return LRTuple(message=final_msg, statistic=stat, threshold=threshold)


def flatten_database(df, merge_id, row_name=None, identical_columns=None):
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
    grouped_varying = df[[merge_id] + list(varying_columns)].groupby(
        by=merge_id
    )

    def treat(x):
        """Treat a group of data.

        :param x: group of data
        :type x: pandas.DataFrame

        :return: the same data organized in one row, with proper column names
        :rtype: pandas.DataFrame

        :raise biogemeError:  if there are duplicates in the name of
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
                f'This column cannot be used to name the new '
                f'columns:\n{x[[row_name, merge_id]]}. '
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
    if identical_columns:
        return pd.concat([common_data, flat_data], axis='columns').drop(
            columns=[merge_id]
        )
    return flat_data.drop(columns=[merge_id])


def covariance_cross_nested(i, j, nests):
    """Calculate the covariance between the error terms of two
    alternatives of a cross-nested logit model. It is assumed that
    the homogeneity parameter mu of the model has been normalized
    to one.

    :param i: first alternative
    :type i: int

    :param j: first alternative
    :type j: int

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: value of the correlation
    :rtype: float

    :raise biogemeError: if the requested number is non positive or a float

    """
    set_of_alternatives = {alt for m in nests for alt in m[1]}

    if i not in set_of_alternatives:
        raise excep.biogemeError(f'Unknown alternative: {i}')
    if j not in set_of_alternatives:
        raise excep.biogemeError(f'Unknown alternative: {j}')

    if i == j:
        return np.pi * np.pi / 6.0

    def integrand(z_i, z_j):
        """Function to be integrated to calculate the correlation between
        alternative i and alternative j.

        :param z_i: argument corresponding to alternative i
        :type z_i: float

        :param z_j: argument corresponding to alternative j
        :type z_j: float
        """
        y_i = -np.log(z_i)
        y_j = -np.log(z_j)
        xi_i = -np.log(y_i)
        xi_j = -np.log(y_j)
        dy_i = -1 / z_i
        dy_j = -1 / z_j
        dxi_i = -dy_i / y_i
        dxi_j = -dy_j / y_j

        G_sum = 0.0
        Gi_sum = 0.0
        Gj_sum = 0.0
        Gij_sum = 0.0
        for m in nests:
            mu_m = m[0]
            alphas = m[1]
            alpha_i = alphas.get(i, 0)
            if alpha_i != 0:
                term_i = (alpha_i * y_i) ** mu_m
            else:
                term_i = 0
            alpha_j = alphas.get(j, 0)
            if alpha_j != 0:
                term_j = (alpha_j * y_j) ** mu_m
            else:
                term_j = 0
            the_sum = term_i + term_j
            p1 = (1.0 / mu_m) - 1
            p2 = (1.0 / mu_m) - 2
            G_sum += the_sum ** (1.0 / mu_m)
            if alpha_i != 0:
                Gi_sum += alpha_i**mu_m * y_i ** (mu_m - 1) * the_sum**p1
            if alpha_j != 0:
                Gj_sum += alpha_j**mu_m * y_j ** (mu_m - 1) * the_sum**p1
            if mu_m != 1.0 and alpha_i != 0 and alpha_j != 0:
                Gij_sum += (
                    (1 - mu_m)
                    * the_sum**p2
                    * (alpha_i * alpha_j) ** mu_m
                    * (y_i * y_j) ** (mu_m - 1)
                )

        F = np.exp(-G_sum)
        F_second = F * y_i * y_j * (Gi_sum * Gj_sum - Gij_sum)

        return xi_i * xi_j * F_second * dxi_i * dxi_j

    integral, _ = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
    return integral - np.euler_gamma * np.euler_gamma


def correlation_nested(nests):
    """Calculate the correlation matrix of the error terms of all
    alternatives of a nested logit model. It is assumed that the
    homogeneity parameter mu of the model has been normalized to one.

    :param nests: A tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb

    :type nests: tuple

    :return: correlation matrix
    :rtype: pd.DataFrame

    """
    set_of_alternatives = {alt for m in nests for alt in m[1]}
    list_of_alternatives = sorted(set_of_alternatives)
    index = {alt: i for i, alt in enumerate(list_of_alternatives)}
    J = len(list_of_alternatives)
    correlation = np.identity(J)
    for m in nests:
        mu_m = m[0]
        alt_m = m[1]
        for i, j in itertools.combinations(alt_m, 2):
            correlation[index[i]][index[j]] = correlation[index[j]][
                index[i]
            ] = 1.0 - 1.0 / (mu_m * mu_m)

    return pd.DataFrame(
        correlation, index=list_of_alternatives, columns=list_of_alternatives
    )


def correlation_cross_nested(nests):
    """Calculate the correlation matrix of the error terms of all
    alternatives of a cross-nested logit model. It is assumed that
    the homogeneity parameter mu of the model has been normalized
    to one.


    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: value of the correlation
    :rtype: float

    :raise biogemeError: if the requested number is non positive or a float

    :return: correlation matrix
    :rtype: pd.DataFrame

    """
    set_of_alternatives = {alt for m in nests for alt in m[1]}
    list_of_alternatives = sorted(set_of_alternatives)
    J = len(list_of_alternatives)

    covar = np.empty((J, J))
    for i, alt_i in enumerate(list_of_alternatives):
        for j, alt_j in enumerate(list_of_alternatives):
            covar[i][j] = covariance_cross_nested(alt_i, alt_j, nests)
            if i != j:
                covar[j][i] = covar[i][j]

    v = np.sqrt(np.diag(covar))
    outer_v = np.outer(v, v)
    correlation = covar / outer_v
    correlation[covar == 0] = 0
    return pd.DataFrame(
        correlation, index=list_of_alternatives, columns=list_of_alternatives
    )


def calculate_correlation(nests, results, alternative_names=None):
    """Calculate the correlation matrix of a nested or cross-nested
    logit model.

    :param nests:  A tuple containing as many items as nests.

        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,

        - for the nested logit model, a list containing the list of
          identifiers of the alternatives belonging to the nest.

        - for the cross-nested logit model, a dictionary mapping the
          alternative ids with the cross-nested parameters for the
          corresponding nest. If an alternative is missing in the
          dictionary, the corresponding alpha is set to zero.


        Example for the nested logit::
            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb

        Example for the cross-nested logit::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple(tuple(biogeme.expressions.Expression, list(int))), or
                 tuple(tuple(biogeme.Expression, dict(int:biogeme.expressions.Expression)))

    :param results: estimation results
    :type results: biogeme.results.bioResults

    :param alternative_names: a dictionary mapping the alternative IDs
        with their name. If None, the IDs are used as names.
    :type alternative_names: dict(int: str)
    """

    betas = results.getBetaValues()

    cnl = isinstance(nests[0][1], dict)

    def get_estimated_expression(expr):
        """Returns the estimated value of the nest parameter.

        :param expr: expression to calculate
        :type expr: biogeme.expressions.Expression or float.

        :return: calculated value
        :rtype: float

        :raise biogemeError: if the input value is not an expression
            or a float.

        """

        if isinstance(expr, Expression):
            expr.changeInitValues(betas)
            return expr.getValue_c(prepareIds=True)
        if isinstance(expr, (int, float)):
            return expr
        raise excep.biogemeError(f'Invalid type: {type(expr)}')

    def numerical_tuple(the_tuple):
        mu_m = get_estimated_expression(the_tuple[0])
        alpha_m = the_tuple[1]
        if isinstance(alpha_m, dict):
            # Cross-nested logit
            estimated_alpha_m = {
                alt
                if alternative_names is None
                else alternative_names[alt]: get_estimated_expression(e)
                for alt, e in alpha_m.items()
            }
            return mu_m, estimated_alpha_m
        return mu_m, [
            alt if alternative_names is None else alternative_names[alt]
            for alt in alpha_m
        ]

    estimated_nests = tuple(numerical_tuple(m) for m in nests)

    if cnl:
        return correlation_cross_nested(estimated_nests)
    return correlation_nested(estimated_nests)
