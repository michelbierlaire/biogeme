import pandas as pd

from biogeme.exceptions import BiogemeError
from biogeme.deprecated import deprecated


def count_number_of_groups(df: pd.DataFrame, column: str) -> int:
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
        >>>count_number_of_groups(df,'ID')
        6

        >>>count_number_of_groups(df,'value')
        7

    """
    df['_bio_groups'] = pd.Series(df[column] != df[column].shift(1)).cumsum()
    result = len(df['_bio_groups'].unique())
    df.drop(columns=['_bio_groups'], inplace=True)
    return result


@deprecated(count_number_of_groups)
def countNumberOfGroups(df: pd.DataFrame, column: str) -> int:
    pass


def flatten_database(
    df: pd.DataFrame,
    merge_id: str,
    row_name: str | None = None,
    identical_columns: list[str] | None = None,
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
    df_copy = df.copy()
    all_columns = set(df_copy.columns)
    duplicate = f'{merge_id}_biogeme_tmp_duplicate'
    df_copy[duplicate] = df_copy.loc[:, merge_id]
    grouped = df_copy.groupby(by=duplicate)

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
        all_varying_cols = grouped.apply(get_varying_cols, include_groups=False)
        varying_columns = set.union(*all_varying_cols)
        identical_columns = list(all_columns - varying_columns)
        varying_columns = list(varying_columns)
    else:
        identical_columns = set(identical_columns)
        identical_columns.add(merge_id)
        varying_columns = list(all_columns - identical_columns)

    # Take the first row for columns that are identical
    if identical_columns:
        common_data = df_copy[list(identical_columns)].drop_duplicates(
            merge_id, keep='first'
        )
        common_data.index = common_data[merge_id]
    # Treat the other columns
    # Include merge_id and a duplicate
    tmp_df = df_copy[[merge_id] + list(varying_columns)].copy()
    tmp_df[duplicate] = tmp_df[merge_id].copy()
    grouped_varying = tmp_df.groupby(by=duplicate)

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
            raise BiogemeError(err_msg)
        if row_name is not None and not x[row_name].is_unique:
            err_msg = (
                f'Entries in column [{row_name}] are not unique. '
                f'This column cannot be used to name the new '
                f'columns:\n{x[[row_name, merge_id]]}. '
            )
            raise BiogemeError(err_msg)

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
        treated_df = pd.DataFrame([all_values], columns=all_columns)
        return treated_df

    flat_data = grouped_varying.apply(treat, include_groups=False)
    flat_data.index = flat_data[merge_id]

    # We remove the column 'merge_id' as it is stored as index.
    if identical_columns:
        return pd.concat([common_data, flat_data], axis='columns').drop(
            columns=[merge_id]
        )
    return flat_data.drop(columns=[merge_id])
