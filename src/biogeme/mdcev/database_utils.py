"""Implementation of useful database manipulations for the MDCEV model.

Michel Bierlaire
Fri Jul 25 2025, 16:45:19
"""

from collections.abc import Iterable

from biogeme.database import Database


def mdcev_count(
    database: Database, list_of_columns: list[str], new_column: str
) -> None:
    """For the MDCEV models, we calculate the number of
        alternatives that are chosen, that is the number of
        columns with a non zero entry, and add this as a new column

    :param database: database to modify
    :param list_of_columns: list of columns containing the quantity of each good.
    :param new_column: name of the new column where the result is stored
    """
    database.dataframe[new_column] = database.dataframe[list_of_columns].apply(
        lambda x: (x != 0).sum(), axis=1
    )


def mdcev_row_split(
    database: Database, a_range: Iterable[int] | None = None
) -> list[Database]:
    """
    For the MDCEV model, we generate a list of Database objects, each of them associated with a different row of
    the database,

    :param database: input database.
    :param a_range: specify the desired range of rows.
    :return: list of rows, each in a Database format
    """
    if a_range is None:
        the_range = range(len(database.dataframe))
    else:
        # Validate the provided range
        max_index = len(database.dataframe) - 1
        if any(i < 0 or i > max_index for i in a_range):
            raise IndexError(
                'One or more indices in a_range are out of the valid range.'
            )
        the_range = a_range

    rows_of_database = [
        Database(name=f'row_{i}', dataframe=database.dataframe.iloc[[i]])
        for i in the_range
    ]
    return rows_of_database
