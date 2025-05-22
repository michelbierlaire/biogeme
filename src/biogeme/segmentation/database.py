import logging

import pandas as pd

from biogeme.expressions import Variable
from .segmentation import DiscreteSegmentationTuple
from ..exceptions import BiogemeError

logger = logging.getLogger(__name__)
"""Logger that controls the output of
        messages to the screen and log file.
        """


def generate_segmentation(
    dataframe: pd.DataFrame,
    variable: Variable | str,
    mapping: dict[int, str] | None = None,
    reference: str | None = None,
) -> DiscreteSegmentationTuple:
    """Generate a segmentation tuple for a variable.

    :param dataframe: data frame.
    :param variable: Variable object or name of the variable
    :param mapping: mapping associating values of the variable to
        names. If incomplete, default names are provided.
    :param reference: name of the reference category. If None, an
        arbitrary category is selected as reference.

    """

    the_variable = variable if isinstance(variable, Variable) else Variable(variable)

    # Check if the variable is in the database.
    if the_variable.name not in dataframe.columns:
        error_msg = f'Unknown the_variable {the_variable.name}'
        raise BiogemeError(error_msg)

    # Extract all unique values from the data base.
    unique_values = set(dataframe[the_variable.name].unique())

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

    the_mapping = {
        int(value): f'{the_variable.name}_{int(value)}' for value in unique_values
    }

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


def verify_segmentation(
    dataframe: pd.DataFrame, segmentation: DiscreteSegmentationTuple
) -> None:
    """Verifies if the definition of the segmentation is consistent with the data

    :param dataframe: dataframe to check.
    :param segmentation: definition of the segmentation
    :raise BiogemeError: if the segmentation is not consistent with the data.
    """

    variable = segmentation.variable

    # Check if the variable is in the database.
    if variable.name not in dataframe.columns:
        error_msg = f'Unknown variable {variable.name}'
        raise BiogemeError(error_msg)

    # Extract all unique values from the data base.
    unique_values = set(dataframe[variable.name].unique())
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
