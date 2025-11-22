from typing import NamedTuple

from biogeme.database import Database
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.validation.prepare_validation import split


class EstimationValidationModels(NamedTuple):
    estimation: ModelElements
    validation: ModelElements


def split_databases(
    model_elements: ModelElements, slices: int, groups: str | None = None
) -> list[EstimationValidationModels]:
    """

    :param model_elements: modeling elements, including the database and the draws that will be split.
    :param slices: The number of folds/slices. Must be >= 2.
    :param groups: Optional name of the column containing group identifiers.
                       If provided, all rows with the same group ID are kept in the same fold.

    :return: A list of EstimationValidationIndices tuples, one per fold.
    """
    slices = split(
        dataframe=model_elements.database.dataframe, slices=slices, groups=groups
    )
    results = []
    database_name = model_elements.database.name
    for index, split_indices in enumerate(slices, 1):
        estimation_df = model_elements.database.dataframe.iloc[split_indices.estimation]
        estimation_data = Database(
            name=f'{database_name} estimation {index}',
            dataframe=estimation_df,
        )
        estimation_draws_management = model_elements.draws_management.extract_slice(
            split_indices.estimation
        )
        estimation_model_elements = ModelElements(
            expressions=model_elements.expressions,
            adapter=RegularAdapter(database=estimation_data),
            draws_management=estimation_draws_management,
            use_jit=model_elements.use_jit,
        )
        validation_df = model_elements.database.dataframe.iloc[split_indices.validation]
        validation_data = Database(
            name=f'{database_name} estimation {index}',
            dataframe=validation_df,
        )
        validation_draws_management = model_elements.draws_management.extract_slice(
            split_indices.validation
        )
        validation_model_elements = ModelElements(
            expressions={
                f'{name} [validation fold {index}]': expression
                for name, expression in model_elements.expressions.items()
            },
            adapter=RegularAdapter(database=validation_data),
            draws_management=validation_draws_management,
            use_jit=model_elements.use_jit,
        )
        the_pair = EstimationValidationModels(
            estimation=estimation_model_elements, validation=validation_model_elements
        )
        results.append(the_pair)
    return results
