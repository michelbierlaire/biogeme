import pandas as pd

from biogeme.database import Database
from biogeme.expressions import ExpressionOrNumeric, validate_and_convert
from ..audit_tuple import AuditTuple

CHOICE_LABEL = 'Choice'
AVAILABILITY_LABEL = 'Avail. '


def audit_chosen_alternative(
    choice: ExpressionOrNumeric,
    availability: dict[int, ExpressionOrNumeric],
    database: Database,
) -> AuditTuple:
    from .model_elements import ModelElements
    from biogeme.calculator import MultiRowEvaluator

    """Checks all the rows in the database such that the chosen alternative is not available"""

    list_of_warnings = []

    dict_of_expressions = {CHOICE_LABEL: choice} | {
        f'{AVAILABILITY_LABEL}{alt_id:.1f}': the_expression
        for alt_id, the_expression in availability.items()
    }
    model_elements = ModelElements(expressions=dict_of_expressions, database=database)

    the_evaluator: MultiRowEvaluator = MultiRowEvaluator(model_elements=model_elements)

    results: pd.DataFrame = the_evaluator.evaluate({})

    def chosen_unavailable(row):
        choice_id = row[CHOICE_LABEL]
        return row[f'{AVAILABILITY_LABEL}{choice_id:.1f}'] == 0

    invalid_indices = results.apply(chosen_unavailable, axis=1)
    problematic_rows = results[invalid_indices]

    list_of_errors = [
        f'Row index {idx}: chosen alternative {row[CHOICE_LABEL]} is not available'
        for idx, row in problematic_rows.iterrows()
    ]
    return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)
