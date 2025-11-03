import pandas as pd

from biogeme.audit_tuple import AuditTuple
from biogeme.database import Database
from biogeme.default_parameters import MISSING_VALUE
from biogeme.expressions import (
    Expression,
    ExpressionOrNumeric,
    PanelLikelihoodTrajectory,
    Variable,
    list_of_variables_in_expression,
)

CHOICE_LABEL = 'Choice'
AVAILABILITY_LABEL = 'Avail. '


def audit_variables(expression: Expression, database: Database) -> AuditTuple:

    all_variables: list[Variable] = list_of_variables_in_expression(expression)
    list_of_errors = []
    list_of_warnings = []
    for variable in all_variables:
        if variable.name not in database.dataframe.columns:
            error_msg = f'Variable "{variable.name}" not found in the database. Available variables: {database.dataframe.columns}'
            list_of_errors.append(error_msg)

    return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)


def audit_panel(expression: Expression, database: Database) -> AuditTuple:
    list_of_errors = []
    list_of_warnings = []
    if not database.is_panel():
        return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)

    all_variables: list[Variable] = list_of_variables_in_expression(expression)
    if all_variables and not expression.embed_expression(PanelLikelihoodTrajectory):
        error_msg = (
            f'Expression {expression} does not contain  "PanelLikelihoodTrajectory" although the data has been '
            f'declared to have a panel structure.'
        )
        list_of_errors.append(error_msg)
    return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)


def audit_chosen_alternative(
    choice: ExpressionOrNumeric,
    availability: dict[int, ExpressionOrNumeric],
    database: Database,
    use_jit: bool,
) -> AuditTuple:
    from .model_elements import ModelElements
    from biogeme.jax_calculator import MultiRowEvaluator

    """Checks all the rows in the database such that the chosen alternative is not available"""

    list_of_errors = []

    dict_of_expressions = {CHOICE_LABEL: choice} | {
        f'{AVAILABILITY_LABEL}{alt_id:.1f}': the_expression
        for alt_id, the_expression in availability.items()
    }
    model_elements = ModelElements(
        expressions=dict_of_expressions, database=database, use_jit=use_jit
    )

    the_evaluator: MultiRowEvaluator = MultiRowEvaluator(
        model_elements=model_elements, numerically_safe=True, use_jit=use_jit
    )
    results: pd.DataFrame = the_evaluator.evaluate({})

    def chosen_unavailable(row):
        choice_id = row[CHOICE_LABEL]
        if choice_id == MISSING_VALUE:
            return False
        return row[f'{AVAILABILITY_LABEL}{choice_id:.1f}'] == 0

    invalid_indices = results.apply(chosen_unavailable, axis=1)
    problematic_rows = results[invalid_indices]

    list_of_warnings = [
        f'Row index {idx}: chosen alternative {row[CHOICE_LABEL]} is not available'
        for idx, row in problematic_rows.iterrows()
    ]
    return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)
