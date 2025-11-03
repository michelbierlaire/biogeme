"""Audit procedure for arithmetic expressions

Michel Bierlaire
Fri Mar 28 17:01:01 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from biogeme.audit_tuple import AuditTuple

from .belongs_to import BelongsTo
from .comparison_expressions import ComparisonOperator
from .draws import Draws
from .integrate import IntegrateNormal
from .logit_expressions import LogLogit
from .montecarlo import MonteCarlo
from .random_variable import RandomVariable
from .visitor import ExpressionVisitor

_audit_visitor = ExpressionVisitor()
register_audit = _audit_visitor.register

if TYPE_CHECKING:
    from .base_expressions import Expression


def audit_expression(expr: Expression) -> AuditTuple:
    """
    Audits an expression tree for structural consistency.

    :param expr: The root expression to audit.
    :return: A tuple containing two lists:
             - error_messages: list of error strings.
             - warning_messages: list of warning strings.
    """
    error_messages = []
    warning_messages = []
    ancestors = []
    context = {
        "errors": error_messages,
        "warnings": warning_messages,
        "ancestors": ancestors,  # Stack of ancestor expressions
    }
    _audit_visitor.visit(expr, context)
    return AuditTuple(errors=error_messages, warnings=warning_messages)


def audit_default(expr: Expression, context: dict[str, list[str]]) -> None:
    """
    Default audit function for expressions that do not have a specific audit function.

    :param expr: The current expression node being audited.
    :param context: Dictionary to collect error and warning messages.
    """
    if hasattr(expr, 'left') and hasattr(expr, 'right'):
        if isinstance(expr, ComparisonOperator):
            if isinstance(expr.left, ComparisonOperator) or isinstance(
                expr.right, ComparisonOperator
            ):
                warning = (
                    f"The expression [{expr}] may be a chained comparison (e.g., a <= b <= c), "
                    f"which Biogeme interprets as nested comparisons, not as Python-style chains. "
                    f"Consider splitting it: (a <= b) & (b <= c)."
                )
                context["warnings"].append(warning)


@register_audit(MonteCarlo)
def audit_montecarlo(expr: MonteCarlo, context: dict[str, list[str]]) -> None:
    """
    Audits a MonteCarlo expression for structural consistency.

    :param expr: The MonteCarlo expression to audit.
    :param context: Dictionary to collect error and warning messages.
    """
    if not expr.embed_expression(Draws):
        context["errors"].append(
            f'MonteCarlo expression {repr(expr)} does not contain any Draws expression.'
        )
    if any(child.embed_expression(MonteCarlo) for child in expr.get_children()):
        context["errors"].append(
            f'MonteCarlo expression {repr(expr)} cannot contain another MonteCarlo expression.'
        )


@register_audit(IntegrateNormal)
def audit_integrate(expr: IntegrateNormal, context: dict[str, list[str]]) -> None:
    """
    Audits an Integrate expression for structural consistency.

    :param expr: The Integrate expression to audit.
    :param context: Dictionary to collect error and warning messages.
    """
    if not expr.embed_expression(RandomVariable):
        context["warnings"].append(
            f'Integrate expression {repr(expr)} does not contain any RandomVariable expression.'
        )


@register_audit(BelongsTo)
def audit_belongsto(expr: BelongsTo, context: dict[str, list[str]]) -> None:
    """
    Audits a BelongsTo expression for structural consistency.

    :param expr: The BelongsTo expression to audit.
    :param context: Dictionary to collect error and warning messages.
    """
    if not all(float(x).is_integer() for x in expr.the_set):
        the_warning = (
            f'The set of numbers used in the expression "BelongsTo" contains '
            f'numbers that are not integer. If it is the intended use, ignore '
            f'this warning: {expr.the_set}.'
        )
        context["warnings"].append(the_warning)


@register_audit(LogLogit)
def audit_loglogit(expr: LogLogit, context: dict[str, list[str]]) -> None:
    """
    Audits a LogLogit expression for structural consistency.

    :param expr: The LogLogit expression to audit.
    :param context: Dictionary to collect error and warning messages.
    """
    if expr.av is None:
        return
    if expr.util.keys() != expr.av.keys():
        the_error = 'Incompatible list of alternatives in logit expression. '
        my_set = expr.util.keys() - expr.av.keys()
        if my_set:
            my_set_content = ', '.join(f'{str(k)} ' for k in my_set)
            the_error += (
                'Id(s) used for utilities and not for availabilities: '
            ) + my_set_content
        my_set = expr.av.keys() - expr.util.keys()
        if my_set:
            my_set_content = ', '.join(f'{str(k)} ' for k in my_set)
            the_error += (
                ' Id(s) used for availabilities and not for utilities: '
            ) + my_set_content
        context["errors"].append(the_error)


@register_audit(RandomVariable)
def audit_randomvariable(expr: RandomVariable, context: dict[str, list[str]]) -> None:
    if not any(
        isinstance(ancestor, IntegrateNormal) for ancestor in context['ancestors']
    ):
        context['errors'].append(
            f'RandomVariable {repr(expr)} is not embedded inside an IntegrateNormal expression.'
        )


@register_audit(Draws)
def audit_draws(expr: Draws, context: dict[str, list[str]]) -> None:
    if not any(isinstance(ancestor, MonteCarlo) for ancestor in context['ancestors']):
        context['warnings'].append(
            f'Draws {repr(expr)} is not embedded inside a MonteCarlo expression. For maximum likelihood estimation, it is a fatal error. For Bayesian estimation, it is not a problem.'
        )
