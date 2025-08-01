import logging
from typing import NamedTuple
from itertools import chain

from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)


class AuditTuple(NamedTuple):
    errors: list[str]
    warnings: list[str]


def merge_audit_tuples(audit_tuples: list[AuditTuple]) -> AuditTuple:
    merged_errors = list(chain.from_iterable(a.errors for a in audit_tuples))
    merged_warnings = list(chain.from_iterable(a.warnings for a in audit_tuples))
    return AuditTuple(errors=merged_errors, warnings=merged_warnings)


def display_messages(audit_tuple: AuditTuple) -> None:
    for warning in audit_tuple.warnings:
        if warning:
            logger.warning(warning)
    if audit_tuple.errors:
        raise BiogemeError(audit_tuple.errors)
