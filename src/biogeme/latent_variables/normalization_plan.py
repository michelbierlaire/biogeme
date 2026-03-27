from __future__ import annotations

"""Normalization plan objects."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .normalization_refs import ParameterRef


class ConflictPolicy(str, Enum):
    ERROR = 'error'
    OVERWRITE = 'overwrite'
    IGNORE_SAME = 'ignore_same'


@dataclass(frozen=True, slots=True)
class Fixing:
    """One explicit parameter fixing.

    :param target: Semantic parameter reference.
    :param value: Numeric value to impose.
    :param note: Optional human-readable note.
    """

    target: ParameterRef
    value: float
    note: str | None = None


class NormalizationPlan:
    """Collection of explicit fixings.

    The plan itself does not decide whether a fixing is meaningful for a given
    model. That responsibility belongs to validation.
    """

    def __init__(self, fixings: Iterable[Fixing] | None = None) -> None:
        self._fixings: dict[ParameterRef, Fixing] = {}
        if fixings is not None:
            for fixing in fixings:
                self.add(fixing)

    def __len__(self) -> int:
        return len(self._fixings)

    def __iter__(self):
        for ref in sorted(self._fixings, key=lambda r: r.key()):
            yield self._fixings[ref]

    def add(self, fixing: Fixing, *, on_conflict: ConflictPolicy = ConflictPolicy.ERROR) -> None:
        existing = self._fixings.get(fixing.target)
        if existing is None:
            self._fixings[fixing.target] = fixing
            return
        if existing.value == fixing.value:
            if on_conflict == ConflictPolicy.OVERWRITE:
                self._fixings[fixing.target] = fixing
            return
        if on_conflict == ConflictPolicy.OVERWRITE:
            self._fixings[fixing.target] = fixing
            return
        raise ValueError(
            f"Conflicting fixings for '{fixing.target}': {existing.value} vs {fixing.value}."
        )

    def get(self, target: ParameterRef) -> float | None:
        fixing = self._fixings.get(target)
        return None if fixing is None else fixing.value

    def get_fixing(self, target: ParameterRef) -> Fixing | None:
        return self._fixings.get(target)

    def is_fixed(self, target: ParameterRef) -> bool:
        return target in self._fixings

    def as_list(self) -> list[Fixing]:
        return list(iter(self))
