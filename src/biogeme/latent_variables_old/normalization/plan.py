# normalization/plan.py
"""
Normalization plan objects.

A normalization plan is a set of explicit fixings: each fixing identifies a
target parameter (as a `ParameterRef`) and a value to impose.

This module contains no builder logic and no Biogeme imports.

Michel Bierlaire
Wed Mar 04 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .parameter_refs import ParameterRef


class ConflictPolicy(str, Enum):
    """Policy to resolve conflicts when adding or merging fixings."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    IGNORE_SAME = "ignore_same"


@dataclass(frozen=True, slots=True)
class Fixing:
    """A single parameter fixing.

    :param target:
        Semantic reference to the fixed parameter.
    :param value:
        Numeric value to impose.
    :param note:
        Optional note (e.g., 'reference indicator', 'user override').
    """

    target: ParameterRef
    value: float
    note: str | None = None


class NormalizationPlan:
    """A collection of fixings (expert-mode normalization).

    The plan is intentionally minimal: it is simply a set of explicit fixings.
    Builders consult the plan to decide whether to create a free parameter or
    return a constant value.

    The plan supports conflict handling policies to allow composition of plans.

    Notes
    -----
    - An empty plan is valid.
    - This class does not attempt to judge model identification.
    """

    def __init__(self, fixings: Iterable[Fixing] | None = None) -> None:
        self._fixings: dict[ParameterRef, Fixing] = {}
        if fixings is not None:
            for f in fixings:
                self.add(f)

    def __len__(self) -> int:
        return len(self._fixings)

    def __iter__(self):
        # Deterministic ordering for display/debugging
        for ref in sorted(self._fixings.keys(), key=lambda r: r.key()):
            yield self._fixings[ref]

    def add(
        self, fixing: Fixing, *, on_conflict: ConflictPolicy = ConflictPolicy.ERROR
    ) -> None:
        """Add a fixing to the plan.

        :param fixing:
            The fixing to add.
        :param on_conflict:
            How to resolve a conflict if the target is already fixed.
        :raises ValueError:
            If the target is already fixed with a different value and
            ``on_conflict`` is ERROR or IGNORE_SAME.
        """
        existing = self._fixings.get(fixing.target)
        if existing is None:
            self._fixings[fixing.target] = fixing
            return

        if existing.value == fixing.value:
            if on_conflict in (
                ConflictPolicy.ERROR,
                ConflictPolicy.IGNORE_SAME,
                ConflictPolicy.OVERWRITE,
            ):
                # Same value: always safe. OVERWRITE can update note if desired.
                if on_conflict == ConflictPolicy.OVERWRITE:
                    self._fixings[fixing.target] = fixing
                return

        # Different value
        if on_conflict == ConflictPolicy.OVERWRITE:
            self._fixings[fixing.target] = fixing
            return

        raise ValueError(
            f"Conflicting fixings for '{fixing.target}': "
            f"{existing.value} vs {fixing.value}."
        )

    def merge(
        self,
        other: "NormalizationPlan",
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.ERROR,
    ) -> "NormalizationPlan":
        """Return a new plan that is the merge of this plan with another.

        :param other:
            Another normalization plan.
        :param conflict_policy:
            Conflict resolution policy.
        :return:
            A new plan containing fixings from both plans.
        """
        merged = NormalizationPlan(self.as_list())
        for f in other.as_list():
            merged.add(f, on_conflict=conflict_policy)
        return merged

    def as_list(self) -> list[Fixing]:
        """Return fixings as a deterministically ordered list."""
        return list(iter(self))

    def is_fixed(self, target: ParameterRef) -> bool:
        """Check if a target is fixed."""
        return target in self._fixings

    def get(self, target: ParameterRef) -> float | None:
        """Return the fixed value for a target, or None if not fixed."""
        fixing = self._fixings.get(target)
        return None if fixing is None else fixing.value

    def get_fixing(self, target: ParameterRef) -> Fixing | None:
        """Return the full Fixing object for a target, or None if not fixed."""
        return self._fixings.get(target)

    def require(self, target: ParameterRef) -> float:
        """Return the fixed value for a target, raising if not fixed."""
        fixing = self._fixings.get(target)
        if fixing is None:
            raise KeyError(f"Target '{target}' is not fixed in the normalization plan.")
        return fixing.value
