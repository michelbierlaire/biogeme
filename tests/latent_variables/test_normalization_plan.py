from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum

import pytest
from biogeme.latent_variables.normalization_plan import (
    ConflictPolicy,
    Fixing,
    NormalizationPlan,
)


class FakeParameterRef:
    """Minimal hashable/sortable stand-in for ParameterRef."""

    def __init__(self, name: str) -> None:
        self.name = name

    def key(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeParameterRef) and self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"FakeParameterRef({self.name!r})"


def make_ref(name: str) -> FakeParameterRef:
    return FakeParameterRef(name)


def make_fixing(name: str, value: float, note: str | None = None) -> Fixing:
    return Fixing(target=make_ref(name), value=value, note=note)


def test_conflict_policy_is_string_enum_with_expected_values() -> None:
    assert issubclass(ConflictPolicy, str)
    assert issubclass(ConflictPolicy, Enum)

    assert ConflictPolicy.ERROR.value == "error"
    assert ConflictPolicy.OVERWRITE.value == "overwrite"
    assert ConflictPolicy.IGNORE_SAME.value == "ignore_same"


def test_fixing_fields_are_preserved() -> None:
    ref = make_ref("beta_time")
    fixing = Fixing(target=ref, value=1.5, note="normalization")

    assert fixing.target is ref
    assert fixing.value == 1.5
    assert fixing.note == "normalization"


def test_fixing_note_defaults_to_none() -> None:
    fixing = Fixing(target=make_ref("beta_cost"), value=0.0)

    assert fixing.note is None


def test_fixing_is_frozen_and_slotted() -> None:
    fixing = Fixing(target=make_ref("beta_x"), value=2.0)

    assert hasattr(Fixing, "__slots__")
    assert "__dict__" not in dir(fixing)

    with pytest.raises(FrozenInstanceError):
        fixing.value = 3.0  # type: ignore[misc]


def test_plan_init_without_fixings_creates_empty_plan() -> None:
    plan = NormalizationPlan()

    assert len(plan) == 0
    assert list(plan) == []
    assert plan.as_list() == []


def test_plan_init_with_fixings_adds_all_items() -> None:
    f1 = make_fixing("b", 2.0, "second")
    f2 = make_fixing("a", 1.0, "first")

    plan = NormalizationPlan([f1, f2])

    assert len(plan) == 2
    assert plan.is_fixed(f1.target)
    assert plan.is_fixed(f2.target)
    assert plan.get(f1.target) == 2.0
    assert plan.get(f2.target) == 1.0


def test_len_tracks_number_of_distinct_targets() -> None:
    plan = NormalizationPlan()

    assert len(plan) == 0

    plan.add(make_fixing("a", 1.0))
    assert len(plan) == 1

    plan.add(make_fixing("b", 2.0))
    assert len(plan) == 2


def test_iter_returns_fixings_sorted_by_parameter_key() -> None:
    fixing_c = make_fixing("c", 3.0)
    fixing_a = make_fixing("a", 1.0)
    fixing_b = make_fixing("b", 2.0)

    plan = NormalizationPlan([fixing_c, fixing_a, fixing_b])

    ordered = list(plan)

    assert ordered == [fixing_a, fixing_b, fixing_c]


def test_add_inserts_new_fixing_when_target_absent() -> None:
    plan = NormalizationPlan()
    fixing = make_fixing("alpha", 4.2, "added")

    plan.add(fixing)

    assert len(plan) == 1
    assert plan.get_fixing(fixing.target) is fixing
    assert plan.get(fixing.target) == 4.2
    assert plan.is_fixed(fixing.target) is True


def test_add_same_value_default_keeps_existing_fixing() -> None:
    original = make_fixing("alpha", 1.0, "original")
    duplicate = make_fixing("alpha", 1.0, "duplicate")

    plan = NormalizationPlan([original])
    plan.add(duplicate)

    stored = plan.get_fixing(original.target)
    assert len(plan) == 1
    assert stored is original
    assert stored is not duplicate
    assert stored.note == "original"


def test_add_same_value_ignore_same_keeps_existing_fixing() -> None:
    original = make_fixing("alpha", 1.0, "original")
    duplicate = make_fixing("alpha", 1.0, "duplicate")

    plan = NormalizationPlan([original])
    plan.add(duplicate, on_conflict=ConflictPolicy.IGNORE_SAME)

    stored = plan.get_fixing(original.target)
    assert len(plan) == 1
    assert stored is original
    assert stored.note == "original"


def test_add_same_value_overwrite_replaces_existing_fixing() -> None:
    original = make_fixing("alpha", 1.0, "original")
    replacement = make_fixing("alpha", 1.0, "replacement")

    plan = NormalizationPlan([original])
    plan.add(replacement, on_conflict=ConflictPolicy.OVERWRITE)

    stored = plan.get_fixing(original.target)
    assert len(plan) == 1
    assert stored is replacement
    assert stored.note == "replacement"


def test_add_conflicting_value_with_overwrite_replaces_existing_fixing() -> None:
    original = make_fixing("alpha", 1.0, "original")
    replacement = make_fixing("alpha", 2.0, "replacement")

    plan = NormalizationPlan([original])
    plan.add(replacement, on_conflict=ConflictPolicy.OVERWRITE)

    stored = plan.get_fixing(original.target)
    assert len(plan) == 1
    assert stored is replacement
    assert plan.get(original.target) == 2.0
    assert stored.note == "replacement"


def test_add_conflicting_value_default_error_raises() -> None:
    original = make_fixing("alpha", 1.0)
    conflicting = make_fixing("alpha", 2.0)

    plan = NormalizationPlan([original])

    with pytest.raises(
        ValueError,
        match=r"Conflicting fixings for 'alpha': 1\.0 vs 2\.0\.",
    ):
        plan.add(conflicting)


def test_add_conflicting_value_ignore_same_still_raises() -> None:
    original = make_fixing("alpha", 1.0)
    conflicting = make_fixing("alpha", 2.0)

    plan = NormalizationPlan([original])

    with pytest.raises(
        ValueError,
        match=r"Conflicting fixings for 'alpha': 1\.0 vs 2\.0\.",
    ):
        plan.add(conflicting, on_conflict=ConflictPolicy.IGNORE_SAME)


def test_get_returns_none_for_missing_target() -> None:
    plan = NormalizationPlan()

    assert plan.get(make_ref("missing")) is None


def test_get_returns_value_for_present_target() -> None:
    fixing = make_fixing("beta", 7.5)
    plan = NormalizationPlan([fixing])

    assert plan.get(make_ref("beta")) == 7.5


def test_get_fixing_returns_none_for_missing_target() -> None:
    plan = NormalizationPlan()

    assert plan.get_fixing(make_ref("missing")) is None


def test_get_fixing_returns_fixing_for_present_target() -> None:
    fixing = make_fixing("beta", 7.5, "kept")
    plan = NormalizationPlan([fixing])

    assert plan.get_fixing(make_ref("beta")) is fixing


def test_is_fixed_is_false_for_missing_target() -> None:
    plan = NormalizationPlan()

    assert plan.is_fixed(make_ref("missing")) is False


def test_is_fixed_is_true_for_present_target() -> None:
    fixing = make_fixing("beta", 7.5)
    plan = NormalizationPlan([fixing])

    assert plan.is_fixed(make_ref("beta")) is True


def test_as_list_returns_sorted_fixings() -> None:
    fixing_c = make_fixing("c", 3.0)
    fixing_a = make_fixing("a", 1.0)
    fixing_b = make_fixing("b", 2.0)

    plan = NormalizationPlan([fixing_c, fixing_a, fixing_b])

    assert plan.as_list() == [fixing_a, fixing_b, fixing_c]


def test_as_list_returns_new_list_each_time() -> None:
    fixing = make_fixing("a", 1.0)
    plan = NormalizationPlan([fixing])

    first = plan.as_list()
    second = plan.as_list()

    assert first == [fixing]
    assert second == [fixing]
    assert first is not second


def test_iter_and_as_list_reflect_overwritten_entry() -> None:
    original = make_fixing("x", 1.0, "old")
    replacement = make_fixing("x", 2.0, "new")
    other = make_fixing("a", 0.0, "other")

    plan = NormalizationPlan([original, other])
    plan.add(replacement, on_conflict=ConflictPolicy.OVERWRITE)

    assert list(plan) == [other, replacement]
    assert plan.as_list() == [other, replacement]
