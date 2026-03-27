from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from biogeme.latent_variables.normalization_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    ParameterRef,
    StructuralCoefficient,
    StructuralSigma,
    ThresholdDelta,
    ThresholdFirst,
)


def test_parameter_ref_key_returns_class_name_tuple() -> None:
    ref = ParameterRef()

    assert ref.key() == ("ParameterRef",)


def test_parameter_ref_is_frozen_and_slotted() -> None:
    ref = ParameterRef()

    assert hasattr(ParameterRef, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        ref.some_attribute = "x"  # type: ignore[misc,attr-defined]


def test_structural_coefficient_fields_and_key() -> None:
    ref = StructuralCoefficient(latent_name="LV1", variable_name="income")

    assert ref.latent_name == "LV1"
    assert ref.variable_name == "income"
    assert ref.key() == ("StructuralCoefficient", "LV1", "income")


def test_structural_coefficient_is_parameter_ref() -> None:
    ref = StructuralCoefficient(latent_name="LV_A", variable_name="time")

    assert isinstance(ref, ParameterRef)


def test_structural_coefficient_is_frozen_and_slotted() -> None:
    ref = StructuralCoefficient(latent_name="LV1", variable_name="x1")

    assert hasattr(StructuralCoefficient, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.latent_name = "LV2"  # type: ignore[misc]


def test_structural_sigma_fields_and_key() -> None:
    ref = StructuralSigma(latent_name="LV2")

    assert ref.latent_name == "LV2"
    assert ref.key() == ("StructuralSigma", "LV2")


def test_structural_sigma_is_parameter_ref() -> None:
    ref = StructuralSigma(latent_name="LV_B")

    assert isinstance(ref, ParameterRef)


def test_structural_sigma_is_frozen_and_slotted() -> None:
    ref = StructuralSigma(latent_name="LV3")

    assert hasattr(StructuralSigma, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.latent_name = "other"  # type: ignore[misc]


def test_measurement_intercept_fields_and_key() -> None:
    ref = MeasurementIntercept(indicator_name="indicator_1")

    assert ref.indicator_name == "indicator_1"
    assert ref.key() == ("MeasurementIntercept", "indicator_1")


def test_measurement_intercept_is_parameter_ref() -> None:
    ref = MeasurementIntercept(indicator_name="ind_a")

    assert isinstance(ref, ParameterRef)


def test_measurement_intercept_is_frozen_and_slotted() -> None:
    ref = MeasurementIntercept(indicator_name="ind_x")

    assert hasattr(MeasurementIntercept, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.indicator_name = "ind_y"  # type: ignore[misc]


def test_measurement_loading_fields_and_key() -> None:
    ref = MeasurementLoading(latent_name="LV1", indicator_name="indicator_2")

    assert ref.latent_name == "LV1"
    assert ref.indicator_name == "indicator_2"
    assert ref.key() == ("MeasurementLoading", "LV1", "indicator_2")


def test_measurement_loading_is_parameter_ref() -> None:
    ref = MeasurementLoading(latent_name="LV_C", indicator_name="ind_c")

    assert isinstance(ref, ParameterRef)


def test_measurement_loading_is_frozen_and_slotted() -> None:
    ref = MeasurementLoading(latent_name="LV1", indicator_name="ind1")

    assert hasattr(MeasurementLoading, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.indicator_name = "ind2"  # type: ignore[misc]


def test_measurement_sigma_fields_and_key() -> None:
    ref = MeasurementSigma(indicator_name="indicator_3")

    assert ref.indicator_name == "indicator_3"
    assert ref.key() == ("MeasurementSigma", "indicator_3")


def test_measurement_sigma_is_parameter_ref() -> None:
    ref = MeasurementSigma(indicator_name="ind_sigma")

    assert isinstance(ref, ParameterRef)


def test_measurement_sigma_is_frozen_and_slotted() -> None:
    ref = MeasurementSigma(indicator_name="ind3")

    assert hasattr(MeasurementSigma, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.indicator_name = "changed"  # type: ignore[misc]


def test_threshold_first_fields_and_key() -> None:
    ref = ThresholdFirst(type_name="likert5")

    assert ref.type_name == "likert5"
    assert ref.key() == ("ThresholdFirst", "likert5")


def test_threshold_first_is_parameter_ref() -> None:
    ref = ThresholdFirst(type_name="agreement_scale")

    assert isinstance(ref, ParameterRef)


def test_threshold_first_is_frozen_and_slotted() -> None:
    ref = ThresholdFirst(type_name="type_a")

    assert hasattr(ThresholdFirst, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.type_name = "type_b"  # type: ignore[misc]


def test_threshold_delta_fields_and_key() -> None:
    ref = ThresholdDelta(type_name="likert7", index=3)

    assert ref.type_name == "likert7"
    assert ref.index == 3
    assert ref.key() == ("ThresholdDelta", "likert7", 3)


def test_threshold_delta_accepts_zero_and_negative_indices() -> None:
    zero_ref = ThresholdDelta(type_name="type_zero", index=0)
    negative_ref = ThresholdDelta(type_name="type_neg", index=-1)

    assert zero_ref.key() == ("ThresholdDelta", "type_zero", 0)
    assert negative_ref.key() == ("ThresholdDelta", "type_neg", -1)


def test_threshold_delta_is_parameter_ref() -> None:
    ref = ThresholdDelta(type_name="ordered", index=2)

    assert isinstance(ref, ParameterRef)


def test_threshold_delta_is_frozen_and_slotted() -> None:
    ref = ThresholdDelta(type_name="type_c", index=1)

    assert hasattr(ThresholdDelta, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.index = 99  # type: ignore[misc]


def test_dataclass_equality_and_distinct_keys_across_reference_types() -> None:
    structural_coefficient_1 = StructuralCoefficient(
        latent_name="LV", variable_name="x"
    )
    structural_coefficient_2 = StructuralCoefficient(
        latent_name="LV", variable_name="x"
    )
    structural_sigma = StructuralSigma(latent_name="LV")
    measurement_intercept = MeasurementIntercept(indicator_name="x")
    measurement_loading = MeasurementLoading(latent_name="LV", indicator_name="x")
    measurement_sigma = MeasurementSigma(indicator_name="x")
    threshold_first = ThresholdFirst(type_name="LV")
    threshold_delta = ThresholdDelta(type_name="LV", index=1)

    assert structural_coefficient_1 == structural_coefficient_2

    keys = {
        structural_coefficient_1.key(),
        structural_sigma.key(),
        measurement_intercept.key(),
        measurement_loading.key(),
        measurement_sigma.key(),
        threshold_first.key(),
        threshold_delta.key(),
    }
    assert len(keys) == 7


def test_references_are_hashable() -> None:
    refs = {
        ParameterRef(),
        StructuralCoefficient(latent_name="LV1", variable_name="x1"),
        StructuralSigma(latent_name="LV1"),
        MeasurementIntercept(indicator_name="ind1"),
        MeasurementLoading(latent_name="LV1", indicator_name="ind1"),
        MeasurementSigma(indicator_name="ind1"),
        ThresholdFirst(type_name="type1"),
        ThresholdDelta(type_name="type1", index=2),
    }

    assert len(refs) == 8
