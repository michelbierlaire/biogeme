from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from biogeme.latent_variables.naming import DefaultNamingPolicy, NamingPolicy


def test_default_naming_policy_is_frozen_and_slotted() -> None:
    policy = DefaultNamingPolicy()

    assert hasattr(DefaultNamingPolicy, "__slots__")
    assert "__dict__" not in dir(policy)

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        policy.some_attribute = "x"  # type: ignore[misc,attr-defined]


def test_default_naming_policy_equality_and_repr() -> None:
    left = DefaultNamingPolicy()
    right = DefaultNamingPolicy()

    assert left == right
    assert repr(left) == "DefaultNamingPolicy()"


def test_default_naming_policy_structural_prefix() -> None:
    policy = DefaultNamingPolicy()

    result = policy.structural_prefix("latent_var")

    assert result == "struct_latent_var"


def test_default_naming_policy_structural_beta_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.structural_beta_name("latent_var", "income")

    assert result == "struct_latent_var_income"


def test_default_naming_policy_structural_beta_name_uses_structural_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DefaultNamingPolicy()
    calls: list[str] = []

    def fake_structural_prefix(self: DefaultNamingPolicy, latent_name: str) -> str:
        calls.append(latent_name)
        return "PREFIX"

    monkeypatch.setattr(
        DefaultNamingPolicy, "structural_prefix", fake_structural_prefix
    )

    result = policy.structural_beta_name("LV", "time")

    assert result == "PREFIX_time"
    assert calls == ["LV"]


def test_default_naming_policy_structural_sigma_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.structural_sigma_name("latent_var")

    assert result == "struct_latent_var_sigma"


def test_default_naming_policy_structural_sigma_name_uses_structural_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DefaultNamingPolicy()
    calls: list[str] = []

    def fake_structural_prefix(self: DefaultNamingPolicy, latent_name: str) -> str:
        calls.append(latent_name)
        return "PREFIX"

    monkeypatch.setattr(
        DefaultNamingPolicy, "structural_prefix", fake_structural_prefix
    )

    result = policy.structural_sigma_name("LV")

    assert result == "PREFIX_sigma"
    assert calls == ["LV"]


def test_default_naming_policy_structural_draw_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.structural_draw_name("latent_var")

    assert result == "struct_latent_var_draws"


def test_default_naming_policy_structural_draw_name_uses_structural_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DefaultNamingPolicy()
    calls: list[str] = []

    def fake_structural_prefix(self: DefaultNamingPolicy, latent_name: str) -> str:
        calls.append(latent_name)
        return "PREFIX"

    monkeypatch.setattr(
        DefaultNamingPolicy, "structural_prefix", fake_structural_prefix
    )

    result = policy.structural_draw_name("LV")

    assert result == "PREFIX_draws"
    assert calls == ["LV"]


def test_default_naming_policy_measurement_prefix() -> None:
    policy = DefaultNamingPolicy()

    result = policy.measurement_prefix("indicator_1")

    assert result == "measurement_indicator_1"


def test_default_naming_policy_measurement_intercept_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.measurement_intercept_name("indicator_1")

    assert result == "measurement_intercept_indicator_1"


def test_default_naming_policy_measurement_loading_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.measurement_loading_name("LV1", "indicator_1")

    assert result == "measurement_coefficient_LV1_indicator_1"


def test_default_naming_policy_measurement_sigma_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.measurement_sigma_name("indicator_1")

    assert result == "measurement_indicator_1_sigma"


def test_default_naming_policy_measurement_sigma_name_uses_measurement_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DefaultNamingPolicy()
    calls: list[str] = []

    def fake_measurement_prefix(self: DefaultNamingPolicy, indicator_name: str) -> str:
        calls.append(indicator_name)
        return "MEAS"

    monkeypatch.setattr(
        DefaultNamingPolicy, "measurement_prefix", fake_measurement_prefix
    )

    result = policy.measurement_sigma_name("IND")

    assert result == "MEAS_sigma"
    assert calls == ["IND"]


def test_default_naming_policy_threshold_tau1_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.threshold_tau1_name("likert5")

    assert result == "likert5_tau_1"


def test_default_naming_policy_threshold_delta_name() -> None:
    policy = DefaultNamingPolicy()

    result = policy.threshold_delta_name("likert5", 3)

    assert result == "likert5_delta_3"


def test_default_naming_policy_handles_empty_strings() -> None:
    policy = DefaultNamingPolicy()

    assert policy.structural_prefix("") == "struct_"
    assert policy.structural_beta_name("", "") == "struct__"
    assert policy.structural_sigma_name("") == "struct__sigma"
    assert policy.structural_draw_name("") == "struct__draws"
    assert policy.measurement_prefix("") == "measurement_"
    assert policy.measurement_intercept_name("") == "measurement_intercept_"
    assert policy.measurement_loading_name("", "") == "measurement_coefficient__"
    assert policy.measurement_sigma_name("") == "measurement__sigma"
    assert policy.threshold_tau1_name("") == "_tau_1"
    assert policy.threshold_delta_name("", 0) == "_delta_0"


def test_default_naming_policy_preserves_special_characters() -> None:
    policy = DefaultNamingPolicy()

    assert policy.structural_prefix("LV-a b") == "struct_LV-a b"
    assert policy.structural_beta_name("LV-a b", "x/y") == "struct_LV-a b_x/y"
    assert policy.measurement_prefix("ind:1") == "measurement_ind:1"
    assert (
        policy.measurement_loading_name("LV-a b", "ind:1")
        == "measurement_coefficient_LV-a b_ind:1"
    )
    assert policy.threshold_tau1_name("type-α") == "type-α_tau_1"
    assert policy.threshold_delta_name("type-α", 12) == "type-α_delta_12"


def test_default_naming_policy_structurally_matches_protocol() -> None:
    policy: NamingPolicy = DefaultNamingPolicy()

    assert policy.structural_prefix("LV") == "struct_LV"
    assert policy.structural_beta_name("LV", "x") == "struct_LV_x"
    assert policy.structural_sigma_name("LV") == "struct_LV_sigma"
    assert policy.structural_draw_name("LV") == "struct_LV_draws"
    assert policy.measurement_prefix("ind") == "measurement_ind"
    assert policy.measurement_intercept_name("ind") == "measurement_intercept_ind"
    assert (
        policy.measurement_loading_name("LV", "ind") == "measurement_coefficient_LV_ind"
    )
    assert policy.measurement_sigma_name("ind") == "measurement_ind_sigma"
    assert policy.threshold_tau1_name("type") == "type_tau_1"
    assert policy.threshold_delta_name("type", 2) == "type_delta_2"
