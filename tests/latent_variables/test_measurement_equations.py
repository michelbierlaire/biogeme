import pytest
from biogeme.expressions import Beta, Expression, Numeric

# Adjust this import path if your module location differs
from biogeme.latent_variables.measurement_equations import (
    _ordered_model,
    log_measurement_equations_jax,
    measurement_equations_jax,
)


class DummyNormalization:
    def __init__(self, indicator: str, coefficient: float):
        self.indicator = indicator
        self.coefficient = coefficient


class DummyLatentVariable:
    """Minimal LatentVariable substitute to satisfy measurement_equations._ordered_model."""

    def __init__(
        self, name: str, indicators: list[str], normalization: DummyNormalization
    ):
        self.name = name
        self.indicators = indicators
        self.normalization = normalization
        self.draw_type_jax: str | None = None
        self._struct_calls: list[str] = []

    @property
    def structural_equation_jax(self) -> Expression:
        # _ordered_model sets draw_type_jax before using this property; record it.
        assert self.draw_type_jax is not None
        self._struct_calls.append(self.draw_type_jax)
        # Return a deterministic expression
        return Numeric(0.5)


class DummyLikertIndicator:
    """Minimal LikertIndicator substitute."""

    def __init__(
        self,
        name: str,
        type_label: str,
        intercept_value: float,
        scale_value: float,
    ):
        self.name = name
        self.type = type_label
        self._intercept_value = intercept_value
        self._scale_value = scale_value
        self.coeff_calls: list[str] = []

    @property
    def intercept_parameter(self) -> Expression:
        # Use Beta to ensure Expression behavior beyond Numeric
        return Beta(
            f"measurement_intercept_{self.name}", self._intercept_value, None, None, 0
        )

    @property
    def scale_parameter(self) -> Expression:
        # positive scale; used as sigma_star unless indicator is normalization anchor
        return Numeric(self._scale_value)

    def get_lv_coefficient_parameter(self, latent_variable_name: str) -> Expression:
        self.coeff_calls.append(latent_variable_name)
        return Beta(
            f"measurement_coefficient_{latent_variable_name}_{self.name}",
            1.0,
            None,
            None,
            0,
        )


class DummyLikertType:
    """Minimal LikertType substitute."""

    def __init__(
        self,
        type_label: str,
        categories: list[int],
        neutral_labels: list[int],
        scale_normalization: str,
    ):
        self.type = type_label
        self.categories = categories
        self.neutral_labels = neutral_labels
        self.scale_normalization = scale_normalization
        self.threshold_calls = 0

    def get_thresholds(self) -> list[Expression]:
        self.threshold_calls += 1
        # Two cut-points
        return [Numeric(-1.0), Numeric(1.0)]


def _cutpoints_of(expr: Expression) -> list[Expression] | None:
    """
    Best-effort extraction of OrderedProbit cutpoints.
    This stays robust if internals change: we try attributes first, else fall back to None.
    """
    for attr in ("cutpoints", "_cutpoints"):
        if hasattr(expr, attr):
            val = getattr(expr, attr)
            if isinstance(val, list):
                return val
    return None


def test_ordered_model_happy_path_and_normalization_effect_on_cutpoints():
    # LV1 anchors on I1
    lv1 = DummyLatentVariable(
        name="LV1",
        indicators=["I1", "I2"],
        normalization=DummyNormalization(indicator="I1", coefficient=2.0),
    )

    ind1 = DummyLikertIndicator(
        name="I1", type_label="T", intercept_value=0.1, scale_value=10.0
    )
    ind2 = DummyLikertIndicator(
        name="I2", type_label="T", intercept_value=-0.2, scale_value=2.0
    )
    lt = DummyLikertType(
        type_label="T",
        categories=[1, 2, 3],
        neutral_labels=[2],
        scale_normalization="I1",
    )

    ordered_ll = _ordered_model(
        latent_variables=[lv1],
        likert_indicators=[ind1, ind2],
        likert_types=[lt],
        draw_type="NORMAL",
    )

    # All indicators appear
    assert set(ordered_ll.keys()) == {"I1", "I2"}
    assert all(isinstance(v, Expression) for v in ordered_ll.values())

    # draw_type_jax is set and structural_equation_jax used for each indicator
    assert lv1.draw_type_jax == "NORMAL"
    assert lv1._struct_calls == ["NORMAL", "NORMAL"]

    # thresholds used once per indicator
    assert lt.threshold_calls == 2

    # Best-effort check that normalization indicator has sigma_star = 1.0:
    # its cutpoints should be unchanged (not divided by 10.0),
    # while the non-anchor indicator cutpoints should be scaled by its sigma_star (2.0).
    cp1 = _cutpoints_of(ordered_ll["I1"])
    cp2 = _cutpoints_of(ordered_ll["I2"])
    if cp1 is not None and cp2 is not None:
        # anchor: sigma_star forced to 1.0 => [-1, 1]
        assert pytest.approx(cp1[0].get_value(), abs=1e-12) == -1.0
        assert pytest.approx(cp1[1].get_value(), abs=1e-12) == 1.0
        # non-anchor: sigma_star=2.0 => [-0.5, 0.5]
        assert pytest.approx(cp2[0].get_value(), abs=1e-12) == -0.5
        assert pytest.approx(cp2[1].get_value(), abs=1e-12) == 0.5


def test_ordered_model_raises_for_unknown_type():
    lv1 = DummyLatentVariable(
        name="LV1",
        indicators=["I1"],
        normalization=DummyNormalization(indicator="I1", coefficient=1.0),
    )
    ind1 = DummyLikertIndicator(
        name="I1", type_label="UNKNOWN", intercept_value=0.0, scale_value=3.0
    )
    lt = DummyLikertType(
        type_label="T", categories=[1, 2], neutral_labels=[], scale_normalization="I1"
    )

    with pytest.raises(ValueError, match=r"Unknown type for indicator I1: UNKNOWN"):
        _ordered_model(
            latent_variables=[lv1],
            likert_indicators=[ind1],
            likert_types=[lt],
            draw_type="NORMAL",
        )


def test_ordered_model_raises_if_indicator_sets_inconsistent():
    # LV declares I1 and I2, but likert_indicators only provides I1
    lv1 = DummyLatentVariable(
        name="LV1",
        indicators=["I1", "I2"],
        normalization=DummyNormalization(indicator="I1", coefficient=1.0),
    )
    ind1 = DummyLikertIndicator(
        name="I1", type_label="T", intercept_value=0.0, scale_value=2.0
    )
    lt = DummyLikertType(
        type_label="T",
        categories=[1, 2, 3],
        neutral_labels=[2],
        scale_normalization="I1",
    )

    with pytest.raises(Exception):
        _ordered_model(
            latent_variables=[lv1],
            likert_indicators=[ind1],
            likert_types=[lt],
            draw_type="NORMAL",
        )


def test_measurement_equations_jax_runs_through_multipleproduct():
    lv1 = DummyLatentVariable(
        name="LV1",
        indicators=["I1"],
        normalization=DummyNormalization(indicator="I1", coefficient=1.0),
    )
    ind1 = DummyLikertIndicator(
        name="I1", type_label="T", intercept_value=0.0, scale_value=2.0
    )
    lt = DummyLikertType(
        type_label="T",
        categories=[1, 2, 3],
        neutral_labels=[2],
        scale_normalization="I1",
    )

    expr = measurement_equations_jax(
        latent_variables=[lv1],
        likert_indicators=[ind1],
        likert_types=[lt],
        draw_type="NORMAL",
    )

    assert isinstance(expr, Expression)


def test_log_measurement_equations_jax_runs_through_multiplesum_and_log():
    lv1 = DummyLatentVariable(
        name="LV1",
        indicators=["I1"],
        normalization=DummyNormalization(indicator="I1", coefficient=1.0),
    )
    ind1 = DummyLikertIndicator(
        name="I1", type_label="T", intercept_value=0.0, scale_value=2.0
    )
    lt = DummyLikertType(
        type_label="T",
        categories=[1, 2, 3],
        neutral_labels=[2],
        scale_normalization="I1",
    )

    expr = log_measurement_equations_jax(
        latent_variables=[lv1],
        likert_indicators=[ind1],
        likert_types=[lt],
        draw_type="NORMAL",
    )

    assert isinstance(expr, Expression)
