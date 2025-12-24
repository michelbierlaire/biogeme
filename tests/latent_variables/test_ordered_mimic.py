import logging

import pytest
from biogeme.expressions import Expression, Numeric
# If your module path differs, adjust these two imports.
from biogeme.latent_variables import ordered_mimic as om
from biogeme.latent_variables.ordered_mimic import EstimationMode, OrderedMimic


class StubLikertIndicator:
    def __init__(
        self, name: str, *, sigma_factory=None, positive_parameter_factory=None
    ):
        self.name = name
        self.sigma_factory = sigma_factory
        self.positive_parameter_factory = positive_parameter_factory


class StubLikertType:
    def __init__(
        self, type_label: str, *, sigma_factory=None, positive_parameter_factory=None
    ):
        self.type = type_label
        self.sigma_factory = sigma_factory
        self.positive_parameter_factory = positive_parameter_factory


class StubLatentVariable:
    def __init__(
        self,
        name: str,
        indicators,
        *,
        draw_type_jax=None,
        sigma_factory=None,
    ):
        self.name = name
        self.indicators = indicators
        self.draw_type_jax = draw_type_jax
        self.sigma_factory = sigma_factory


def test_post_init_sets_defaults_ml(monkeypatch):
    calls = {"sigma": [], "positive": []}

    def fake_make_sigma_factory(*, use_log: bool):
        calls["sigma"].append(use_log)
        return f"sigma(use_log={use_log})"

    def fake_make_positive_parameter_factory(*, use_log: bool):
        calls["positive"].append(use_log)
        return f"pos(use_log={use_log})"

    monkeypatch.setattr(om, "make_sigma_factory", fake_make_sigma_factory)
    monkeypatch.setattr(
        om, "make_positive_parameter_factory", fake_make_positive_parameter_factory
    )

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")

    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type=None,
        sigma_factory=None,
    )

    assert m.draw_type == "NORMAL_MLHS_ANTI"
    assert m.sigma_factory == "sigma(use_log=True)"
    assert calls["sigma"] == [True]
    assert calls["positive"] == [True]

    # injected factories
    assert ind.sigma_factory == m.sigma_factory
    assert ind.positive_parameter_factory == m._positive_factory
    assert lt.sigma_factory == m.sigma_factory
    assert lt.positive_parameter_factory == m._positive_factory


def test_post_init_sets_defaults_bayesian(monkeypatch):
    calls = {"sigma": [], "positive": []}

    def fake_make_sigma_factory(*, use_log: bool):
        calls["sigma"].append(use_log)
        return f"sigma(use_log={use_log})"

    def fake_make_positive_parameter_factory(*, use_log: bool):
        calls["positive"].append(use_log)
        return f"pos(use_log={use_log})"

    monkeypatch.setattr(om, "make_sigma_factory", fake_make_sigma_factory)
    monkeypatch.setattr(
        om, "make_positive_parameter_factory", fake_make_positive_parameter_factory
    )

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")

    m = OrderedMimic(
        estimation_mode=EstimationMode.BAYESIAN,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type=None,
        sigma_factory=None,
    )

    assert m.draw_type == "Normal"
    assert m.sigma_factory == "sigma(use_log=False)"
    assert calls["sigma"] == [False]
    assert calls["positive"] == [False]


def test_register_likert_indicators_raises_if_missing_factories(monkeypatch):
    # create a valid instance, then force state to hit error branches in private method
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
    )

    m.sigma_factory = None
    with pytest.raises(ValueError, match="Sigma factory is undefined"):
        m._register_likert_indicators()

    m.sigma_factory = "sigma"
    m._positive_factory = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Positive parameter factory is undefined"):
        m._register_likert_indicators()


def test_register_likert_types_raises_if_missing_factories(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
    )

    m.sigma_factory = None
    with pytest.raises(ValueError, match="Sigma factory is undefined"):
        m._register_likert_types()

    m.sigma_factory = "sigma"
    m._positive_factory = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Positive parameter factory is undefined"):
        m._register_likert_types()


def test_add_latent_variable_injects_defaults_and_converts_indicators(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")

    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="DT",
        sigma_factory="SIG",
    )

    lv = StubLatentVariable(
        name="LV1", indicators=["I1", "I2"], draw_type_jax=None, sigma_factory=None
    )
    out = m.add_latent_variable(lv)

    assert out is lv
    assert lv.draw_type_jax == "DT"
    assert lv.sigma_factory == "SIG"
    assert isinstance(lv.indicators, set)
    assert lv.indicators == {"I1", "I2"}


def test_add_latent_variable_raises_on_duplicate(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="DT",
        sigma_factory="SIG",
    )

    m.add_latent_variable(StubLatentVariable(name="LV1", indicators=["I1"]))
    with pytest.raises(ValueError, match="registered twice"):
        m.add_latent_variable(StubLatentVariable(name="LV1", indicators=["I1"]))


def test_add_latent_variable_raises_if_sigma_factory_missing(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        sigma_factory=None,
    )

    # force missing sigma_factory and hit the guard
    m.sigma_factory = None
    with pytest.raises(ValueError, match="Sigma factory is undefined"):
        m.add_latent_variable(StubLatentVariable(name="LV1", indicators=["I1"]))


def test_add_latent_variable_raises_if_indicators_not_iterable(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="DT",
        sigma_factory="SIG",
    )

    class NotIterable:
        pass

    with pytest.raises(ValueError, match="invalid indicators"):
        m.add_latent_variable(StubLatentVariable(name="LV1", indicators=NotIterable()))


def test_accessors_return_and_raise_keyerror(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")

    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="DT",
        sigma_factory="SIG",
    )

    lv = StubLatentVariable(name="LV1", indicators=["I1"])
    m.add_latent_variable(lv)

    assert m.get_likert_indicator("I1") is ind
    assert m.get_likert_type("T") is lt
    assert m.get_latent_variable("LV1") is lv
    assert m.latent_variables == [lv]

    with pytest.raises(KeyError):
        _ = m.get_likert_indicator("missing")
    with pytest.raises(KeyError):
        _ = m.get_likert_type("missing")
    with pytest.raises(KeyError):
        _ = m.get_latent_variable("missing")


def test_measurement_equations_raises_when_empty(monkeypatch):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[],
        likert_types=[],
        draw_type="DT",
        sigma_factory="SIG",
    )

    with pytest.raises(ValueError, match="No latent variables"):
        _ = m.measurement_equations()

    # Add LV but still no indicators registered
    m._latent_variables["LV1"] = StubLatentVariable("LV1", ["I1"])
    with pytest.raises(ValueError, match="No Likert indicators"):
        _ = m.measurement_equations()


def test_measurement_equations_warns_in_bayesian_and_passes_override_draw_type(
    monkeypatch, caplog
):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    called = {}

    def fake_measurement_equations_jax(
        *, latent_variables, likert_indicators, likert_types, draw_type
    ):
        called["args"] = {
            "latent_variables": latent_variables,
            "likert_indicators": likert_indicators,
            "likert_types": likert_types,
            "draw_type": draw_type,
        }
        return Numeric(7.0)

    monkeypatch.setattr(om, "measurement_equations_jax", fake_measurement_equations_jax)

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.BAYESIAN,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="BASE",
        sigma_factory="SIG",
    )
    m.add_latent_variable(StubLatentVariable("LV1", ["I1"]))

    caplog.set_level(logging.WARNING)
    expr = m.measurement_equations(draw_type="OVERRIDE")

    assert isinstance(expr, Expression)
    assert expr.get_value() == pytest.approx(7.0)

    assert any(
        "usually used for maximum likelihood" in rec.message for rec in caplog.records
    )
    assert called["args"]["draw_type"] == "OVERRIDE"


def test_log_measurement_equations_warns_in_ml_and_uses_fallback_normal(
    monkeypatch, caplog
):
    monkeypatch.setattr(om, "make_sigma_factory", lambda *, use_log: "sigma")
    monkeypatch.setattr(om, "make_positive_parameter_factory", lambda *, use_log: "pos")

    called = {}

    def fake_log_measurement_equations_jax(
        *, latent_variables, likert_indicators, likert_types, draw_type
    ):
        called["args"] = {
            "latent_variables": latent_variables,
            "likert_indicators": likert_indicators,
            "likert_types": likert_types,
            "draw_type": draw_type,
        }
        return Numeric(3.0)

    monkeypatch.setattr(
        om, "log_measurement_equations_jax", fake_log_measurement_equations_jax
    )

    ind = StubLikertIndicator("I1")
    lt = StubLikertType("T")
    m = OrderedMimic(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        likert_indicators=[ind],
        likert_types=[lt],
        draw_type="BASE",
        sigma_factory="SIG",
    )
    m.add_latent_variable(StubLatentVariable("LV1", ["I1"]))

    # Force the internal fallback branch `( ... ) or "NORMAL"`
    m.draw_type = None

    caplog.set_level(logging.WARNING)
    expr = m.log_measurement_equations(draw_type=None)

    assert isinstance(expr, Expression)
    assert expr.get_value() == pytest.approx(3.0)

    assert any("usually used for Bayesian" in rec.message for rec in caplog.records)
    assert called["args"]["draw_type"] == "NORMAL"
