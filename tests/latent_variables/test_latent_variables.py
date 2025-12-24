import pytest
from biogeme.expressions import Expression, Numeric
from biogeme.latent_variables import LatentVariable, Normalization


class DummyStructuralEquation:
    """Minimal stand-in for StructuralEquation for unit tests."""

    def __init__(self):
        self.sigma_factory = None
        self.calls = []

    def expression(self, *, draw_type: str) -> Expression:
        # record call and return a valid Biogeme Expression
        self.calls.append({"draw_type": draw_type})
        return Numeric(123.0)


class DummySigmaFactory:
    """Stand-in for SigmaFactory (only identity matters in this module)."""

    def __call__(self, name: str):  # signature irrelevant here; stored & forwarded only
        return Numeric(1.0)


def test_normalization_dataclass_stores_fields():
    n = Normalization(indicator="Ind01", coefficient=1.5)
    assert n.indicator == "Ind01"
    assert n.coefficient == 1.5


def test_latent_variable_dataclass_stores_fields():
    se = DummyStructuralEquation()
    norm = Normalization(indicator="Ind01", coefficient=1.0)

    lv = LatentVariable(
        name="LV1",
        structural_equation=se,
        indicators=["Ind01", "Ind02"],
        normalization=norm,
        draw_type_jax="NORMAL",
        sigma_factory=DummySigmaFactory(),
    )

    assert lv.name == "LV1"
    assert lv.structural_equation is se
    assert list(lv.indicators) == ["Ind01", "Ind02"]
    assert lv.normalization is norm
    assert lv.draw_type_jax == "NORMAL"
    assert lv.sigma_factory is not None


def test_structural_equation_jax_raises_if_draw_type_missing():
    se = DummyStructuralEquation()
    norm = Normalization(indicator="Ind01", coefficient=1.0)

    lv = LatentVariable(
        name="LV1",
        structural_equation=se,
        indicators=["Ind01"],
        normalization=norm,
        draw_type_jax=None,
        sigma_factory=DummySigmaFactory(),
    )

    with pytest.raises(ValueError, match="The type of draws has not been defined\\."):
        _ = lv.structural_equation_jax


def test_structural_equation_jax_raises_if_sigma_factory_missing():
    se = DummyStructuralEquation()
    norm = Normalization(indicator="Ind01", coefficient=1.0)

    lv = LatentVariable(
        name="LV1",
        structural_equation=se,
        indicators=["Ind01"],
        normalization=norm,
        draw_type_jax="NORMAL",
        sigma_factory=None,
    )

    with pytest.raises(ValueError, match="The sigma factory has not been defined\\."):
        _ = lv.structural_equation_jax


def test_structural_equation_jax_injects_sigma_factory_and_calls_expression():
    se = DummyStructuralEquation()
    norm = Normalization(indicator="Ind01", coefficient=1.0)
    sf = DummySigmaFactory()

    lv = LatentVariable(
        name="LV1",
        structural_equation=se,
        indicators=("Ind01", "Ind02"),  # exercise Iterable[str] with a tuple
        normalization=norm,
        draw_type_jax="NORMAL",
        sigma_factory=sf,
    )

    out = lv.structural_equation_jax

    # property returns Expression
    assert isinstance(out, Expression)
    assert out.get_value() == pytest.approx(123.0)

    # sigma_factory is injected into structural_equation
    assert se.sigma_factory is sf

    # expression called exactly once with expected draw_type
    assert se.calls == [{"draw_type": "NORMAL"}]
