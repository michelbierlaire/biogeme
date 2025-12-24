import pytest

from biogeme.expressions import Expression, Numeric

# Adjust if your project structure differs
from biogeme.latent_variables.structural_equation import StructuralEquation


class SpySigmaFactory:
    """SigmaFactory spy that records calls and returns a deterministic Expression."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, prefix: str) -> Expression:
        self.calls.append({"prefix": prefix})
        # return a deterministic positive-ish scale
        return Numeric(2.0)


def test_prefix_uses_struct_name():
    se = StructuralEquation(name="LV1", explanatory_variables=["x1", "x2"])
    assert se.prefix == "struct_LV1"


def test_get_expression_deterministic_part_builds_betas_for_each_variable_and_is_expression():
    se = StructuralEquation(name="LV1", explanatory_variables=["age", "income"])
    expr = se.get_expression_deterministic_part()

    assert isinstance(expr, Expression)

    # Robust check without relying on internal Expression node types:
    # ensure the Beta names appear in the string representation
    s = str(expr)
    assert "struct_LV1_age" in s
    assert "struct_LV1_income" in s

    # Ensure both Variable names appear too
    assert "age" in s
    assert "income" in s


def test_expression_raises_if_no_scale_and_sigma_factory_missing():
    se = StructuralEquation(
        name="LV1", explanatory_variables=["x1"], sigma_factory=None
    )

    with pytest.raises(ValueError, match="Sigma factory is undefined"):
        _ = se.expression(draw_type="NORMAL")


def test_expression_uses_sigma_factory_when_scale_parameter_not_provided():
    sf = SpySigmaFactory()
    se = StructuralEquation(
        name="LV1", explanatory_variables=["x1", "x2"], sigma_factory=sf
    )

    expr = se.expression(draw_type="NORMAL")
    assert isinstance(expr, Expression)

    # sigma_factory called with prefix == se.prefix
    assert sf.calls == [{"prefix": "struct_LV1"}]

    # Draws node should reference name and draw_type in the expression string
    s = str(expr)
    assert "struct_LV1_draws" in s
    assert "NORMAL" in s


def test_expression_does_not_call_sigma_factory_when_scale_parameter_is_given():
    sf = SpySigmaFactory()
    se = StructuralEquation(name="LV1", explanatory_variables=["x1"], sigma_factory=sf)

    # Provide explicit scale parameter so sigma_factory is not used
    scale = Numeric(5.0)
    expr = se.expression(draw_type="NORMAL", scale_parameter=scale)

    assert isinstance(expr, Expression)
    assert sf.calls == []  # no call

    s = str(expr)
    assert "struct_LV1_draws" in s
    assert "NORMAL" in s


def test_expression_works_with_empty_explanatory_variables():
    # Edge case: no explanatory variables -> deterministic part should still be an Expression
    sf = SpySigmaFactory()
    se = StructuralEquation(name="LV_EMPTY", explanatory_variables=[], sigma_factory=sf)

    expr = se.expression(draw_type="NORMAL")
    assert isinstance(expr, Expression)
    assert sf.calls == [{"prefix": "struct_LV_EMPTY"}]

    s = str(expr)
    assert "struct_LV_EMPTY_draws" in s


def test_expression_accepts_different_draw_types_and_reflects_in_expression_string():
    sf = SpySigmaFactory()
    se = StructuralEquation(name="LV1", explanatory_variables=["x1"], sigma_factory=sf)

    expr_a = se.expression(draw_type="A")
    expr_b = se.expression(draw_type="B")

    assert isinstance(expr_a, Expression)
    assert isinstance(expr_b, Expression)

    sa = str(expr_a)
    sb = str(expr_b)
    assert "A" in sa
    assert "B" in sb
