import pytest
from biogeme.expressions import Beta, Expression, Numeric
from biogeme.latent_variables.likert_indicators import LikertIndicator, LikertType


class SpyPositiveFactory:
    """Spy factory matching PositiveParameterFactory protocol (callable)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, name: str, prefix: str, value: float) -> Expression:
        self.calls.append({"name": name, "prefix": prefix, "value": value})
        # Return a Numeric so sums/negations work and are deterministic
        return Numeric(value)


class SpySigmaFactory:
    """Spy factory matching SigmaFactory protocol (callable)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, prefix: str) -> Expression:
        self.calls.append({"prefix": prefix})
        # Deterministic positive-like parameter placeholder
        return Beta(f"{prefix}_sigma", 1.0, None, None, 0)


def _num(x: Expression) -> float:
    """Extract float from Numeric in a robust way for these tests."""
    assert isinstance(x, Numeric)
    # Numeric stores value in `.value`
    return float(x.value)


def test_get_thresholds_raises_when_positive_factory_missing():
    lt = LikertType(
        type="T",
        symmetric=True,
        categories=[-1, 0, 1],
        neutral_labels=[0],
        scale_normalization="I1",
        positive_parameter_factory=None,
    )
    with pytest.raises(ValueError, match="Positive parameter factory is undefined"):
        lt.get_thresholds()


def test_get_thresholds_raises_for_too_few_categories():
    pf = SpyPositiveFactory()
    lt = LikertType(
        type="T",
        symmetric=True,
        categories=[0],  # <2
        neutral_labels=[0],
        scale_normalization="I1",
        positive_parameter_factory=pf,
    )
    with pytest.raises(
        ValueError, match="Likert scale must have at least 2 categories"
    ):
        lt.get_thresholds()


@pytest.mark.parametrize(
    "categories, expected_n_tau, expect_zero",
    [
        ([-2, -1, 0, 1, 2], 4, False),  # K=5 => n_tau=4 even => no 0 inserted
        ([-2, -1, 1, 2], 3, True),  # K=4 => n_tau=3 odd  => insert 0
    ],
)
def test_get_thresholds_symmetric_builds_expected_structure(
    categories, expected_n_tau, expect_zero
):
    pf = SpyPositiveFactory()
    lt = LikertType(
        type="LIK",
        symmetric=True,
        categories=categories,
        neutral_labels=[0],
        scale_normalization="I1",
        positive_parameter_factory=pf,
    )

    thresholds = lt.get_thresholds()

    # length check + branch for internal error guard not triggered
    assert len(thresholds) == expected_n_tau

    # Check symmetry by summing symmetric expressions and evaluating the result
    # (Numeric / Expression-based, robust to UnaryMinus and composite expressions)
    nonzero = [t for t in thresholds if not (isinstance(t, Numeric) and _num(t) == 0.0)]
    half = len(nonzero) // 2
    left = nonzero[:half]
    right = nonzero[half:]

    # For symmetry, a + b should evaluate to zero for mirrored pairs
    for a, b in zip(left, reversed(right), strict=True):
        s = a + b
        val = s.get_value()
        assert pytest.approx(val, abs=1e-12) == 0.0

    # Ensure we exercised delta generation: number of deltas = n_tau // 2
    assert len(pf.calls) == expected_n_tau // 2
    for k, call in enumerate(pf.calls):
        assert call["name"] == f"delta_{k}"
        assert call["prefix"] == "LIK"
        # init scheme line executed; just sanity-check it's float-like
        assert isinstance(call["value"], float)


def test_get_thresholds_monotone_with_fixed_first_cutpoint():
    pf = SpyPositiveFactory()
    lt = LikertType(
        type="M",
        symmetric=False,
        categories=[1, 2, 3, 4, 5],  # K=5 => n_tau=4
        neutral_labels=[],
        scale_normalization="I1",
        positive_parameter_factory=pf,
        fix_first_cut_point_for_non_symmetric_thresholds=-1.5,
    )

    thresholds = lt.get_thresholds()
    assert len(thresholds) == 4

    # first cut-point should evaluate to the fixed value
    assert isinstance(thresholds[0], Expression)
    assert pytest.approx(thresholds[0].get_value(), abs=1e-12) == -1.5

    # thresholds are Expressions; verify they evaluate to a strictly increasing sequence
    vals = [t.get_value() for t in thresholds]
    assert all(isinstance(v, (int, float)) for v in vals)
    assert all(a < b for a, b in zip(vals, vals[1:]))

    # delta calls for k in [2..4] => delta_{k-1} -> delta_1..delta_3
    assert [c["name"] for c in pf.calls] == ["delta_1", "delta_2", "delta_3"]
    assert all(c["prefix"] == "M" for c in pf.calls)


def test_get_thresholds_monotone_with_free_first_cutpoint_is_beta_then_numeric():
    pf = SpyPositiveFactory()
    lt = LikertType(
        type="FREE",
        symmetric=False,
        categories=[0, 1, 2, 3],  # K=4 => n_tau=3
        neutral_labels=[],
        scale_normalization="I1",
        positive_parameter_factory=pf,
        fix_first_cut_point_for_non_symmetric_thresholds=None,
    )

    thresholds = lt.get_thresholds()
    assert len(thresholds) == 3

    # First threshold is unconstrained Beta per code
    assert isinstance(thresholds[0], Beta)
    assert thresholds[0].name == "FREE_tau_1"

    # Subsequent thresholds: Beta + Numeric -> Expression; in Biogeme it will be an Expression.
    # We can still assert factories got called and the loop executed.
    assert [c["name"] for c in pf.calls] == ["delta_1", "delta_2"]
    assert all(c["prefix"] == "FREE" for c in pf.calls)


def test_internal_length_guard_raises_runtime_error(monkeypatch):
    pf = SpyPositiveFactory()
    lt = LikertType(
        type="X",
        symmetric=True,
        categories=[-1, 0, 1],  # K=3 => n_tau=2
        neutral_labels=[0],
        scale_normalization="I1",
        positive_parameter_factory=pf,
    )

    # Force _build_symmetric to return wrong length to hit RuntimeError line
    def bad_build(*args, **kwargs):
        return [Numeric(0.0)]  # should be 2

    monkeypatch.setattr(lt, "_build_symmetric", bad_build)
    with pytest.raises(RuntimeError, match="expected 2 cutpoints"):
        lt.get_thresholds()


def test_likert_indicator_intercept_name_and_parameter():
    ind = LikertIndicator(name="Envir01", statement="...", type="env")
    assert ind.intercept_parameter_name == "measurement_intercept_Envir01"

    beta = ind.intercept_parameter
    assert isinstance(beta, Beta)
    assert beta.name == "measurement_intercept_Envir01"


def test_likert_indicator_scale_parameter_raises_without_sigma_factory():
    ind = LikertIndicator(name="X", statement="...", type="t", sigma_factory=None)
    with pytest.raises(ValueError, match="Sigma factory is undefined"):
        _ = ind.scale_parameter


def test_likert_indicator_scale_parameter_calls_factory_with_expected_prefix():
    sf = SpySigmaFactory()
    ind = LikertIndicator(name="NbCar", statement="...", type="cars", sigma_factory=sf)

    expr = ind.scale_parameter
    assert isinstance(expr, Beta)
    assert expr.name == "measurement_NbCar_sigma"

    assert sf.calls == [{"prefix": "measurement_NbCar"}]


def test_lv_coefficient_name_and_parameter():
    ind = LikertIndicator(name="Mobil03", statement="...", type="mobil")

    pname = ind.get_lv_coefficient_parameter_name("car_centric_attitude")
    assert pname == "measurement_coefficient_car_centric_attitude_Mobil03"

    beta = ind.get_lv_coefficient_parameter("car_centric_attitude")
    assert isinstance(beta, Beta)
    assert beta.name == pname
