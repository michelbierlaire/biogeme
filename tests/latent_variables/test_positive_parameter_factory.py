import pytest

from biogeme.expressions import Beta, Expression
from biogeme.floating_point import SMALL_POSITIVE

from biogeme.latent_variables import (
    make_positive_parameter_factory,
    make_sigma_factory,
)


def _assert_is_expression(x: object) -> None:
    assert isinstance(x, Expression), f"Expected Expression, got {type(x)!r}"


def test_make_positive_parameter_factory_use_log_true_returns_exp_of_beta_and_value_is_positive():
    factory = make_positive_parameter_factory(use_log=True)

    expr = factory(name="delta_0", prefix="TYPEA", value=0.0)
    _assert_is_expression(expr)

    # Naming convention for log-space parameter
    # Expression is exp(Beta(...)), so the Beta name must match.
    s = str(expr)
    assert "TYPEA_delta_0_log" in s

    # Should evaluate to exp(0)=1
    assert pytest.approx(expr.get_value(), abs=1e-12) == 1.0


def test_make_positive_parameter_factory_use_log_false_returns_bounded_beta():
    factory = make_positive_parameter_factory(use_log=False)

    expr = factory(name="delta_1", prefix="TYPEB", value=2.5)
    _assert_is_expression(expr)

    # In direct mode, it is a Beta
    assert isinstance(expr, Beta)
    assert expr.name == "TYPEB_delta_1"

    # Lower bound should be SMALL_POSITIVE, upper bound None
    assert expr.lower_bound == SMALL_POSITIVE
    assert expr.upper_bound is None

    # Should evaluate to the initial value
    assert pytest.approx(expr.get_value(), abs=1e-12) == 2.5
    assert expr.get_value() > 0.0


def test_make_positive_parameter_factory_different_prefixes_create_distinct_parameter_names():
    factory = make_positive_parameter_factory(use_log=False)

    e1 = factory(name="sigma", prefix="A", value=1.0)
    e2 = factory(name="sigma", prefix="B", value=1.0)

    assert isinstance(e1, Beta) and isinstance(e2, Beta)
    assert e1.name == "A_sigma"
    assert e2.name == "B_sigma"
    assert e1.name != e2.name


def test_make_sigma_factory_use_log_true_builds_sigma_log_and_exp_value_matches():
    sigma_factory = make_sigma_factory(use_log=True)

    expr = sigma_factory(prefix="meas_I1")
    _assert_is_expression(expr)

    # Should be exp(Beta("meas_I1_sigma_log", -1, ...))
    s = str(expr)
    assert "meas_I1_sigma_log" in s

    # exp(-1) = 0.367879...
    assert pytest.approx(expr.get_value(), rel=1e-12) == pytest.approx(
        2.718281828459045 ** (-1), rel=1e-12
    )
    assert expr.get_value() > 0.0


def test_make_sigma_factory_use_log_false_builds_bounded_sigma_beta_with_value_1():
    sigma_factory = make_sigma_factory(use_log=False)

    expr = sigma_factory(prefix="meas_I2")
    _assert_is_expression(expr)

    # In direct mode, sigma is a Beta with lb=SMALL_POSITIVE and default value 1
    assert isinstance(expr, Beta)
    assert expr.name == "meas_I2_sigma"
    assert expr.lower_bound == SMALL_POSITIVE
    assert expr.upper_bound is None
    assert pytest.approx(expr.get_value(), abs=1e-12) == 1.0
    assert expr.get_value() > 0.0


def test_sigma_factory_is_specialization_of_positive_factory_naming_and_value_scheme():
    # This test ensures the internal `value = -1 if use_log else 1` line is covered,
    # and that sigma_factory calls positive_factory("sigma", prefix, value=...)
    sf_log = make_sigma_factory(use_log=True)
    sf_direct = make_sigma_factory(use_log=False)

    e_log = sf_log(prefix="P")
    e_dir = sf_direct(prefix="P")

    # log-space sigma should contain suffix _sigma_log and evaluate to exp(-1)
    assert "P_sigma_log" in str(e_log)
    assert pytest.approx(e_log.get_value(), rel=1e-12) == pytest.approx(
        2.718281828459045 ** (-1), rel=1e-12
    )

    # direct sigma should be Beta named P_sigma with value 1 and lower bound SMALL_POSITIVE
    assert isinstance(e_dir, Beta)
    assert e_dir.name == "P_sigma"
    assert e_dir.lower_bound == SMALL_POSITIVE
    assert pytest.approx(e_dir.get_value(), abs=1e-12) == 1.0


@pytest.mark.parametrize(
    "use_log, init_value, expected_value",
    [
        (True, 0.0, 1.0),  # exp(0)=1
        (True, 1.0, 2.718281828459045),  # exp(1)
        (False, 0.5, 0.5),  # direct Beta
        (False, 3.2, 3.2),
    ],
)
def test_positive_factory_get_value_matches_parameterization(
    use_log, init_value, expected_value
):
    factory = make_positive_parameter_factory(use_log=use_log)
    expr = factory(name="x", prefix="PFX", value=init_value)
    _assert_is_expression(expr)

    # Use get_value() instead of type checks: robust to expression composition
    assert pytest.approx(expr.get_value(), rel=1e-12, abs=1e-12) == expected_value
    assert expr.get_value() > 0.0
