import pandas as pd

# Adjust this import to your real module path if needed
# e.g. from biogeme.bayesian_estimation.pandas_output import ...
from biogeme.bayesian_estimation.pandas_output import (
    _build_parameters_dataframe,
    get_pandas_estimated_parameters,
    get_pandas_other_variables,
)


# ----------------------------------------------------------------------
# Dummy objects to mimic BayesianResults content
# ----------------------------------------------------------------------
class DummyEstimate:
    """Simple container mimicking one parameter entry in BayesianResults."""

    def __init__(
        self,
        mean: float,
        median: float,
        mode: float,
        std_err: float,
        z_value: float,
        p_value: float,
        hdi_low: float,
        hdi_high: float,
        rhat: float,
        ess_bulk: float,
        ess_tail: float,
    ):
        self.mean = mean
        self.median = median
        self.mode = mode
        self.std_err = std_err
        self.z_value = z_value
        self.p_value = p_value
        self.hdi_low = hdi_low
        self.hdi_high = hdi_high
        self.rhat = rhat
        self.effective_sample_size_bulk = ess_bulk
        self.effective_sample_size_tail = ess_tail


class DummyBayesianResults:
    """
    Minimal stand-in for BayesianResults for the purpose of testing
    pandas_output._build_parameters_dataframe.
    """

    def __init__(
        self,
        est_keys: list[str],
        other_keys: list[str],
        parameters: dict[str, DummyEstimate],
    ):
        self._est_keys = list(est_keys)
        self._other_keys = list(other_keys)
        self.parameters = dict(parameters)

    def parameter_estimates(self):
        return self._est_keys

    def other_variables(self):
        return self._other_keys


# Common column order for all tests
EXPECTED_COLUMNS = [
    "Id",
    "Name",
    "Value (mean)",
    "Value (median)",
    "Value (mode)",
    "std err.",
    "z-value",
    "p-value",
    "HDI low",
    "HDI high",
    "R hat",
    "ESS (bulk)",
    "ESS (tail)",
]


def _make_dummy_results_basic() -> DummyBayesianResults:
    params = {
        "beta_time": DummyEstimate(
            mean=-0.5,
            median=-0.495,
            mode=-0.49,
            std_err=0.1,
            z_value=-5.0,
            p_value=0.0001,
            hdi_low=-0.7,
            hdi_high=-0.3,
            rhat=1.01,
            ess_bulk=800.0,
            ess_tail=900.0,
        ),
        "beta_cost": DummyEstimate(
            mean=-0.2,
            median=-0.195,
            mode=-0.19,
            std_err=0.05,
            z_value=-4.0,
            p_value=0.0003,
            hdi_low=-0.3,
            hdi_high=-0.1,
            rhat=1.00,
            ess_bulk=1000.0,
            ess_tail=950.0,
        ),
    }
    return DummyBayesianResults(
        est_keys=["beta_time", "beta_cost"],
        other_keys=["beta_cost"],  # arbitrary choice just for testing “other_variables”
        parameters=params,
    )


# ----------------------------------------------------------------------
# Core behavior tests for _build_parameters_dataframe
# ----------------------------------------------------------------------
def test_build_parameters_dataframe_basic_no_rename_no_sort():
    """Basic sanity check: two parameters, no renaming, no sorting."""
    res = _make_dummy_results_basic()

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=None,
        sort_by_name=False,
    )

    # Shape and columns
    assert list(df.columns) == EXPECTED_COLUMNS[1:]  # index 'Id' is hidden
    assert df.shape == (2, len(EXPECTED_COLUMNS) - 1)

    # Index properties
    assert list(df.index) == [0, 1]
    assert df.index.name is None

    # Check first row content
    row0 = df.loc[0]
    est0 = res.parameters["beta_time"]
    assert row0["Name"] == "beta_time"
    assert row0["Value (mean)"] == est0.mean
    assert row0["Value (median)"] == est0.median
    assert row0["Value (mode)"] == est0.mode
    assert row0["std err."] == est0.std_err
    assert row0["z-value"] == est0.z_value
    assert row0["p-value"] == est0.p_value
    assert row0["HDI low"] == est0.hdi_low
    assert row0["HDI high"] == est0.hdi_high
    assert row0["R hat"] == est0.rhat
    assert row0["ESS (bulk)"] == est0.effective_sample_size_bulk
    assert row0["ESS (tail)"] == est0.effective_sample_size_tail


def test_build_parameters_dataframe_other_variables():
    """When estimated_parameters=False, we should use other_variables() keys."""
    res = _make_dummy_results_basic()

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=False,
        renaming_parameters=None,
        sort_by_name=False,
    )

    # Only one key in other_variables
    assert df.shape[0] == 1
    assert "beta_cost" in df["Name"].tolist()


def test_build_parameters_dataframe_missing_parameter_entry_skipped():
    """
    If the key returned by parameter_estimates() is not present in
    estimation_results.parameters, it should be skipped.
    """
    base = _make_dummy_results_basic()
    # Add a key in the list, but not in parameters dict
    res = DummyBayesianResults(
        est_keys=["beta_time", "missing_param"],
        other_keys=[],
        parameters=base.parameters,
    )

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=None,
        sort_by_name=False,
    )

    # Only beta_time should appear
    assert df.shape[0] == 1
    assert df.iloc[0]["Name"] == "beta_time"


def test_build_parameters_dataframe_empty_keys_returns_empty_df():
    """If there are no keys, the DataFrame should be empty but valid."""
    res = DummyBayesianResults(est_keys=[], other_keys=[], parameters={})

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=None,
        sort_by_name=False,
    )

    assert df.empty
    # Column order still as expected (minus index)
    assert list(df.columns) == EXPECTED_COLUMNS[1:]
    assert df.index.name is None


def test_build_parameters_dataframe_renaming_parameters_applied():
    """Names should be renamed according to renaming_parameters mapping."""
    res = _make_dummy_results_basic()
    renaming = {
        "beta_time": "Time coefficient",
        "beta_cost": "Cost coefficient",
    }

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=renaming,
        sort_by_name=False,
    )

    assert set(df["Name"]) == {"Time coefficient", "Cost coefficient"}


def test_build_parameters_dataframe_renaming_parameters_with_duplicates():
    """
    Duplicate target names in renaming_parameters should not crash the function.
    Behavior: names are still applied, even if they collide.
    """
    res = _make_dummy_results_basic()
    renaming = {
        "beta_time": "coef",
        "beta_cost": "coef",  # duplicate
    }

    df = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=renaming,
        sort_by_name=False,
    )

    # Both rows should be present, even if they have identical "Name"
    assert df.shape[0] == 2
    assert all(name == "coef" for name in df["Name"])


def test_build_parameters_dataframe_sort_by_name_stable():
    """
    When sort_by_name=True, data should be sorted lexicographically by Name, then Id.
    """
    # Construct keys in "unsorted" order with names that will sort differently
    params = {
        "beta_z": DummyEstimate(1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1),
        "beta_a": DummyEstimate(2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1),
        "beta_m": DummyEstimate(3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1),
    }
    res = DummyBayesianResults(
        est_keys=["beta_z", "beta_a", "beta_m"], other_keys=[], parameters=params
    )

    df_unsorted = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=None,
        sort_by_name=False,
    )
    df_sorted = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=None,
        sort_by_name=True,
    )

    # Unsorted order should follow est_keys order
    assert list(df_unsorted["Name"]) == ["beta_z", "beta_a", "beta_m"]

    # Sorted order should be alphabetical
    assert list(df_sorted["Name"]) == ["beta_a", "beta_m", "beta_z"]

    # Index corresponds to original Ids; we only require the same set of Ids and no name
    assert sorted(df_sorted.index.tolist()) == [0, 1, 2]
    assert df_sorted.index.name is None


# ----------------------------------------------------------------------
# Wrapper functions tests
# ----------------------------------------------------------------------
def test_get_pandas_estimated_parameters_delegation():
    """get_pandas_estimated_parameters should behave like _build_parameters_dataframe with estimated_parameters=True."""
    res = _make_dummy_results_basic()
    renaming = {"beta_time": "Time", "beta_cost": "Cost"}

    df_direct = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=True,
        renaming_parameters=renaming,
        sort_by_name=True,
    )
    df_wrapper = get_pandas_estimated_parameters(
        estimation_results=res,
        renaming_parameters=renaming,
        sort_by_name=True,
    )

    pd.testing.assert_frame_equal(df_direct, df_wrapper)


def test_get_pandas_other_variables_delegation():
    """get_pandas_other_variables should behave like _build_parameters_dataframe with estimated_parameters=False."""
    res = _make_dummy_results_basic()

    df_direct = _build_parameters_dataframe(
        estimation_results=res,
        estimated_parameters=False,
        renaming_parameters=None,
        sort_by_name=False,
    )
    df_wrapper = get_pandas_other_variables(
        estimation_results=res,
        renaming_parameters=None,
        sort_by_name=False,
    )

    pd.testing.assert_frame_equal(df_direct, df_wrapper)
