import logging
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from biogeme.assisted import AssistedSpecification, ParetoPostProcessing
from biogeme.exceptions import BiogemeError


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeSetElement:
    """Minimal stand-in for biogeme_optimization.pareto.SetElement."""

    def __init__(self, element_id: str):
        self.element_id = element_id


class FakePareto:
    """Fake Pareto container for ParetoPostProcessing tests."""

    def __init__(self, filename: str):
        self.filename = filename
        # prepopulate with two fake elements, using valid configuration string IDs
        self.pareto = [
            FakeSetElement("asc:MALE-GA;b_cost:GA;train_tt:boxcox"),
            FakeSetElement("asc:MALE-GA;b_cost:no_seg;train_tt:log"),
        ]
        self._stats_called = False
        self._plot_calls = []

    def statistics(self):
        self._stats_called = True
        return ["stat line 1", "stat line 2"]

    def plot(
        self,
        objective_x,
        objective_y,
        label_x,
        label_y,
        margin_x,
        margin_y,
        ax,
    ):
        self._plot_calls.append(
            dict(
                objective_x=objective_x,
                objective_y=objective_y,
                label_x=label_x,
                label_y=label_y,
                margin_x=margin_x,
                margin_y=margin_y,
                ax=ax,
            )
        )
        # return a simple sentinel that a caller could inspect
        return "PLOT_OK"


class FakeBiogemeParams:
    """Minimal stand-in for biogeme.parameters.Parameters used in AssistedSpecification."""

    def __init__(self, values: dict[tuple[str, str | None], Any] | None = None):
        self._values = values or {}
        self.file_name = "fake_parameters.toml"

    def read_file(self, filename: str | None) -> None:
        # For unit tests we ignore the file and keep _values
        self.file_name = filename or self.file_name

    def get_value(self, name: str, section: str | None = None) -> Any:
        key = (name, section)
        if key in self._values:
            return self._values[key]
        # Some tests rely on these defaults
        if name == "maximum_number_catalog_expressions":
            return 100
        if name == "largest_neighborhood" and section == "AssistedSpecification":
            return 3
        if name == "number_of_neighbors" and section == "AssistedSpecification":
            return 5
        if name == "maximum_attempts" and section == "AssistedSpecification":
            return 10
        if name == "maximum_number_parameters" and section == "AssistedSpecification":
            return 100
        raise KeyError(f"Missing parameter {name!r} / section {section!r}")


class FakeCentralController:
    """Fake for biogeme.catalog.CentralController."""

    def __init__(self, expression, maximum_number_of_configurations: int):
        self.expression = expression
        self.max_configs = maximum_number_of_configurations
        self._operators_prepared = False

    def prepare_operators(self) -> dict[str, Any]:
        """Return a dict of fake controller operators."""
        self._operators_prepared = True

        def op(config, step: int):
            """Controller operator: take a configuration-like object and return a modified FakeConfiguration.

            In the real AssistedSpecification code, `config` may be either a FakeConfiguration
            (in our tests) or a real catalog.Configuration instance. We therefore avoid
            assuming the presence of a `.config_id` attribute and fall back to `str(config)`
            when needed.
            """
            base_id = getattr(config, "config_id", None)
            if base_id is None:
                # For real catalog.Configuration, we rely on its string representation
                # to be the configuration identifier.
                base_id = str(config)
            return FakeConfiguration(f"{base_id}_step{step}"), step

        return {"fake_op": op}

    def number_of_configurations(self) -> int:
        return self.max_configs

    def expression_configuration_iterator(self):
        """Yield a few fake configurations."""
        for cid in ["C0", "C1", "C2"]:
            yield FakeConfiguration(cid)


class FakeConfiguration:
    """Minimal replacement for biogeme.catalog.Configuration used in AssistedSpecification."""

    def __init__(self, config_id: str):
        self.config_id = config_id

    @classmethod
    def from_string(cls, s: str) -> "FakeConfiguration":
        return cls(s)

    def __repr__(self) -> str:  # just for logging friendliness
        return f"FakeConfiguration({self.config_id!r})"


class FakeSpecificationElement:
    """Object returned by Specification.get_element, wrapping objectives and element_id."""

    def __init__(self, element_id: str, objectives: list[float]):
        self.element_id = element_id
        self.objectives = objectives

    def __repr__(self):
        return (
            f"FakeSpecificationElement(id={self.element_id!r}, obj={self.objectives!r})"
        )


class FakeSpecification:
    """Fake for biogeme.catalog.specification.Specification."""

    # class attributes that AssistedSpecification mutates
    generic_name: str | None = None
    user_defined_validity_check = None
    central_controller: FakeCentralController | None = None
    biogeme_parameters: FakeBiogemeParams | None = None
    expression = None
    database = None
    pareto = None

    def __init__(self, configuration: FakeConfiguration | None = None):
        self.configuration = configuration or FakeConfiguration("DEFAULT")
        self._validity = (True, "")
        self._element_counter = 0

    @classmethod
    def from_string_id(cls, s: str) -> "FakeSpecification":
        return cls(FakeConfiguration(s))

    @classmethod
    def default_specification(cls) -> "FakeSpecification":
        return cls(FakeConfiguration("DEFAULT"))

    @property
    def validity(self) -> tuple[bool, str]:
        return self._validity

    def get_element(
        self, multi_objectives
    ) -> FakeSpecificationElement:  # type: ignore[override]
        """Return a fake SetElement-like object with an element_id and objectives."""
        self._element_counter += 1
        obj = multi_objectives(None)
        return FakeSpecificationElement(self.configuration.config_id, obj)

    def __repr__(self):
        return f"FakeSpecification({self.configuration!r})"


class FakeParetoClass:
    """Fake for biogeme_optimization.vns.ParetoClass."""

    def __init__(self, max_neighborhood: int, pareto_file: str):
        self.max_neighborhood = max_neighborhood
        self.filename = pareto_file
        self.comments: list[str] = []
        self.pareto: list[FakeSpecificationElement] = []

    def add(self, element: FakeSpecificationElement) -> None:
        self.pareto.append(element)

    def dump(self) -> None:
        # no-op in tests
        pass

    def length_of_all_sets(self) -> tuple[int, int]:
        """Return (#pareto_elements, #models). Simplified here as (len, len)."""
        n = len(self.pareto)
        return n, n


# ---------------------------------------------------------------------------
# ParetoPostProcessing tests
# ---------------------------------------------------------------------------


def test_pareto_post_processing_init_without_loglike_raises_biogeme_error(monkeypatch):
    class DummyBiogeme:
        log_like = None
        database = pd.DataFrame({"x": [1, 2, 3]})
        biogeme_parameters = FakeBiogemeParams()
        model_name = "dummy"

    # Patch Pareto to avoid touching real filesystem
    monkeypatch.setattr("biogeme.assisted.Pareto", FakePareto)

    with pytest.raises(BiogemeError, match="No log likelihood function"):
        ParetoPostProcessing(
            biogeme_object=DummyBiogeme(), pareto_file_name="dummy.pareto"
        )


def test_pareto_post_processing_reestimate_calls_estimate_for_each_element(monkeypatch):
    # Fake biogeme object
    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_LIKE_EXPR"
            self.database = pd.DataFrame({"x": [1, 2, 3]})
            self.biogeme_parameters = FakeBiogemeParams()
            self.model_name = "dummy_name"

    # Patch Pareto to our fake one
    fake_pareto = FakePareto("fake.pareto")
    monkeypatch.setattr("biogeme.assisted.Pareto", lambda filename: fake_pareto)

    # Patch ModelNames
    class FakeModelNames:
        def __init__(self, prefix: str):
            self.prefix = prefix
            self.counter = 0

        def __call__(self, config_id: str) -> str:
            self.counter += 1
            return f"{self.prefix}_{self.counter}"

    monkeypatch.setattr(
        "biogeme.assisted.biogeme.tools.unique_ids.ModelNames", FakeModelNames
    )

    # Patch CentralController
    monkeypatch.setattr(
        "biogeme.assisted.CentralController",
        lambda expression, maximum_number_of_configurations: FakeCentralController(
            expression, maximum_number_of_configurations
        ),
    )

    # Patch BIOGEME.from_configuration_and_controller
    class FakeBiogemeFromConfig:
        def __init__(self, config_id: str):
            self.config_id = config_id
            self.model_name = None
            self.biogeme_parameters = FakeBiogemeParams()
            self.estimate_calls = []

        def estimate(self, recycle: bool = False):
            self.estimate_calls.append(recycle)
            return f"RESULT_{self.config_id}_{'R' if recycle else 'N'}"

        @classmethod
        def from_configuration_and_controller(
            cls, config_id, central_controller, database, parameters
        ):
            return cls(config_id)

    monkeypatch.setattr("biogeme.assisted.BIOGEME", FakeBiogemeFromConfig, raising=True)

    pp = ParetoPostProcessing(DummyBiogeme(), pareto_file_name="fake.pareto")

    results = pp.reestimate(recycle=True)

    # We had two Pareto elements with specific configuration IDs → 2 results
    expected_ids = {
        "asc:MALE-GA;b_cost:GA;train_tt:boxcox",
        "asc:MALE-GA;b_cost:no_seg;train_tt:log",
    }
    assert set(results.keys()) == expected_ids
    for cfg_id in expected_ids:
        assert results[cfg_id] == f"RESULT_{cfg_id}_R"


def test_pareto_post_processing_log_statistics_logs_messages(monkeypatch, caplog):
    dummy_biogeme = MagicMock()
    dummy_biogeme.log_like = "LOG"
    dummy_biogeme.database = pd.DataFrame({"x": [1, 2]})
    dummy_biogeme.biogeme_parameters = FakeBiogemeParams()
    dummy_biogeme.model_name = "dummy"

    fake_pareto = FakePareto("fake.pareto")
    monkeypatch.setattr("biogeme.assisted.Pareto", lambda filename: fake_pareto)
    monkeypatch.setattr(
        "biogeme.assisted.CentralController",
        lambda expression, maximum_number_of_configurations: FakeCentralController(
            expression, maximum_number_of_configurations
        ),
    )

    pp = ParetoPostProcessing(dummy_biogeme, pareto_file_name="fake.pareto")

    caplog.set_level(logging.INFO)
    pp.log_statistics()
    # Both statistics lines should have been logged
    assert any("stat line 1" in rec.message for rec in caplog.records)
    assert any("stat line 2" in rec.message for rec in caplog.records)


def test_pareto_post_processing_plot_delegates_to_pareto(monkeypatch):
    dummy_biogeme = MagicMock()
    dummy_biogeme.log_like = "LOG"
    dummy_biogeme.database = pd.DataFrame({"x": [1, 2]})
    dummy_biogeme.biogeme_parameters = FakeBiogemeParams()
    dummy_biogeme.model_name = "dummy"

    fake_pareto = FakePareto("fake.pareto")
    monkeypatch.setattr("biogeme.assisted.Pareto", lambda filename: fake_pareto)
    monkeypatch.setattr(
        "biogeme.assisted.CentralController",
        lambda expression, maximum_number_of_configurations: FakeCentralController(
            expression, maximum_number_of_configurations
        ),
    )

    pp = ParetoPostProcessing(dummy_biogeme, pareto_file_name="fake.pareto")

    result = pp.plot(
        objective_x=0,
        objective_y=1,
        label_x="x",
        label_y="y",
        margin_x=1,
        margin_y=2,
        ax=None,
    )
    assert result == "PLOT_OK"
    assert len(fake_pareto._plot_calls) == 1
    call = fake_pareto._plot_calls[0]
    assert call["objective_x"] == 0
    assert call["objective_y"] == 1
    assert call["label_x"] == "x"
    assert call["label_y"] == "y"


# ---------------------------------------------------------------------------
# AssistedSpecification tests
# ---------------------------------------------------------------------------


def test_assisted_specification_ctor_sets_up_operators(monkeypatch):
    # Patch dependencies inside biogeme.assisted
    monkeypatch.setattr("biogeme.assisted.CentralController", FakeCentralController)
    monkeypatch.setattr("biogeme.assisted.Specification", FakeSpecification)
    monkeypatch.setattr("biogeme.assisted.ParetoClass", FakeParetoClass)

    # Fake biogeme object
    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_EXPR"
            self.database = pd.DataFrame({"x": [1, 2, 3]})
            self.biogeme_parameters = FakeBiogemeParams()
            self.model_name = "dummy_model"
            self.maximum_number_catalog_expressions = 100

    def multi_obj(result):
        return [1.0, 2.0]  # 2 objectives, arbitrary

    spec = AssistedSpecification(
        biogeme_object=DummyBiogeme(),
        multi_objectives=multi_obj,
        pareto_file_name="fake.pareto",
        validity=None,
    )

    # One operator defined by FakeCentralController.prepare_operators
    assert "fake_op" in spec.operators
    assert callable(spec.operators["fake_op"])


def test_assisted_generate_operator_wraps_controller_operator(monkeypatch):
    monkeypatch.setattr("biogeme.assisted.CentralController", FakeCentralController)
    monkeypatch.setattr("biogeme.assisted.Specification", FakeSpecification)
    monkeypatch.setattr("biogeme.assisted.ParetoClass", FakeParetoClass)

    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_EXPR"
            self.database = pd.DataFrame({"x": [1, 2]})
            self.biogeme_parameters = FakeBiogemeParams()
            self.model_name = "dummy_model"
            self.maximum_number_catalog_expressions = 100

    def multi_obj(_):
        return [42.0]

    spec = AssistedSpecification(
        biogeme_object=DummyBiogeme(),
        multi_objectives=multi_obj,
        pareto_file_name="fake.pareto",
        validity=None,
    )

    # Take the only operator we created in FakeCentralController
    op = next(iter(spec.operators.values()))

    element = FakeSetElement("asc:GA;b_cost:GA;train_tt:boxcox")
    new_element, n_mod = op(element, step=3)

    # The fake operator appends "_step<step>"
    assert isinstance(new_element, FakeSpecificationElement)
    assert new_element.element_id == "asc:GA;b_cost:GA;train_tt:boxcox_step3"
    assert n_mod == 3
    assert new_element.objectives == [42.0]


def test_assisted_is_valid_delegates_to_specification(monkeypatch):
    monkeypatch.setattr("biogeme.assisted.CentralController", FakeCentralController)
    monkeypatch.setattr("biogeme.assisted.Specification", FakeSpecification)
    monkeypatch.setattr("biogeme.assisted.ParetoClass", FakeParetoClass)

    # Make AssistedSpecification.is_valid accept FakeSetElement
    monkeypatch.setattr("biogeme.assisted.SetElement", FakeSetElement)

    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_EXPR"
            self.database = pd.DataFrame({"x": [1, 2]})
            self.biogeme_parameters = FakeBiogemeParams()
            self.model_name = "dummy"
            self.maximum_number_catalog_expressions = 100

    def multi_obj(_):
        return [0.0]

    assisted = AssistedSpecification(
        biogeme_object=DummyBiogeme(),
        multi_objectives=multi_obj,
        pareto_file_name="fake.pareto",
        validity=None,
    )

    elem = FakeSetElement("CONF_X")
    valid, why = assisted.is_valid(elem)
    assert valid is True
    assert isinstance(why, str)


def test_assisted_run_when_all_configurations_enumerated(monkeypatch, caplog):
    """Path where number_of_specifications <= maximum_number_catalog_expressions."""
    # Patch

    def small_cc_factory(expression, maximum_number_of_configurations):
        # pretend the catalogue has only 3 configurations,
        # which is <= maximum_number_catalog_expressions (=10)
        return FakeCentralController(expression, maximum_number_of_configurations=3)

    monkeypatch.setattr("biogeme.assisted.CentralController", small_cc_factory)
    monkeypatch.setattr("biogeme.assisted.Specification", FakeSpecification)
    monkeypatch.setattr("biogeme.assisted.ParetoClass", FakeParetoClass)

    # We don't want to run vns in this branch, but let's patch it anyway for safety
    vns_mock = MagicMock(name="vns")
    monkeypatch.setattr("biogeme.assisted.vns", vns_mock)

    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_EXPR"
            self.database = pd.DataFrame({"x": [1, 2, 3]})
            self.biogeme_parameters = FakeBiogemeParams(
                {("maximum_number_catalog_expressions", None): 10}
            )
            self.model_name = "dummy"
            self.maximum_number_catalog_expressions = 10

    def multi_obj(_):
        return [1.0]

    caplog.set_level(logging.INFO)
    assisted = AssistedSpecification(
        biogeme_object=DummyBiogeme(),
        multi_objectives=multi_obj,
        pareto_file_name="fake.pareto",
        validity=None,
    )

    # Patch ParetoPostProcessing so run() doesn't try to actually estimate models
    class DummyPP:
        def __init__(self, biogeme_object, pareto_file_name):
            self.biogeme_object = biogeme_object
            self.pareto_file_name = pareto_file_name

        def reestimate(self) -> dict[str, Any]:
            return {"dummy_conf": "dummy_result"}

        def log_statistics(self) -> None:
            pass

    monkeypatch.setattr("biogeme.assisted.ParetoPostProcessing", DummyPP)

    results = assisted.run()
    # We expect our dummy result returned
    assert results == {"dummy_conf": "dummy_result"}
    # And vns should NOT have been called in this branch
    vns_mock.assert_not_called()


def test_assisted_run_when_heuristic_vns_used(monkeypatch):
    """Path where number_of_specifications > maximum_number_catalog_expressions."""

    # CentralController with many configurations to trigger VNS branch
    def central_controller_factory(expression, maximum_number_of_configurations):
        cc = FakeCentralController(expression, maximum_number_of_configurations)
        # Override number_of_configurations to report a large number
        cc.max_configs = 1_000
        return cc

    monkeypatch.setattr(
        "biogeme.assisted.CentralController", central_controller_factory
    )
    monkeypatch.setattr("biogeme.assisted.Specification", FakeSpecification)
    monkeypatch.setattr("biogeme.assisted.ParetoClass", FakeParetoClass)

    # vns will be called and should return a Pareto-like object
    def fake_vns(
        problem, first_solutions, pareto, number_of_neighbors, maximum_attempts
    ):
        # For simplicity just return the same pareto with an extra element
        pareto.add(FakeSpecificationElement("VNS_CONF", [0.0]))
        return pareto

    monkeypatch.setattr("biogeme.assisted.vns", fake_vns)

    class DummyBiogeme:
        def __init__(self):
            self.log_like = "LOG_EXPR"
            self.database = pd.DataFrame({"x": [1, 2, 3]})
            self.biogeme_parameters = FakeBiogemeParams(
                {("maximum_number_catalog_expressions", None): 10}
            )
            self.model_name = "dummy"
            self.maximum_number_catalog_expressions = 10

    def multi_obj(_):
        return [1.0]

    # Dummy post-processing as above
    class DummyPP:
        def __init__(self, biogeme_object, pareto_file_name):
            self.biogeme_object = biogeme_object
            self.pareto_file_name = pareto_file_name

        def reestimate(self) -> dict[str, Any]:
            return {"vns_conf": "vns_result"}

        def log_statistics(self) -> None:
            pass

    monkeypatch.setattr("biogeme.assisted.ParetoPostProcessing", DummyPP)

    assisted = AssistedSpecification(
        biogeme_object=DummyBiogeme(),
        multi_objectives=multi_obj,
        pareto_file_name="fake.pareto",
        validity=None,
    )

    results = assisted.run()
    assert results == {"vns_conf": "vns_result"}
