# biogeme/tests/bayesian/test_sampling_strategy.py
"""
Comprehensive tests for biogeme/bayesian_estimation/sampling_strategy.py

Covers:
- SamplerPlan constructors: from_name, from_mapping, from_user_plan
- SamplerPlanner.plan across all hardware permutations:
  * No JAX/NumPyro available  -> PyMC fallback
  * JAX CPU single device     -> NumPyro vectorized
  * JAX CPU multi devices >= chains -> NumPyro parallel
  * JAX CPU multi devices <  chains -> NumPyro vectorized
  * JAX GPU single device + chains > guard -> still vectorized (warn)
  * prefer_vectorized_single_device=False + allow_pymc_multiprocessing_fallback=True -> PyMC
  * prefer_vectorized_single_device=False + fallback=False -> NumPyro vectorized (if JAX present)
  * chains == 1 -> NumPyro vectorized when JAX present
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
import unittest

MOD_NAME = "biogeme.bayesian_estimation.sampling_strategy"
LOGGER_NAME = MOD_NAME  # module uses logging.getLogger(__name__)


def _reload_module():
    """Reload the module under test so it sees the current mocked environment."""
    if MOD_NAME in sys.modules:
        return importlib.reload(sys.modules[MOD_NAME])
    return importlib.import_module(MOD_NAME)


# --------------------------
# Lightweight environment stubs
# --------------------------


def _install_numpyro_and_pymc_sampling_jax() -> None:
    """Install minimal numpyro + pymc.sampling.jax so imports in planner succeed."""
    sys.modules["numpyro"] = sys.modules.get("numpyro") or types.ModuleType("numpyro")

    # Ensure parent packages are importable
    pymc_pkg = sys.modules.get("pymc") or types.ModuleType("pymc")
    pymc_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["pymc"] = pymc_pkg

    sampling_pkg = sys.modules.get("pymc.sampling") or types.ModuleType("pymc.sampling")
    sampling_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["pymc.sampling"] = sampling_pkg

    sampling_jax = sys.modules.get("pymc.sampling.jax") or types.ModuleType(
        "pymc.sampling.jax"
    )

    def _dummy_sample_numpyro_nuts(*_args, **_kwargs):
        return None

    sampling_jax.sample_numpyro_nuts = _dummy_sample_numpyro_nuts
    sys.modules["pymc.sampling.jax"] = sampling_jax


def _install_dummy_jax(devices: list[types.SimpleNamespace]) -> None:
    """Install a tiny 'jax' module with devices() and local_device_count()."""
    jax = sys.modules.get("jax") or types.ModuleType("jax")

    def devices_fn():
        return devices

    def local_device_count():
        return len(devices)

    jax.devices = devices_fn
    jax.local_device_count = local_device_count
    sys.modules["jax"] = jax


def _uninstall_jax_stack() -> None:
    """Remove jax / numpyro / pymc.sampling.jax to trigger PyMC fallback."""
    for m in ("jax", "numpyro", "pymc.sampling.jax", "pymc.sampling", "pymc"):
        sys.modules.pop(m, None)


def _device(platform: str, id_: int, kind: str = "Device") -> types.SimpleNamespace:
    d = types.SimpleNamespace()
    d.platform = platform
    d.id = id_
    d.device_kind = kind
    return d


# --------------------------
# Tests: SamplerPlan constructors
# --------------------------


class TestSamplerPlanConstructors(unittest.TestCase):
    def setUp(self):
        # Snapshot sys.modules so we can restore after each test
        self._snapshot = sys.modules.copy()

        # Ensure biogeme.tools exists (the module under test imports & logs via it).
        # We don't fabricate packages; we only attach a tools submodule if biogeme exists.
        try:
            import biogeme.tools  # noqa: F401
        except Exception:
            try:
                import biogeme  # noqa: F401

                tools_mod = types.ModuleType("biogeme.tools")

                def report_jax_cpu_devices():
                    return "dummy report"

                def warning_cpu_devices():
                    return None

                tools_mod.report_jax_cpu_devices = report_jax_cpu_devices
                tools_mod.warning_cpu_devices = warning_cpu_devices
                sys.modules["biogeme.tools"] = tools_mod
                setattr(sys.modules["biogeme"], "tools", tools_mod)
            except Exception:
                # If there's no biogeme on sys.path, the import of the module under test will fail anyway.
                pass

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._snapshot)

    def test_from_name_valid(self):
        m = _reload_module()
        SP = m.SamplerPlan

        self.assertEqual(
            SP.from_name("numpyro-parallel"),
            SP("numpyro", "parallel", None, "user-defined"),
        )
        self.assertEqual(
            SP.from_name("parallel"), SP("numpyro", "parallel", None, "user-defined")
        )
        self.assertEqual(
            SP.from_name("numpyro-vectorized"),
            SP("numpyro", "vectorized", None, "user-defined"),
        )
        self.assertEqual(
            SP.from_name("vectorized"),
            SP("numpyro", "vectorized", None, "user-defined"),
        )
        self.assertEqual(SP.from_name("pymc"), SP("pymc", None, None, "user-defined"))
        self.assertEqual(SP.from_name("pm"), SP("pymc", None, None, "user-defined"))

    def test_from_name_invalid(self):
        m = _reload_module()
        SP = m.SamplerPlan
        with self.assertRaises(ValueError):
            SP.from_name("auto")
        with self.assertRaises(ValueError):
            SP.from_name("unknown-plan")

    def test_from_mapping_valid_numpyro(self):
        m = _reload_module()
        SP = m.SamplerPlan

        p = SP.from_mapping({"backend": "numpyro", "chain_method": "parallel"})
        self.assertEqual(
            (p.backend, p.chain_method, p.cores), ("numpyro", "parallel", None)
        )

        p2 = SP.from_mapping(
            {"backend": "numpyro", "chain_method": "vectorized", "cores": None}
        )
        self.assertEqual(
            (p2.backend, p2.chain_method, p2.cores), ("numpyro", "vectorized", None)
        )

    def test_from_mapping_valid_pymc(self):
        m = _reload_module()
        SP = m.SamplerPlan

        p = SP.from_mapping({"backend": "pymc", "cores": 4})
        self.assertEqual((p.backend, p.chain_method, p.cores), ("pymc", None, 4))

        p2 = SP.from_mapping(
            {"backend": "pymc", "cores": 2, "chain_method": "parallel"}
        )
        self.assertEqual((p2.backend, p2.chain_method, p2.cores), ("pymc", None, 2))

    def test_from_mapping_invalid(self):
        m = _reload_module()
        SP = m.SamplerPlan
        with self.assertRaises(ValueError):
            SP.from_mapping({"backend": "other"})
        with self.assertRaises(ValueError):
            SP.from_mapping({"backend": "numpyro", "chain_method": "something"})

    def test_from_user_plan_dispatch(self):
        m = _reload_module()
        SP = m.SamplerPlan

        passthrough = SP("pymc", None, 3, "u")
        self.assertIs(SP.from_user_plan(passthrough), passthrough)

        from_name = SP.from_user_plan("numpyro-parallel")
        self.assertEqual(
            (from_name.backend, from_name.chain_method), ("numpyro", "parallel")
        )

        from_dict = SP.from_user_plan({"backend": "pymc", "cores": 5})
        self.assertEqual((from_dict.backend, from_dict.cores), ("pymc", 5))

        with self.assertRaises(TypeError):
            SP.from_user_plan(123)  # type: ignore[arg-type


# --------------------------
# Tests: SamplerPlanner.plan (auto selection)
# --------------------------


class TestSamplerPlannerPlan(unittest.TestCase):
    def setUp(self):
        self._snapshot = sys.modules.copy()

        # Ensure biogeme.tools exists if the real package is importable
        try:
            import biogeme.tools  # noqa: F401
        except Exception:
            try:
                import biogeme  # noqa: F401

                tools_mod = types.ModuleType("biogeme.tools")

                def report_jax_cpu_devices():
                    return "dummy report"

                def warning_cpu_devices():
                    return None

                tools_mod.report_jax_cpu_devices = report_jax_cpu_devices
                tools_mod.warning_cpu_devices = warning_cpu_devices
                sys.modules["biogeme.tools"] = tools_mod
                setattr(sys.modules["biogeme"], "tools", tools_mod)
            except Exception:
                pass

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._snapshot)

    def test_no_jax_stack_fallback_to_pymc(self):
        """If no JAX/NumPyro/pymc.sampling.jax, plan must fallback to PyMC with cores=chains."""
        _uninstall_jax_stack()
        m = _reload_module()

        planner = m.SamplerPlanner(number_of_threads=4)
        plan = planner.plan(chains=5)
        self.assertEqual(plan.backend, "pymc")
        self.assertEqual(plan.cores, 5)
        self.assertIsNone(plan.chain_method)

    def test_cpu_single_device_vectorized(self):
        """Single CPU device -> vectorized (default)."""
        _install_dummy_jax([_device("cpu", 0)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(number_of_threads=4)
        plan = planner.plan(chains=3)
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "vectorized")
        self.assertIsNone(plan.cores)

    def test_cpu_multi_devices_parallel(self):
        """n_dev >= chains > 1 -> parallel."""
        _install_dummy_jax([_device("cpu", 0), _device("cpu", 1), _device("cpu", 2)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(number_of_threads=4)
        plan = planner.plan(chains=3)
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "parallel")

    def test_cpu_multi_devices_but_less_than_chains_vectorized(self):
        """n_dev < chains -> vectorized (since parallel branch doesn't match)."""
        _install_dummy_jax([_device("cpu", 0), _device("cpu", 1)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(number_of_threads=4)
        plan = planner.plan(
            chains=4
        )  # 2 devices, 4 chains -> not parallel, expect vectorized
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "vectorized")

    def test_gpu_single_device_guardrail_warning_and_vectorized(self):
        """Single GPU device + many chains -> still vectorized, with a warning."""
        _install_dummy_jax([_device("gpu", 0, "A100")])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(number_of_threads=4, max_vectorized_chains_on_gpu=1)
        with self.assertLogs(LOGGER_NAME, level="WARNING") as cm:
            plan = planner.plan(chains=3)
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "vectorized")
        self.assertTrue(any("Single GPU detected" in rec for rec in cm.output))

    def test_disable_vectorized_enable_pymc_fallback(self):
        """prefer_vectorized_single_device=False + allow_pymc_multiprocessing_fallback=True -> PyMC."""
        _install_dummy_jax([_device("cpu", 0)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(
            number_of_threads=4,
            prefer_vectorized_single_device=False,
            allow_pymc_multiprocessing_fallback=True,
        )
        plan = planner.plan(chains=3)
        self.assertEqual(plan.backend, "pymc")
        self.assertIsNone(plan.chain_method)
        self.assertEqual(plan.cores, 3)

    def test_disable_vectorized_without_fallback_keeps_numpyro(self):
        """prefer_vectorized_single_device=False + fallback=False -> default NumPyro vectorized (if JAX present)."""
        _install_dummy_jax([_device("cpu", 0)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        planner = m.SamplerPlanner(
            number_of_threads=4,
            prefer_vectorized_single_device=False,
            allow_pymc_multiprocessing_fallback=False,
        )
        plan = planner.plan(chains=2)
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "vectorized")

    def test_chains_equal_one_edge_case(self):
        """chains == 1 -> parallel branch not applicable, expect vectorized if JAX present."""
        _install_dummy_jax([_device("cpu", 0), _device("cpu", 1)])
        _install_numpyro_and_pymc_sampling_jax()
        m = _reload_module()

        plan = m.SamplerPlanner(number_of_threads=4).plan(chains=1)
        self.assertEqual(plan.backend, "numpyro")
        self.assertEqual(plan.chain_method, "vectorized")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
