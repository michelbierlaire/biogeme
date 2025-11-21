import unittest
from unittest.mock import patch

from biogeme.bayesian_estimation import sampling_strategy as m


class TestDescribeStrategies(unittest.TestCase):
    def test_describe_strategies_format_and_content(self) -> None:
        """
        describe_strategies should return a comma-separated string of
        'name (desc)' entries, matching SAMPLER_STRATEGIES_DESCRIPTION.
        """
        text = m.describe_strategies()
        # Basic format: entries separated by ", "
        parts = [p.strip() for p in text.split(",")]
        self.assertEqual(len(parts), len(m.SAMPLER_STRATEGIES_DESCRIPTION))

        for name, desc in m.SAMPLER_STRATEGIES_DESCRIPTION.items():
            expected_piece = f"'{name}' ({desc})"
            self.assertIn(expected_piece, parts)


class TestSelectChainMethod(unittest.TestCase):
    def test_select_chain_method_parallel_when_multiple_devices(self) -> None:
        self.assertEqual(m._select_chain_method(2), "parallel")
        self.assertEqual(m._select_chain_method(8), "parallel")

    def test_select_chain_method_vectorized_when_single_or_zero_device(self) -> None:
        self.assertEqual(m._select_chain_method(1), "vectorized")
        self.assertEqual(m._select_chain_method(0), "vectorized")


class TestMakeSamplingConfigAutomatic(unittest.TestCase):
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=True,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_devices_summary",
        return_value=(4, ["cpu"]),
    )
    def test_automatic_jax_available_multiple_devices(
        self, mock_devices_summary, mock_jax_available
    ) -> None:
        cfg = m.make_sampling_config(strategy="automatic", target_accept=0.9)

        self.assertEqual(cfg.backend, "numpyro")
        self.assertEqual(cfg.chain_method, "parallel")  # >1 device
        self.assertIsNone(cfg.cores)
        self.assertEqual(cfg.target_accept, 0.9)
        self.assertIsNone(cfg.init)
        self.assertIsNone(cfg.max_treedepth)
        self.assertIsNone(cfg.nuts_kwargs)

        mock_jax_available.assert_called_once()
        mock_devices_summary.assert_called_once()

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=True,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_devices_summary",
        return_value=(1, ["cpu"]),
    )
    def test_automatic_jax_available_single_device_vectorized(
        self, mock_devices_summary, mock_jax_available
    ) -> None:
        cfg = m.make_sampling_config(strategy="automatic", target_accept=0.8)

        self.assertEqual(cfg.backend, "numpyro")
        self.assertEqual(cfg.chain_method, "vectorized")  # 1 device
        self.assertIsNone(cfg.cores)
        self.assertEqual(cfg.target_accept, 0.8)

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=True,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_devices_summary",
        return_value=(0, []),
    )
    def test_automatic_jax_available_zero_devices_fallback_to_vectorized(
        self, mock_devices_summary, mock_jax_available
    ) -> None:
        """
        Even if JAX is importable, if devices list is empty we treat it as 1 device
        and choose 'vectorized'.
        """
        cfg = m.make_sampling_config(strategy="automatic", target_accept=0.7)

        self.assertEqual(cfg.backend, "numpyro")
        self.assertEqual(cfg.chain_method, "vectorized")
        self.assertIsNone(cfg.cores)

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=False,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._cpu_core_count", return_value=16
    )
    def test_automatic_jax_not_available_falls_back_to_pymc(
        self, mock_core_count, mock_jax_available
    ) -> None:
        cfg = m.make_sampling_config(strategy="automatic", target_accept=0.95)

        self.assertEqual(cfg.backend, "pymc")
        self.assertIsNone(cfg.chain_method)
        self.assertEqual(cfg.cores, 16)
        self.assertEqual(cfg.target_accept, 0.95)
        self.assertIsNone(cfg.init)
        self.assertIsNone(cfg.max_treedepth)
        self.assertIsNone(cfg.nuts_kwargs)

        mock_jax_available.assert_called_once()
        mock_core_count.assert_called_once()


class TestMakeSamplingConfigExplicitStrategies(unittest.TestCase):
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=True,
    )
    def test_numpyro_parallel_with_jax_available(self, mock_jax_available) -> None:
        cfg = m.make_sampling_config(strategy="numpyro-parallel", target_accept=0.9)

        self.assertEqual(cfg.backend, "numpyro")
        self.assertEqual(cfg.chain_method, "parallel")
        self.assertIsNone(cfg.cores)
        self.assertEqual(cfg.target_accept, 0.9)
        mock_jax_available.assert_called_once()

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=False,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._cpu_core_count", return_value=8
    )
    def test_numpyro_parallel_without_jax_falls_back_to_pymc(
        self, mock_core_count, mock_jax_available
    ) -> None:
        cfg = m.make_sampling_config(strategy="numpyro-parallel", target_accept=0.9)

        self.assertEqual(cfg.backend, "pymc")
        self.assertIsNone(cfg.chain_method)
        self.assertEqual(cfg.cores, 8)
        self.assertEqual(cfg.target_accept, 0.9)
        mock_jax_available.assert_called_once()
        mock_core_count.assert_called_once()

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=True,
    )
    def test_numpyro_vectorized_with_jax_available(self, mock_jax_available) -> None:
        cfg = m.make_sampling_config(strategy="numpyro-vectorized", target_accept=0.75)

        self.assertEqual(cfg.backend, "numpyro")
        self.assertEqual(cfg.chain_method, "vectorized")
        self.assertIsNone(cfg.cores)
        self.assertEqual(cfg.target_accept, 0.75)
        mock_jax_available.assert_called_once()

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._jax_available",
        return_value=False,
    )
    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._cpu_core_count", return_value=4
    )
    def test_numpyro_vectorized_without_jax_falls_back_to_pymc(
        self, mock_core_count, mock_jax_available
    ) -> None:
        cfg = m.make_sampling_config(strategy="numpyro-vectorized", target_accept=0.88)

        self.assertEqual(cfg.backend, "pymc")
        self.assertIsNone(cfg.chain_method)
        self.assertEqual(cfg.cores, 4)
        self.assertEqual(cfg.target_accept, 0.88)
        mock_jax_available.assert_called_once()
        mock_core_count.assert_called_once()

    @patch(
        "biogeme.bayesian_estimation.sampling_strategy._cpu_core_count", return_value=12
    )
    def test_pymc_strategy_uses_cpu_core_count(self, mock_core_count) -> None:
        cfg = m.make_sampling_config(strategy="pymc", target_accept=0.99)

        self.assertEqual(cfg.backend, "pymc")
        self.assertIsNone(cfg.chain_method)
        self.assertEqual(cfg.cores, 12)
        self.assertEqual(cfg.target_accept, 0.99)
        mock_core_count.assert_called_once()


class TestMakeSamplingConfigMisc(unittest.TestCase):
    def test_strategy_is_case_and_whitespace_insensitive(self) -> None:
        """
        Strategy string should be trimmed and case-insensitive.
        """
        with patch(
            "biogeme.bayesian_estimation.sampling_strategy._jax_available",
            return_value=True,
        ), patch(
            "biogeme.bayesian_estimation.sampling_strategy._jax_devices_summary",
            return_value=(2, ["cpu"]),
        ):
            cfg1 = m.make_sampling_config(
                strategy="numpyro-parallel", target_accept=0.5
            )
            cfg2 = m.make_sampling_config(
                strategy="  NUmPyRO-PaRalLel  ", target_accept=0.5
            )

        self.assertEqual(cfg1.backend, cfg2.backend)
        self.assertEqual(cfg1.chain_method, cfg2.chain_method)
        self.assertEqual(cfg1.cores, cfg2.cores)
        self.assertEqual(cfg1.target_accept, cfg2.target_accept)

    def test_invalid_strategy_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            m.make_sampling_config(strategy="unknown-strategy", target_accept=0.5)

        msg = str(ctx.exception)
        self.assertIn("Unknown sampling strategy", msg)
        self.assertIn("automatic", msg)
        self.assertIn("numpyro-parallel", msg)
        self.assertIn("numpyro-vectorized", msg)
        self.assertIn("pymc", msg)


if __name__ == "__main__":
    unittest.main()
