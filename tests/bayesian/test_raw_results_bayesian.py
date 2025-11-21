import os
import tempfile
import unittest
from datetime import timedelta

import arviz as az
import numpy as np
import xarray as xr
from biogeme.bayesian_estimation.raw_bayesian_results import RawBayesianResults


def make_idata(chains: int = 2, draws: int = 5, n_obs: int = 7) -> az.InferenceData:
    """Minimal valid InferenceData with posterior + observed_data."""
    # posterior: scalar RV 'alpha' with dims (chain, draw)
    alpha = xr.DataArray(
        np.random.randn(chains, draws),
        dims=("chain", "draw"),
        coords={"chain": np.arange(chains), "draw": np.arange(draws)},
        name="alpha",
    )
    posterior = xr.Dataset({"alpha": alpha})

    # observed_data: vector 'y' with dim ('obs',)
    y = xr.DataArray(
        np.zeros(n_obs, dtype=int),
        dims=("obs",),
        coords={"obs": np.arange(n_obs)},
        name="y",
    )
    observed = xr.Dataset({"y": y})

    return az.InferenceData(posterior=posterior, observed_data=observed)


class TestRawBayesianResults(unittest.TestCase):
    # ---------- Basic properties ----------
    def test_basic_properties(self):
        idata = make_idata(chains=3, draws=11)
        r = RawBayesianResults(
            idata=idata,
            model_name="demo",
            user_notes="notes",
            data_name="datasetX",
            beta_names=["b0", "b1"],
            sampler="NUTS",
            target_accept=0.9,
            log_like_name='my_name',
            number_of_observations=123,
            run_time=timedelta(seconds=12.5),
        )
        self.assertEqual(r.model_name, "demo")
        self.assertEqual(r.log_like_name, "my_name")
        self.assertEqual(r.user_notes, "notes")
        self.assertEqual(r.data_name, "datasetX")
        self.assertEqual(r.beta_names, ["b0", "b1"])
        self.assertEqual(r.sampler, "NUTS")
        self.assertAlmostEqual(r.target_accept, 0.9)
        self.assertIsInstance(r.run_time, timedelta)
        self.assertEqual(r.chains, 3)
        self.assertEqual(r.draws, 11)
        self.assertEqual(r.number_of_observations, 123)

    # ---------- Persistence: save/load round trip ----------
    def test_save_and_load_round_trip(self):
        idata = make_idata(chains=2, draws=6, n_obs=10)
        r = RawBayesianResults(
            idata=idata,
            model_name="b01logit",
            user_notes="Swissmetro demo",
            data_name="swissmetro",
            beta_names=["ASC_TRAIN", "ASC_CAR", "B_TIME", "B_COST"],
            sampler="NUTS",
            log_like_name='log_like',
            number_of_observations=10,
            target_accept=0.85,
            run_time=timedelta(seconds=123.4),
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "posterior_bundle.nc")
            r.save(path)
            self.assertTrue(os.path.exists(path), "NetCDF file not written")

            r2 = RawBayesianResults.load(path)

            # Posterior structure preserved
            self.assertEqual(r2.chains, r.chains)
            self.assertEqual(r2.draws, r.draws)
            self.assertEqual(r2.number_of_observations, r.number_of_observations)

            # Metadata preserved
            self.assertEqual(r2.model_name, "b01logit")
            self.assertEqual(r2.log_like_name, "log_like")
            self.assertEqual(r2.user_notes, "Swissmetro demo")
            self.assertEqual(r2.data_name, "swissmetro")
            self.assertEqual(
                r2.beta_names, ["ASC_TRAIN", "ASC_CAR", "B_TIME", "B_COST"]
            )
            self.assertEqual(r2.sampler, "NUTS")
            self.assertAlmostEqual(r2.target_accept, 0.85)
            self.assertIsInstance(r2.run_time, timedelta)
            self.assertAlmostEqual(r2.run_time.total_seconds(), 123.4, places=6)
            self.assertEqual(r.number_of_observations, 10)

    # ---------- Persistence: load file with NO meta group ----------
    def test_load_file_without_meta_group_defaults(self):
        # Write an idata to nc WITHOUT adding biogeme_meta group
        idata = make_idata(chains=2, draws=3, n_obs=4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bare.nc")
            az.to_netcdf(idata, path)  # no meta group

            r2 = RawBayesianResults.load(path)

            # Posterior still loads
            self.assertEqual(r2.chains, 2)
            self.assertEqual(r2.draws, 3)
            self.assertEqual(r2.number_of_observations, 0)

            # Metadata should be defaulted (empty/None)
            self.assertEqual(r2.model_name, "")
            self.assertEqual(r2.user_notes, "")
            self.assertEqual(r2.data_name, "")
            self.assertEqual(r2.beta_names, [])
            self.assertIsNone(r2.sampler)
            self.assertIsNone(r2.target_accept)
            self.assertIsNone(r2.run_time)

    # ---------- Persistence: save with non-JSON-serializable beta_names ----------
    def test_save_with_non_json_serializable_metadata_graceful(self):
        # Force JSON encoding to fail by passing a non-serializable object in beta_names
        idata = make_idata(chains=1, draws=2, n_obs=3)
        non_serializable = object()  # will cause json.dumps to raise TypeError
        r = RawBayesianResults(
            idata=idata,
            model_name="should_default",
            user_notes="also_default",
            data_name="default",
            beta_names=["ok", non_serializable],  # not JSON-serializable
            sampler="NUTS",
            log_like_name='log_like',
            target_accept=0.99,
            number_of_observations=3,
            run_time=timedelta(seconds=1.0),
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bundle_bad_meta.nc")
            # save() should catch (AttributeError, KeyError, TypeError, ValueError)
            # and still write a file with an empty meta group.
            r.save(path)
            self.assertTrue(os.path.exists(path))

            r2 = RawBayesianResults.load(path)
            # Posterior loads
            self.assertEqual(r2.chains, 1)
            self.assertEqual(r2.draws, 2)
            self.assertEqual(r2.number_of_observations, 0)
            # Metadata defaulted due to failed JSON encoding
            self.assertEqual(r2.model_name, "")
            self.assertEqual(r2.user_notes, "")
            self.assertEqual(r2.data_name, "")
            self.assertEqual(r2.beta_names, [])
            self.assertIsNone(r2.sampler)
            self.assertIsNone(r2.target_accept)
            self.assertIsNone(r2.run_time)

    # ---------- Loading nonexistent file ----------
    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            RawBayesianResults.load("this_file_should_not_exist_abcdef.nc")

    # ---------- to_dict snapshot ----------
    def test_to_dict_snapshot(self):
        idata = make_idata(chains=2, draws=4, n_obs=5)
        r = RawBayesianResults(
            idata=idata,
            model_name="snap",
            user_notes="ok",
            data_name="data",
            beta_names=["b0", "b1"],
            sampler=None,
            log_like_name='log_like',
            number_of_observations=5,
            target_accept=None,
            run_time=None,
        )
        d = r.to_dict()
        self.assertEqual(
            set(d.keys()),
            {
                "model_name",
                "user_notes",
                "data_name",
                "number_of_observations",
                "log_like_name",
                "beta_names",
                "sampler",
                "chains",
                "draws",
                "target_accept",
                "run_time_seconds",
            },
        )
        self.assertEqual(d["model_name"], "snap")
        self.assertEqual(d["log_like_name"], "log_like")
        self.assertEqual(d["number_of_observations"], 5)
        self.assertEqual(d["chains"], 2)
        self.assertEqual(d["draws"], 4)
        self.assertIsNone(d["sampler"])
        self.assertIsNone(d["target_accept"])
        self.assertIsNone(d["run_time_seconds"])


if __name__ == "__main__":
    unittest.main()
