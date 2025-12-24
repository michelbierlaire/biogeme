"""
Raw Bayesian estimation results
built from ArviZ InferenceData (PyMC).

Michel Bierlaire
Mon Oct 20 2025, 17:18:07
"""

from __future__ import annotations

import contextlib
import logging
from datetime import timedelta

import arviz as az
import xarray as xr
from biogeme.tools import print_file_size, timeit

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NetCDF-only RawBayesianResults class
# -----------------------------------------------------------------------------
class RawBayesianResults:
    """
    Minimal, NetCDF-only container of Bayesian estimation results.

    This class *holds* an ArviZ InferenceData (PyMC posterior, etc.) and a
    handful of metadata that cannot be robustly deduced from the InferenceData.

    - No YAML sidecar is produced.
    - All information is stored in a single NetCDF file via :meth:`save`.
    - To reload, use :meth:`load` which reads both posterior and metadata
      from the same NetCDF file.

    Stored metadata (beyond what can be inferred from idata):
      * model_name (str)
      * user_notes (str)
      * data_name (str)
      * beta_names (list[str])  # model free/fixed parameter names for reporting
      * sampler (str | None)
      * target_accept (float | None)
      * random_seed (int | None)
      * run_time (timedelta | None)
    """

    _META_GROUP = "biogeme_meta"

    def __init__(
        self,
        *,
        idata: az.InferenceData,
        model_name: str,
        log_like_name: str,
        number_of_observations: int,
        user_notes: str = "",
        data_name: str = "",
        beta_names: list[str] | None = None,
        sampler: str | None = None,
        target_accept: float | None = None,
        run_time: timedelta | None = None,
    ) -> None:
        self._idata = idata
        self._model_name = model_name
        self._log_like_name = log_like_name
        self._user_notes = user_notes
        self._data_name = data_name
        self._beta_names = beta_names or []
        self._sampler = sampler
        self._target_accept = target_accept
        self._run_time = run_time
        self._number_of_observations = number_of_observations

    # ---------------- Properties inferred from idata ----------------
    @property
    def idata(self) -> az.InferenceData:
        return self._idata

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def log_like_name(self) -> str:
        return self._log_like_name

    @property
    def user_notes(self) -> str:
        return self._user_notes

    @property
    def data_name(self) -> str:
        return self._data_name

    @property
    def beta_names(self) -> list[str]:
        return list(self._beta_names)

    @property
    def sampler(self) -> str | None:
        return self._sampler

    @property
    def target_accept(self) -> float | None:
        return self._target_accept

    @property
    def run_time(self) -> timedelta | None:
        return self._run_time

    @property
    def chains(self) -> int:
        try:
            return int(self._idata.posterior.sizes.get("chain", 1))
        except (AttributeError, KeyError, TypeError, ValueError):
            return 1

    @property
    def draws(self) -> int:
        try:
            return int(self._idata.posterior.sizes.get("draw", 0))
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0

    @property
    def number_of_observations(self) -> int:
        return self._number_of_observations

    # ---------------- Persistence (single NetCDF) ----------------
    def _metadata_dataset(self) -> xr.Dataset:
        """Build a tiny xarray Dataset to store metadata as attributes."""
        ds = xr.Dataset()
        ds.attrs["model_name"] = self._model_name
        ds.attrs["user_notes"] = self._user_notes
        ds.attrs["data_name"] = self._data_name
        ds.attrs["log_like_name"] = self._log_like_name
        ds.attrs["number_of_observations"] = self._number_of_observations
        ds.attrs["beta_names"] = list(self._beta_names)
        ds.attrs["sampler"] = self._sampler if self._sampler is not None else ""
        ds.attrs["target_accept"] = (
            float(self._target_accept)
            if self._target_accept is not None
            else float("nan")
        )
        ds.attrs["run_time_seconds"] = (
            float(self._run_time.total_seconds())
            if self._run_time is not None
            else float("nan")
        )
        return ds

    def save(self, path: str) -> None:
        """Write a single NetCDF file with posterior + metadata."""
        logger.info(f'Save simulation results on {path}')
        idata = self._idata.copy()
        # attach (or replace) the metadata group
        try:
            import json

            meta_ds = xr.Dataset()
            meta_ds.attrs.update(
                {
                    "model_name": self._model_name or "",
                    "user_notes": self._user_notes or "",
                    "data_name": self._data_name or "",
                    "log_like_name": self._log_like_name or "",
                    "number_of_observations": self._number_of_observations or "",
                    "beta_names": json.dumps(self._beta_names or []),
                    "sampler": self._sampler or "",
                    "target_accept": (
                        self._target_accept if self._target_accept is not None else ""
                    ),
                    "run_time_seconds": (
                        self._run_time.total_seconds()
                        if self._run_time is not None
                        else ""
                    ),
                }
            )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not JSON-encode metadata cleanly: %s", e)

        # Mirror metadata on a standard group so it survives az.from_netcdf
        try:
            import json as _json

            posterior_attrs = {
                "model_name": self._model_name or "",
                "user_notes": self._user_notes or "",
                "data_name": self._data_name or "",
                "log_like_name": self._log_like_name or "",
                "number_of_observations": self._number_of_observations or "",
                "beta_names": _json.dumps(self._beta_names or []),
                "sampler": self._sampler or "",
                "target_accept": (
                    self._target_accept if self._target_accept is not None else ""
                ),
                "run_time_seconds": (
                    self._run_time.total_seconds() if self._run_time is not None else ""
                ),
            }
            idata.posterior.attrs.update(posterior_attrs)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not set posterior attrs metadata cleanly: %s", e)
        az.to_netcdf(idata, path, engine="h5netcdf")
        logger.info(f'Saved Bayesian results (posterior + metadata) to {path}')

    @classmethod
    @timeit(label='load')
    def load(cls, path: str) -> RawBayesianResults:
        """
        Load from a single NetCDF file written by :meth:`save`.

        Metadata are read from ``idata.posterior.attrs``, where they were
        stored by :meth:`save`. No custom ``biogeme_meta`` group is used
        anymore.
        """
        logger.debug(f"Read file {path}")
        # On Windows, NetCDF backends may keep the file handle open via xarray's
        # file-manager cache. We therefore (i) reduce/disable caching when possible,
        # (ii) eagerly load all groups into memory, and (iii) close datasets.
        # Some xarray versions reject file_cache_maxsize=0, so we fall back safely.
        try:
            cache_ctx = xr.set_options(file_cache_maxsize=1)
        except Exception:
            cache_ctx = contextlib.nullcontext()

        with cache_ctx:
            idata = az.from_netcdf(path, engine="h5netcdf")

        # Detach from disk: load all datasets and close any open file handles.
        # This makes it safe to delete the NetCDF file immediately after loading.
        try:
            for group_name in idata.groups():
                ds = getattr(idata, group_name, None)
                if ds is None:
                    continue
                # Ensure arrays are in memory (not lazy on-disk)
                try:
                    ds.load()
                except Exception:
                    pass
                # Close backend resources if supported
                try:
                    ds.close()
                except Exception:
                    pass
        except Exception:
            # If anything goes wrong, keep behavior backward compatible.
            pass
        # Best-effort: clear xarray's global file cache to ensure no lingering handles.
        try:
            from xarray.backends.file_manager import FILE_CACHE

            FILE_CACHE.clear()
        except Exception:
            pass
        logger.info(f"Loaded NetCDF file size: {print_file_size(path)}")
        # Defaults
        model_name = ""
        user_notes = ""
        log_like_name = ""
        number_of_observations: int = 0
        data_name = ""
        beta_names: list[str] = []
        sampler: str | None = None

        # Read from posterior attrs if available
        try:
            p_attrs = idata.posterior.attrs
        except AttributeError as e:
            logger.info(
                "Posterior group missing or invalid in InferenceData loaded from %s: %s",
                path,
                e,
            )
            p_attrs = {}

        import json as _json

        model_name = p_attrs.get("model_name", model_name)
        user_notes = p_attrs.get("user_notes", user_notes)
        data_name = p_attrs.get("data_name", data_name)
        log_like_name = p_attrs.get("log_like_name", log_like_name)

        # number_of_observations may come as str, int, or be missing
        no_raw = p_attrs.get("number_of_observations", number_of_observations)
        try:
            number_of_observations = int(no_raw)
        except (TypeError, ValueError):
            number_of_observations = 0

        # beta_names stored as JSON string
        beta_names_raw = p_attrs.get("beta_names", "[]")
        try:
            beta_names = _json.loads(beta_names_raw) or beta_names
        except (TypeError, ValueError):
            beta_names = []

        sampler = p_attrs.get("sampler") or sampler

        ta = p_attrs.get("target_accept")
        try:
            target_accept = float(ta) if ta not in (None, "") else None
        except (TypeError, ValueError):
            target_accept = None

        rts = p_attrs.get("run_time_seconds")
        try:
            run_time = (
                timedelta(seconds=float(rts))
                if rts not in (None, "", float("nan"))
                else None
            )
        except (TypeError, ValueError):
            run_time = None

        return cls(
            idata=idata,
            model_name=model_name,
            user_notes=user_notes,
            log_like_name=log_like_name,
            number_of_observations=number_of_observations,
            data_name=data_name,
            beta_names=beta_names,
            sampler=sampler,
            target_accept=target_accept,
            run_time=run_time,
        )

    # Convenience dict for quick reporting
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "user_notes": self.user_notes,
            "data_name": self.data_name,
            "log_like_name": self.log_like_name,
            "number_of_observations": self.number_of_observations,
            "beta_names": self.beta_names,
            "sampler": self.sampler,
            "chains": self.chains,
            "draws": self.draws,
            "target_accept": self.target_accept,
            "run_time_seconds": (
                self.run_time.total_seconds() if self.run_time is not None else None
            ),
        }
