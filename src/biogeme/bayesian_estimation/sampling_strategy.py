"""
Defines the strategy for sampling the MCMC based on hardware configuration and user's preferences.

Michel Bierlaire
Mon Oct 27 2025, 17:01:13

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from biogeme.tools import report_jax_cpu_devices, warning_cpu_devices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplerPlan:
    """Plan describing how to run MCMC."""

    backend: str  # 'numpyro' or 'pymc'
    chain_method: str | None  # 'vectorized' | 'parallel' | None (ignored for pm.sample)
    cores: int | None
    note: str = ""

    # ---------- Constructors ----------

    @classmethod
    def from_name(cls, name: str) -> SamplerPlan:
        """Create a plan from a short human-friendly string."""
        key = name.strip().lower()
        if key in {"auto", "automatic"}:
            raise ValueError(
                "Use plan=None for automatic planning; 'auto' is not a concrete plan."
            )

        lut: dict[str, tuple[str, str | None, int | None]] = {
            "numpyro-parallel": ("numpyro", "parallel", None),
            "parallel": ("numpyro", "parallel", None),
            "numpyro-vectorized": ("numpyro", "vectorized", None),
            "vectorized": ("numpyro", "vectorized", None),
            "pymc": ("pymc", None, None),
            "pm": ("pymc", None, None),
        }
        try:
            backend, chain_method, cores = lut[key]
        except KeyError as e:
            raise ValueError(f"Unknown plan string: {name!r}") from e
        return cls(
            backend=backend, chain_method=chain_method, cores=cores, note="user-defined"
        )

    @classmethod
    def from_mapping(cls, spec: dict[str, Any]) -> SamplerPlan:
        """Create a plan from a dict, e.g. {'backend':'pymc','cores':4}."""
        backend = spec.get("backend")
        chain_method = spec.get("chain_method")
        cores = spec.get("cores")

        allowed_backends = {"numpyro", "pymc"}
        if backend not in allowed_backends:
            raise ValueError("User plan must include backend in {'numpyro','pymc'}.")

        if backend == "numpyro":
            allowed_methods = {"parallel", "vectorized", None}
            if chain_method not in allowed_methods:
                raise ValueError(
                    "For 'numpyro', chain_method must be 'parallel', 'vectorized', or None."
                )
        else:
            # 'pymc' ignores chain_method
            if chain_method is not None:
                logger.warning("Ignoring chain_method for 'pymc' backend.")
                chain_method = None

        return cls(
            backend=str(backend),
            chain_method=chain_method,
            cores=int(cores) if cores is not None else None,
            note="user-defined",
        )

    @classmethod
    def from_user_plan(cls, plan: SamplerPlan | str | dict[str, Any]) -> SamplerPlan:
        """Dispatch to the appropriate constructor (pattern matching)."""
        match plan:
            case SamplerPlan() as p:
                return p
            case str() as s:
                return cls.from_name(s)
            case dict() as d:
                return cls.from_mapping(d)
            case _:
                raise TypeError("User plan must be SamplerPlan, str, or dict.")


class SamplerPlanner:
    """Choose a good default configuration for running chains."""

    def __init__(
        self,
        *,
        number_of_threads: int,
        prefer_vectorized_single_device: bool = True,
        allow_pymc_multiprocessing_fallback: bool = False,
        max_vectorized_chains_on_gpu: int = 2,  # warn if vectorizing more than this on a single GPU
    ) -> None:
        """
        Initialize the sampling planner.

        Assumes JAX is already imported and therefore does NOT attempt to modify
        environment variables. It only reports devices and gives hints.
        """
        self.number_of_threads = number_of_threads
        self.prefer_vectorized_single_device = prefer_vectorized_single_device
        self.allow_pymc_multiprocessing_fallback = allow_pymc_multiprocessing_fallback
        self.max_vectorized_chains_on_gpu = max_vectorized_chains_on_gpu

        # Device reporting (non-intrusive)
        logger.info(report_jax_cpu_devices())
        warning_cpu_devices()

    def plan(self, chains: int) -> SamplerPlan:
        """Return an automatically chosen plan for how to run `chains`."""
        try:
            import jax
            from pymc.sampling.jax import sample_numpyro_nuts  # noqa: F401
            import numpyro  # noqa: F401

            devices = jax.devices()
            n_dev = len(devices)
            platforms = sorted({d.platform for d in devices})
            dev_summary = " | ".join(
                f"{d.platform}:{getattr(d, 'id', '?')}({getattr(d, 'device_kind', getattr(d, 'device', 'Device'))})"
                for d in devices
            )

            logger.info(
                "JAX detected %d device(s) across platform(s): %s. Chains requested: %d",
                n_dev,
                ", ".join(platforms) if platforms else "unknown",
                chains,
            )
            logger.info("Devices: %s", dev_summary if dev_summary else "n/a")

            # Hint: single CPU device but multiple chains
            if platforms == ["cpu"] and n_dev == 1 and chains > 1:
                logger.warning(
                    "Single CPU device detected with chains=%d. "
                    "To parallelize across CPU devices, set XLA_FLAGS before starting Python.",
                    chains,
                )

            # GPU VRAM guardrail: vectorizing many chains on a single GPU can OOM
            if (
                "gpu" in platforms
                and n_dev == 1
                and chains > self.max_vectorized_chains_on_gpu
            ):
                logger.warning(
                    "Single GPU detected with chains=%d; vectorizing many chains can exhaust VRAM. "
                    "Prefer parallel across multiple devices (if available), or reduce chains to <= %d.",
                    chains,
                    self.max_vectorized_chains_on_gpu,
                )

        except (ImportError, ModuleNotFoundError) as e:
            logger.info(
                "NumPyro/JAX backend not available (%s); using PyMC multiprocessing (cores=%d).",
                e,
                chains,
            )
            return SamplerPlan(
                backend="pymc",
                chain_method=None,
                cores=int(chains),
                note="No JAX+NumPyro; pm.sample cores",
            )

        # Multiple devices available → parallel across devices
        if n_dev >= int(chains) and chains > 1:
            return SamplerPlan(
                backend="numpyro",
                chain_method="parallel",
                cores=None,
                note=f"JAX devices={n_dev} >= chains={chains}; numpyro parallel",
            )

        # Single device → vectorized (usually fastest on a single GPU/TPU or CPU)
        if self.prefer_vectorized_single_device and n_dev >= 1:
            return SamplerPlan(
                backend="numpyro",
                chain_method="vectorized",
                cores=None,
                note=f"Single JAX device (n={n_dev}); numpyro vectorized",
            )

        # Optional: prefer 1 process per chain on CPU
        if self.allow_pymc_multiprocessing_fallback:
            return SamplerPlan(
                backend="pymc",
                chain_method=None,
                cores=int(chains),
                note="Fallback to pm.sample cores",
            )

        # Default
        return SamplerPlan(
            backend="numpyro",
            chain_method="vectorized",
            cores=None,
            note=f"Defaulting to numpyro vectorized (devices={n_dev})",
        )
