"""
Naming policy for builders.

All parameter names and prefixes used by builders are centralized here, so that:
- specification objects remain pure (no naming logic),
- normalization plans remain semantic (ParameterRef objects, not strings),
- users can customize naming conventions by providing their own NamingPolicy.

Michel Bierlaire
Thu Mar 05 2026, 11:19:25
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class NamingPolicy(Protocol):
    """Protocol defining how builders name parameters and other objects."""

    # ---- structural equation naming ----
    def structural_prefix(self, latent_name: str) -> str: ...
    def structural_beta_name(
        self, latent_name: str, explanatory_variable: str
    ) -> str: ...
    def structural_draws_name(self, latent_name: str) -> str: ...
    def structural_sigma_prefix(self, latent_name: str) -> str: ...

    # ---- measurement equation naming ----
    def measurement_prefix(self, indicator_name: str) -> str: ...
    def measurement_intercept_name(self, indicator_name: str) -> str: ...
    def measurement_loading_name(
        self, latent_name: str, indicator_name: str
    ) -> str: ...
    def measurement_sigma_prefix(self, indicator_name: str) -> str: ...

    # ---- threshold system naming ----
    def threshold_tau1_name(self, type_name: str) -> str: ...
    def threshold_delta_prefix(self, type_name: str) -> str: ...
    def threshold_delta_name(self, type_name: str, k: int) -> str: ...


@dataclass(frozen=True, slots=True)
class DefaultNamingPolicy:
    """Default naming policy matching the current conventions."""

    structural_root: str = "struct"
    measurement_root: str = "measurement"

    # ---- structural ----
    def structural_prefix(self, latent_name: str) -> str:
        return f"{self.structural_root}_{latent_name}"

    def structural_beta_name(self, latent_name: str, explanatory_variable: str) -> str:
        return f"{self.structural_prefix(latent_name)}_{explanatory_variable}"

    def structural_draws_name(self, latent_name: str) -> str:
        return f"{self.structural_prefix(latent_name)}_draws"

    def structural_sigma_prefix(self, latent_name: str) -> str:
        # sigma_factory(prefix=...) expects a prefix; by default reuse the structural prefix
        return self.structural_prefix(latent_name)

    # ---- measurement ----
    def measurement_prefix(self, indicator_name: str) -> str:
        return f"{self.measurement_root}_{indicator_name}"

    def measurement_intercept_name(self, indicator_name: str) -> str:
        return f"{self.measurement_root}_intercept_{indicator_name}"

    def measurement_loading_name(self, latent_name: str, indicator_name: str) -> str:
        return f"{self.measurement_root}_coefficient_{latent_name}_{indicator_name}"

    def measurement_sigma_prefix(self, indicator_name: str) -> str:
        return self.measurement_prefix(indicator_name)

    # ---- thresholds ----
    def threshold_tau1_name(self, type_name: str) -> str:
        return f"{type_name}_tau_1"

    def threshold_delta_prefix(self, type_name: str) -> str:
        # prefix used by positive_parameter_factory(prefix=..., name=...)
        return type_name

    def threshold_delta_name(self, type_name: str, k: int) -> str:
        # name used by positive_parameter_factory(prefix=..., name=...)
        return f"delta_{k}"
