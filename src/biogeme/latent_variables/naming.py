from __future__ import annotations

"""Naming policy for resolved model objects."""

from dataclasses import dataclass
from typing import Protocol


class NamingPolicy(Protocol):
    def structural_prefix(self, latent_name: str) -> str: ...
    def structural_beta_name(self, latent_name: str, variable_name: str) -> str: ...
    def structural_intercept_name(self, latent_name: str) -> str: ...
    def structural_sigma_name(self, latent_name: str) -> str: ...
    def structural_draw_name(self, latent_name: str) -> str: ...
    def measurement_prefix(self, indicator_name: str) -> str: ...
    def measurement_intercept_name(self, indicator_name: str) -> str: ...
    def measurement_loading_name(
        self, latent_name: str, indicator_name: str
    ) -> str: ...
    def measurement_sigma_name(self, indicator_name: str) -> str: ...
    def threshold_tau1_name(self, type_name: str) -> str: ...
    def threshold_delta_name(self, type_name: str, index: int) -> str: ...


@dataclass(frozen=True, slots=True)
class DefaultNamingPolicy:
    def structural_prefix(self, latent_name: str) -> str:
        return f'struct_{latent_name}'

    def structural_beta_name(self, latent_name: str, variable_name: str) -> str:
        return f'{self.structural_prefix(latent_name)}_{variable_name}'

    def structural_intercept_name(self, latent_name: str) -> str:
        return f'{self.structural_prefix(latent_name)}_intercept'

    def structural_sigma_name(self, latent_name: str) -> str:
        return f'{self.structural_prefix(latent_name)}_sigma'

    def structural_draw_name(self, latent_name: str) -> str:
        return f'{self.structural_prefix(latent_name)}_draws'

    def measurement_prefix(self, indicator_name: str) -> str:
        return f'measurement_{indicator_name}'

    def measurement_intercept_name(self, indicator_name: str) -> str:
        return f'measurement_intercept_{indicator_name}'

    def measurement_loading_name(self, latent_name: str, indicator_name: str) -> str:
        return f'measurement_coefficient_{latent_name}_{indicator_name}'

    def measurement_sigma_name(self, indicator_name: str) -> str:
        return f'{self.measurement_prefix(indicator_name)}_sigma'

    def threshold_tau1_name(self, type_name: str) -> str:
        return f'{type_name}_tau_1'

    def threshold_delta_name(self, type_name: str, index: int) -> str:
        return f'{type_name}_delta_{index}'
