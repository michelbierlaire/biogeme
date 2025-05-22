"""
This module defines the Draws class, which manages the generation and
conversion of random draws for use in simulation-based models.

Michel Bierlaire
Thu Mar 27 08:42:16 2025
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import jax.numpy as jnp
import numpy as np
import pandas as pd

from biogeme.floating_point import JAX_FLOAT
from .factory import DrawFactory
from .native_draws import RandomNumberGeneratorTuple

LOW_NUMBER_OF_DRAWS = 1000
logger = logging.getLogger(__name__)


class DrawsManagement:
    """
    Manages generation of simulation draws and conversion to JAX-compatible format.
    """

    def __init__(
        self,
        sample_size: int,
        number_of_draws: int,
        user_generators: dict[str, RandomNumberGeneratorTuple] | None = None,
    ):
        """
        Constructor for the Draws class.

        :param sample_size: The number of observations (rows) in the sample.
        :param number_of_draws: The number of draws to generate per observation.
        :param user_generators: Optional dictionary of user-defined random number generators.
        """
        if sample_size <= 0:
            raise ValueError(f'Incorrect sample size: {sample_size}')

        self.user_generators = user_generators
        self.sample_size: int = sample_size
        self.number_of_draws: int = number_of_draws
        self.factory: DrawFactory = DrawFactory(user_generators)
        self.draws: np.ndarray | None = None
        self.draw_types: dict[str, str] | None = None
        self.processing_time: timedelta = timedelta(0)

    def generate_draws(
        self,
        draw_types: dict[str, str],
        variable_names: list[str],
    ) -> None:
        """
        Generates random draws using the configured factory.

        :param draw_types: Mapping of variable names to draw types.
        :param variable_names: List of variable names requiring draws.
        :return: The generated draws as a NumPy array.
        """
        self.draw_types = draw_types
        if self.number_of_draws <= 0:
            raise ValueError(f'Incorrect number of draws: {self.number_of_draws}')
        if self.number_of_draws <= LOW_NUMBER_OF_DRAWS:
            warning_msg = f'The number of draws ({self.number_of_draws}) is low. The results may not be meaningful.'
            logger.warning(warning_msg)
        start_time = datetime.now()
        self.draws = self.factory.generate_draws(
            draw_types=draw_types,
            variable_names=variable_names,
            sample_size=self.sample_size,
            number_of_draws=self.number_of_draws,
        )
        self.processing_time = datetime.now() - start_time

    @property
    def draws_jax(self) -> jnp.ndarray:
        """
        Returns the generated draws as a JAX array.
        If no draws have been generated, returns an empty JAX array of shape (sample_size, 1, 1).

        :return: JAX-compatible array of draws.
        """
        if self.draws is not None:
            return jnp.asarray(self.draws, dtype=JAX_FLOAT)
        return jnp.zeros((self.sample_size, 1, 1), dtype=JAX_FLOAT)

    def extract_slice(self, indices: pd.Index) -> DrawsManagement:
        """
        Create a new DrawsManagement instance containing only a subset of draws.

        This is useful to maintain consistency across estimation and validation datasets by slicing
        the original draws array according to the provided indices.

        :param indices: The indices used to extract the subset of draws.
        :return: A new DrawsManagement instance containing the sliced draws.
        """
        sliced_draw_management: DrawsManagement = DrawsManagement(
            sample_size=len(indices),
            number_of_draws=self.number_of_draws,
            user_generators=self.user_generators,
        )
        sliced_draw_management.draws = (
            self.draws[indices] if self.draws is not None else None
        )
        return sliced_draw_management

    def remove_rows(self, indices: pd.Index) -> None:
        """Remove rows. Typically called when the database is modified."""
        if self.draws is None:
            return
        self.draws = self.draws[indices]
