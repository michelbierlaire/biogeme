"""
Factory tools for managing random draw generation in Biogeme.

This module defines classes for encapsulating and orchestrating
the creation of random draws used in simulation-based estimation.

It distinguishes between native and user-defined draw generators,
validates inputs, and constructs a final tensor of draws with shape:
    (sample_size, number_of_draws, number_of_variables)

Michel Bierlaire
Wed Mar 26 19:30:36 2025
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from biogeme.exceptions import BiogemeError
from .native_draws import RandomNumberGeneratorTuple, native_random_number_generators


@dataclass
class DrawSpec:
    """
    Encapsulates the configuration for generating draws for a specific variable.

    :param name: Name of the variable that requires simulated draws.
    :param draw_type: Identifier for the type of draw (native or user-defined).
    :param generator: A callable that takes (sample_size, number_of_draws)
                      and returns a NumPy array of draws.
    """

    name: str
    draw_type: str
    generator: Callable[[int, int], np.ndarray]


class DrawFactory:
    """
    Manages native and user-defined random draw generators and builds
    draw specifications and arrays for multiple variables.

    This class is useful for transforming a mapping of variable names and draw types
    into a single array of random numbers ready for use in simulation-based estimation.
    """

    def __init__(self, user_generators: dict[str, RandomNumberGeneratorTuple] | None):
        """
        Initializes the manager and validates user-defined generators.

        :param user_generators: A dictionary of user-defined draw types.
        :raises ValueError: if any user generator name collides with native keywords.
        """
        self.native_generators = native_random_number_generators
        self.user_generators = (
            {k.upper(): v for k, v in user_generators.items()}
            if user_generators is not None
            else {}
        )
        self._validate_reserved_keywords()

    def _validate_reserved_keywords(self) -> None:
        """Ensure no user-defined generator overrides a native one."""
        for key in self.native_generators:
            if key in self.user_generators:
                raise ValueError(
                    f"{key} is a reserved keyword and cannot be used "
                    "as a user-defined draw generator name."
                )

    def get_generator(
        self, draw_type: str, name: str
    ) -> Callable[[int, int], np.ndarray]:
        """
        Retrieves a draw generator function based on the requested type.

        :param draw_type: Type identifier of the draw (e.g., 'UNIFORM', 'HALTON').
        :param name: Variable name (used for error context).
        :return: A function that generates a NumPy array of draws.
        :raises BiogemeError: if the draw type is not recognized.
        """
        key = draw_type.upper()
        if key in self.native_generators:
            return self.native_generators[key].generator
        if key in self.user_generators:
            return self.user_generators[key].generator

        raise BiogemeError(
            f"Unknown draw type '{draw_type}' for variable '{name}'. "
            f"Available native types: {list(self.native_generators.keys())}. "
            f"User-defined types: {list(self.user_generators.keys())}."
        )

    def make_draw_specs(
        self, draw_types: dict[str, str], variable_names: list[str]
    ) -> list[DrawSpec]:
        """
        Generates a list of DrawSpec objects for each variable.

        :param draw_types: Mapping from variable name to draw type.
        :param variable_names: List of variable names requiring simulated draws.
        :return: List of fully constructed DrawSpec objects.
        """
        return [
            DrawSpec(
                name=name,
                draw_type=draw_types[name],
                generator=self.get_generator(draw_types[name], name),
            )
            for name in variable_names
        ]

    def generate_draws(
        self,
        draw_types: dict[str, str],
        variable_names: list[str],
        sample_size: int,
        number_of_draws: int,
    ) -> np.ndarray:
        """
        Generates a 3D NumPy array of draws for all specified variables.

        :param draw_types: Mapping from variable name to draw type.
        :param variable_names: Ordered list of variable names.
        :param sample_size: Number of observations in the sample.
        :param number_of_draws: Number of Monte Carlo draws per observation.
        :return: A NumPy array of shape (sample_size, number_of_draws, len(variable_names)).
        :raises BiogemeError: if any generator returns a mis-shaped array.
        """
        specs = self.make_draw_specs(draw_types, variable_names)
        draws = []
        for spec in specs:
            array = spec.generator(sample_size, number_of_draws)
            self._check_shape(spec.name, array, sample_size, number_of_draws)
            draws.append(array)
        return np.moveaxis(np.array(draws), 0, -1)

    def _check_shape(self, name, array, sample_size, number_of_draws):
        """
        Verifies that the shape of a generated draw array matches expectations.

        :param name: Variable name.
        :param array: Array returned by the draw generator.
        :param sample_size: Expected number of rows.
        :param number_of_draws: Expected number of columns.
        :raises BiogemeError: if the shape is invalid.
        """
        if array.shape != (sample_size, number_of_draws):
            raise BiogemeError(
                f"Draws for '{name}' must have shape ({sample_size}, {number_of_draws}), "
                f"but got {array.shape}."
            )
