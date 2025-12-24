"""
Provides a suite of native random number generators for use in simulation-based estimation.
Includes uniform, Halton, Latin Hypercube, and normal draws, with optional antithetic and symmetric variants.

Michel Bierlaire
Thu Mar 27 09:46:17 2025
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .generators import (
    get_antithetic,
    get_halton_draws,
    get_latin_hypercube_draws,
    get_normal_wichura_draws,
    get_uniform,
)

RandomNumberGenerator = Callable[[int, int], np.ndarray]


class RandomNumberGeneratorTuple(NamedTuple):
    generator: RandomNumberGenerator
    description: str

    @classmethod
    def from_tuple(
        cls, the_tuple: tuple[RandomNumberGenerator, str]
    ) -> RandomNumberGeneratorTuple:
        """
        Create a RandomNumberGeneratorTuple from a legacy tuple.

        :param the_tuple: A tuple of (generator function, description).
        :return: A RandomNumberGeneratorTuple instance.
        """
        return cls(*the_tuple)


def convert_random_generator_tuple(
    the_tuple: RandomNumberGeneratorTuple | tuple[RandomNumberGenerator, str],
) -> RandomNumberGeneratorTuple:
    """
    Convert a random generator specification to a RandomNumberGeneratorTuple.

    :param the_tuple: Either an instance of RandomNumberGeneratorTuple or a legacy tuple.
    :return: A properly typed RandomNumberGeneratorTuple.
    """
    if isinstance(the_tuple, RandomNumberGeneratorTuple):
        return the_tuple
    if not isinstance(the_tuple, tuple):
        raise TypeError(f'Expecting a tuple, not a {type(the_tuple)}')

    return RandomNumberGeneratorTuple.from_tuple(the_tuple)


def uniform_antithetic(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic uniform random draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_antithetic(get_uniform, sample_size, number_of_draws)


def halton2(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate Halton draws with base 2.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(sample_size, number_of_draws, base=2, skip=10)


def halton3(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate Halton draws with base 3.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(sample_size, number_of_draws, base=3, skip=10)


def halton5(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate Halton draws with base 5.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(sample_size, number_of_draws, base=5, skip=10)


def MLHS_anti(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic Modified Latin Hypercube Sampling draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_antithetic(get_latin_hypercube_draws, sample_size, number_of_draws)


def symm_uniform(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate symmetric uniform random draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_uniform(sample_size, number_of_draws, symmetric=True)


def symm_uniform_antithetic(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic symmetric uniform random draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    number_local_draws = int(number_of_draws / 2)
    local_draws = symm_uniform(sample_size, number_local_draws)
    return np.concatenate((local_draws, -local_draws), axis=1)


def symm_halton2(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate symmetric Halton draws with base 2.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(
        sample_size, number_of_draws, symmetric=True, base=2, skip=10
    )


def symm_halton3(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate symmetric Halton draws with base 3.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(
        sample_size, number_of_draws, symmetric=True, base=3, skip=10
    )


def symm_halton5(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate symmetric Halton draws with base 5.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_halton_draws(
        sample_size, number_of_draws, symmetric=True, base=5, skip=10
    )


def symm_MLHS(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate symmetric Modified Latin Hypercube Sampling draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_latin_hypercube_draws(sample_size, number_of_draws, symmetric=True)


def symm_MLHS_anti(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic symmetric Modified Latin Hypercube Sampling draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    number_local_draws = int(number_of_draws / 2)
    local_draws = symm_MLHS(sample_size, number_local_draws)
    return np.concatenate((local_draws, -local_draws), axis=1)


def normal_antithetic(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic normal random draws.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    return get_normal_wichura_draws(
        sample_size=sample_size,
        number_of_draws=number_of_draws,
        antithetic=True,
    )


def normal_halton2(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate normal draws from Halton base 2 sequence.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    unif = get_halton_draws(sample_size, number_of_draws, base=2, skip=10)
    return get_normal_wichura_draws(
        sample_size,
        number_of_draws,
        uniform_numbers=unif,
        antithetic=False,
    )


def normal_halton3(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate normal draws from Halton base 3 sequence.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    unif = get_halton_draws(sample_size, number_of_draws, base=3, skip=10)
    return get_normal_wichura_draws(
        sample_size,
        number_of_draws,
        uniform_numbers=unif,
        antithetic=False,
    )


def normal_halton5(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate normal draws from Halton base 5 sequence.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    unif = get_halton_draws(sample_size, number_of_draws, base=5, skip=10)
    return get_normal_wichura_draws(
        sample_size,
        number_of_draws,
        uniform_numbers=unif,
        antithetic=False,
    )


def normal_MLHS(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate normal draws from Modified Latin Hypercube Sampling.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    unif = get_latin_hypercube_draws(sample_size, number_of_draws)
    return get_normal_wichura_draws(
        sample_size,
        number_of_draws,
        uniform_numbers=unif,
        antithetic=False,
    )


def normal_MLHS_anti(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Generate antithetic normal draws from Modified Latin Hypercube Sampling.

    :param sample_size: Number of individuals or observations.
    :param number_of_draws: Number of draws per observation.
    :return: A NumPy array of shape (sample_size, number_of_draws).
    """
    unif = get_latin_hypercube_draws(sample_size, int(number_of_draws / 2.0))
    return get_normal_wichura_draws(
        sample_size, number_of_draws, uniform_numbers=unif, antithetic=True
    )


# Dictionary containing native random number generators. Class attribute
native_random_number_generators = {
    'UNIFORM': RandomNumberGeneratorTuple(
        generator=get_uniform, description='Uniform U[0, 1]'
    ),
    'UNIFORM_ANTI': RandomNumberGeneratorTuple(
        generator=uniform_antithetic, description='Antithetic uniform U[0, 1]'
    ),
    'UNIFORM_HALTON2': RandomNumberGeneratorTuple(
        generator=halton2,
        description='Halton draws with base 2, skipping the first 10',
    ),
    'UNIFORM_HALTON3': RandomNumberGeneratorTuple(
        generator=halton3,
        description='Halton draws with base 3, skipping the first 10',
    ),
    'UNIFORM_HALTON5': RandomNumberGeneratorTuple(
        generator=halton5,
        description='Halton draws with base 5, skipping the first 10',
    ),
    'UNIFORM_MLHS': RandomNumberGeneratorTuple(
        generator=get_latin_hypercube_draws,
        description='Modified Latin Hypercube Sampling on [0, 1]',
    ),
    'UNIFORM_MLHS_ANTI': RandomNumberGeneratorTuple(
        generator=MLHS_anti,
        description='Antithetic Modified Latin Hypercube Sampling on [0, 1]',
    ),
    'UNIFORMSYM': RandomNumberGeneratorTuple(
        generator=symm_uniform, description='Uniform U[-1, 1]'
    ),
    'UNIFORMSYM_ANTI': RandomNumberGeneratorTuple(
        generator=symm_uniform_antithetic,
        description='Antithetic uniform U[-1, 1]',
    ),
    'UNIFORMSYM_HALTON2': RandomNumberGeneratorTuple(
        generator=symm_halton2,
        description='Halton draws on [-1, 1] with base 2, skipping the first 10',
    ),
    'UNIFORMSYM_HALTON3': RandomNumberGeneratorTuple(
        generator=symm_halton3,
        description='Halton draws on [-1, 1] with base 3, skipping the first 10',
    ),
    'UNIFORMSYM_HALTON5': RandomNumberGeneratorTuple(
        generator=symm_halton5,
        description='Halton draws on [-1, 1] with base 5, skipping the first 10',
    ),
    'UNIFORMSYM_MLHS': RandomNumberGeneratorTuple(
        generator=symm_MLHS,
        description='Modified Latin Hypercube Sampling on [-1, 1]',
    ),
    'UNIFORMSYM_MLHS_ANTI': RandomNumberGeneratorTuple(
        generator=symm_MLHS_anti,
        description='Antithetic Modified Latin Hypercube Sampling on [-1, 1]',
    ),
    'NORMAL': RandomNumberGeneratorTuple(
        generator=get_normal_wichura_draws, description='Normal N(0, 1) draws'
    ),
    'NORMAL_ANTI': RandomNumberGeneratorTuple(
        generator=normal_antithetic, description='Antithetic normal draws'
    ),
    'NORMAL_HALTON2': RandomNumberGeneratorTuple(
        generator=normal_halton2,
        description='Normal draws from Halton base 2 sequence',
    ),
    'NORMAL_HALTON3': RandomNumberGeneratorTuple(
        generator=normal_halton3,
        description='Normal draws from Halton base 3 sequence',
    ),
    'NORMAL_HALTON5': RandomNumberGeneratorTuple(
        generator=normal_halton5,
        description='Normal draws from Halton base 5 sequence',
    ),
    'NORMAL_MLHS': RandomNumberGeneratorTuple(
        generator=normal_MLHS,
        description='Normal draws from Modified Latin Hypercube Sampling',
    ),
    'NORMAL_MLHS_ANTI': RandomNumberGeneratorTuple(
        generator=normal_MLHS_anti,
        description='Antithetic normal draws from Modified Latin Hypercube Sampling',
    ),
}

assert all(k == k.upper() for k in native_random_number_generators)


def description_of_native_draws() -> dict[str, str]:
    """
    Provides a dictionary of all native draw types and their descriptions.

    :return: Dictionary where keys are generator names and values are human-readable descriptions.
    """
    return {
        key: the_tuple[1] for key, the_tuple in native_random_number_generators.items()
    }
