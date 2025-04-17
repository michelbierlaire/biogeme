OBSOLETE

"""Generation of the draws for each observation"""

from collections.abc import Callable

import numpy as np
from dataclasses import dataclass
from biogeme.factory import DrawGeneratorManager

from .native_draws import (
    RandomNumberGeneratorTuple,
    native_random_number_generators,
    RandomNumberGenerator,
)
from ..exceptions import BiogemeError


@dataclass
class DrawSpec:
    """
    Encapsulates the configuration for generating draws for a specific variable.

    :param name: Name of the variable that requires simulated draws.
    :param draw_type: Identifier for the type of draw (native or user-defined).
    :param generator: A callable that takes (sample_size, number_of_draws) and returns a numpy array of draws.
    """

    name: str
    draw_type: str
    generator: Callable[[int, int], np.ndarray]


def _validate_reserved_keywords(
    user_generators: dict[str, RandomNumberGeneratorTuple]
) -> None:
    """
    Ensures that user-defined generators do not use reserved names.

    :param user_generators: Dictionary of user-defined generator tuples.
    :raise ValueError: if a user-defined generator uses a reserved keyword.
    """
    for k in native_random_number_generators:
        if k in user_generators:
            raise ValueError(
                f'{k} is a reserved keyword for draws'
                f' and cannot be used for user-defined generators'
            )


def _get_draw_generator(
    draw_type: str,
    name: str,
    user_generators: dict[str, RandomNumberGeneratorTuple],
) -> RandomNumberGenerator:
    """
    Retrieves the appropriate draw generator function based on type.

    :param draw_type: Type of draw, either native or user-defined.
    :param name: Name of the variable requiring draws.
    :param user_generators: Dictionary of user-defined generators.
    :return: Callable that generates the draws.
    :raise BiogemeError: if the draw type is unknown.
    """
    native_gen = native_random_number_generators.get(draw_type)
    if native_gen is not None:
        return native_gen.generator
    user_gen = user_generators.get(draw_type)
    if user_gen is not None:
        return user_gen.generator
    raise BiogemeError(
        f'Unknown type of draws for variable {name}: {draw_type}. '
        f'Native types: {list(native_random_number_generators.keys())}. '
        f'User defined: {list(user_generators.keys())}'
    )


def _check_draw_shape(
    name: str,
    array: np.ndarray,
    sample_size: int,
    number_of_draws: int,
) -> None:
    """
    Validates that a draw array has the expected shape.

    :param name: Name of the variable.
    :param array: The array of generated draws.
    :param sample_size: Expected number of rows (observations).
    :param number_of_draws: Expected number of columns (draws).
    :raise BiogemeError: if the array shape is incorrect.
    """
    if array.shape != (sample_size, number_of_draws):
        raise BiogemeError(
            f'The draw generator for {name} must generate a numpy array of shape '
            f'({sample_size}, {number_of_draws}) instead of {array.shape}'
        )


def generate_draws(
    draw_types: dict[str, str],
    names: list[str],
    sample_size: int,
    number_of_draws: int,
    user_random_number_generators: dict[str, RandomNumberGeneratorTuple],
) -> np.ndarray:
    """Generate draws for each variable.

    :param draw_types: A dict indexed by the names of the variables,
                  describing the draws. Each of them can
                  be a native type or any type defined by the
                  function
                  :func:`~biogeme.database.Database.setRandomNumberGenerators`.

                  Native types:

                  - ``'UNIFORM'``: Uniform U[0, 1],
                  - ``'UNIFORM_ANTI``: Antithetic uniform U[0, 1]',
                  - ``'UNIFORM_HALTON2'``: Halton draws with base 2,
                    skipping the first 10,
                  - ``'UNIFORM_HALTON3'``: Halton draws with base 3,
                    skipping the first 10,
                  - ``'UNIFORM_HALTON5'``: Halton draws with base 5,
                    skipping  the first 10,
                  - ``'UNIFORM_MLHS'``: Modified Latin Hypercube
                    Sampling on [0, 1],
                  - ``'UNIFORM_MLHS_ANTI'``: Antithetic Modified
                    Latin Hypercube Sampling on [0, 1],
                  - ``'UNIFORMSYM'``: Uniform U[-1, 1],
                  - ``'UNIFORMSYM_ANTI'``: Antithetic uniform U[-1, 1],
                  - ``'UNIFORMSYM_HALTON2'``: Halton draws on [-1, 1]
                    with base 2, skipping the first 10,
                  - ``'UNIFORMSYM_HALTON3'``: Halton draws on [-1, 1]
                    with base 3, skipping the first 10,
                  - ``'UNIFORMSYM_HALTON5'``: Halton draws on [-1, 1]
                    with base 5, skipping the first 10,
                  - ``'UNIFORMSYM_MLHS'``: Modified Latin Hypercube
                    Sampling on [-1, 1],
                  - ``'UNIFORMSYM_MLHS_ANTI'``: Antithetic Modified
                    Latin Hypercube Sampling on [-1, 1],
                  - ``'NORMAL'``: Normal N(0, 1) draws,
                  - ``'NORMAL_ANTI'``: Antithetic normal draws,
                  - ``'NORMAL_HALTON2'``: Normal draws from Halton
                    base 2 sequence,
                  - ``'NORMAL_HALTON3'``: Normal draws from Halton
                    base 3 sequence,
                  - ``'NORMAL_HALTON5'``: Normal draws from Halton
                    base 5 sequence,
                  - ``'NORMAL_MLHS'``: Normal draws from Modified
                    Latin Hypercube Sampling,
                  - ``'NORMAL_MLHS_ANTI'``: Antithetic normal draws
                    from Modified Latin Hypercube Sampling]

                  For an updated description of the native types, call the function
                  :func:`~biogeme.native_draws.description_of_native_draws`.

    :param names: the list of names of the variables that require draws
        to be generated.
    :param sample_size: number of observations in the sample.
    :param number_of_draws: number of draws to generate.


    :param user_random_number_generators: a dictionary of generators. The keys of the dictionary
       characterize the name of the generators, and must be
       different from the pre-defined generators in Biogeme
       The elements of the
       dictionary are tuples, where the first element is a function that takes two arguments: the
       number of series to generate (typically, the size of the
       database), and the number of draws per series, and returns the array of numbers.
       The second element is a description.

    :return: a 3-dimensional table with draws. The 3 dimensions are

          1. number of individuals
          2. number of draws
          3. number of variables

    Example::

          types = {'randomDraws1': 'NORMAL_MLHS_ANTI',
                   'randomDraws2': 'UNIFORM_MLHS_ANTI',
                   'randomDraws3': 'UNIFORMSYM_MLHS_ANTI'}
          theDrawsTable = my_data.generateDraws(types,
              ['randomDraws1', 'randomDraws2', 'randomDraws3'], 10)


    :raise BiogemeError: if a type of draw is unknown.

    :raise BiogemeError: if the output of the draw generator does not
        have the requested dimensions.

    """

    manager = DrawGeneratorManager(user_generators=user_random_number_generators)
    return manager.generate_draws(
        draw_types=draw_types,
        variable_names=names,
        sample_size=sample_size,
        number_of_draws=number_of_draws,
    )
