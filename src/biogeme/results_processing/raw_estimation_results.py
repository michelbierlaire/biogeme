"""
Implementation of classes containing the estimation results.

Michel Bierlaire
Sun Sep 29 16:54:42 2024
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from biogeme.tools import safe_deserialize_array
from biogeme.version import get_version, versionDate
from yaml import (
    Dumper,
    Loader,
    Node,
    SafeLoader,
    add_constructor,
    add_representer,
    dump,
    load,
)

logger = logging.getLogger(__name__)


@dataclass
class RawEstimationResults:
    """Data class containing the unprocessed estimation results. Must be simple and contain no method, in order to
    serialize it easily."""

    model_name: str
    user_notes: str
    beta_names: list[str]
    beta_values: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    gradient: list[float]
    hessian: list[list[float]]
    bhhh: list[list[float]]
    null_log_likelihood: float
    initial_log_likelihood: float | None
    final_log_likelihood: float
    data_name: str
    sample_size: int
    number_of_observations: int
    monte_carlo: bool
    number_of_draws: int
    types_of_draws: dict[str, str]
    number_of_excluded_data: int
    draws_processing_time: timedelta
    optimization_messages: dict[str, Any]
    convergence: bool
    bootstrap: list[list[float]]
    bootstrap_time: timedelta | None


# Custom handlers to serialize and deserialize complex objects
def timedelta_representer(dumper: Dumper, data: timedelta) -> Node:
    """Represent a timedelta object as the total seconds in a YAML string."""
    return dumper.represent_str(str(data.total_seconds()))


def timedelta_constructor(loader: Loader, node: Node) -> timedelta:
    """Construct a timedelta object from a YAML scalar representing total seconds."""
    value: str = loader.construct_scalar(node)
    return timedelta(seconds=float(value))


# Register the custom handlers with PyYAML
add_representer(timedelta, timedelta_representer)
add_constructor('tag:yaml.org,2002:str', timedelta_constructor)


def check_for_invalid_yaml_values(data: Any, path='root'):
    """Recursively checks for NaN or binary values in the data."""
    if isinstance(data, float) and np.isnan(data):
        raise ValueError(f'Invalid NaN value found at {path}')
    if isinstance(data, bytes):
        raise ValueError(f'Binary data (bytes) not allowed at {path}')
    if isinstance(data, dict):
        for key, value in data.items():
            check_for_invalid_yaml_values(value, f'{path}.{key}')
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            check_for_invalid_yaml_values(item, f'{path}[{i}]')


def contains_python_tags(yaml_string):
    return '!!python/' in yaml_string or '!!binary' in yaml_string


# To serialize the RawEstimationResults instance to a YAML file
def serialize_to_yaml(data: RawEstimationResults, filename: str) -> None:
    """Dump the data in an ASCII file

    :param data: raw estimation results
    :param filename: name of the file
    """
    dict_data = asdict(data)
    check_for_invalid_yaml_values(dict_data)
    # Convert to YAML string
    yaml_string = dump(dict_data)

    # Check for unsafe Python-specific tags
    if contains_python_tags(yaml_string):
        raise ValueError(
            f"The YAML output [{yaml_string}] contains unsafe Python object tags. Aborting serialization of {filename}."
        )

    with open(filename, 'w') as file:
        now = datetime.now()
        print(f'# File {filename} has automatically been generated on {now}', file=file)
        print(f'# biogeme {get_version()} [{versionDate}]\n', file=file)
        file.write(yaml_string)
    logger.info(f'File {filename} has been generated.')


# To deserialize the RawEstimationResults instance from a YAML file
def deserialize_from_yaml(filename) -> RawEstimationResults:
    """Restore data from a YAML file

    :param filename: name of the file
    :return: raw estimation results
    """
    with open(filename, 'r') as file:
        data = load(file, Loader=SafeLoader)
    bootstrap_time = (
        timedelta(seconds=float(data['bootstrap_time']))
        if data['bootstrap_time'] is not None
        else None
    )
    return RawEstimationResults(
        model_name=data['model_name'],
        user_notes=data['user_notes'],
        beta_names=list(data['beta_names']),
        beta_values=list(data['beta_values']),
        lower_bounds=data['lower_bounds'],
        upper_bounds=data['upper_bounds'],
        gradient=safe_deserialize_array(data['gradient']),
        hessian=(
            None if data['hessian'] is None else safe_deserialize_array(data['hessian'])
        ),
        bhhh=safe_deserialize_array(data['bhhh']),
        null_log_likelihood=data['null_log_likelihood'],
        initial_log_likelihood=data['initial_log_likelihood'],
        final_log_likelihood=data['final_log_likelihood'],
        data_name=data['data_name'],
        sample_size=data['sample_size'],
        number_of_observations=data['number_of_observations'],
        monte_carlo=data['monte_carlo'],
        number_of_draws=data['number_of_draws'],
        types_of_draws=data['types_of_draws'],
        number_of_excluded_data=data['number_of_excluded_data'],
        draws_processing_time=timedelta(seconds=float(data['draws_processing_time'])),
        optimization_messages=data['optimization_messages'],
        convergence=data['convergence'],
        bootstrap=data['bootstrap'],
        bootstrap_time=bootstrap_time,
    )
