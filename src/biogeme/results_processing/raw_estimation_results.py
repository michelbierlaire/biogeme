"""
Implementation of classes containing the estimation results.

Michel Bierlaire
Sun Sep 29 16:54:42 2024
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from yaml import (
    SafeLoader,
    add_constructor,
    add_representer,
    dump,
    load,
)

from biogeme.tools import safe_deserialize_array
from biogeme.tools.yaml import (
    check_for_invalid_yaml_values,
    contains_python_tags,
    timedelta_constructor,
    timedelta_representer,
)
from biogeme.version import get_version, versionDate

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
    gradient: list[float] | None
    hessian: list[list[float]] | None
    bhhh: list[list[float]] | None
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
    optimization_complete: bool = True
    gradient_bhhh_complete: bool = True
    hessian_complete: bool = True
    bootstrap_complete: bool = True
    analytical_hessian_mode: str | None = None
    hessian_parameter_block_size: int | None = None
    hessian_observation_batch_size: int | None = None


# Register the custom handlers with PyYAML
add_representer(timedelta, timedelta_representer)
add_constructor('tag:yaml.org,2002:str', timedelta_constructor)


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
            f'The YAML output [{yaml_string}] contains unsafe Python object tags. Aborting serialization of {filename}.'
        )

    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
        ) as file:
            temporary_name = file.name
            now = datetime.now()
            print(
                f'# File {filename} has automatically been generated on {now}',
                file=file,
            )
            print(f'# biogeme {get_version()} [{versionDate}]\n', file=file)
            file.write(yaml_string)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
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
    if data['optimization_messages'] is not None:
        if 'Optimization time' in data['optimization_messages']:
            optimization_time = timedelta(
                seconds=float(data['optimization_messages']['Optimization time'])
            )
            data['optimization_messages']['Optimization time'] = optimization_time
    return RawEstimationResults(
        model_name=data['model_name'],
        user_notes=data['user_notes'],
        beta_names=list(data['beta_names']),
        beta_values=list(data['beta_values']),
        lower_bounds=data['lower_bounds'],
        upper_bounds=data['upper_bounds'],
        gradient=(
            None
            if data['gradient'] is None
            else safe_deserialize_array(data['gradient'])
        ),
        hessian=(
            None if data['hessian'] is None else safe_deserialize_array(data['hessian'])
        ),
        bhhh=(None if data['bhhh'] is None else safe_deserialize_array(data['bhhh'])),
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
        optimization_complete=data.get('optimization_complete', True),
        gradient_bhhh_complete=data.get('gradient_bhhh_complete', True),
        hessian_complete=data.get('hessian_complete', True),
        bootstrap_complete=data.get('bootstrap_complete', True),
        analytical_hessian_mode=data.get('analytical_hessian_mode'),
        hessian_parameter_block_size=data.get('hessian_parameter_block_size'),
        hessian_observation_batch_size=data.get('hessian_observation_batch_size'),
    )
