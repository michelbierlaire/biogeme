from datetime import timedelta
from typing import Any

import numpy as np
from yaml import (
    Dumper,
    Loader,
    Node,
)


def timedelta_representer(dumper: Dumper, data: timedelta) -> Node:
    """Represent a timedelta object as the total seconds in a YAML string."""
    return dumper.represent_str(str(data.total_seconds()))


def timedelta_constructor(loader: Loader, node: Node) -> timedelta:
    """Construct a timedelta object from a YAML scalar representing total seconds."""
    value: str = loader.construct_scalar(node)
    return timedelta(seconds=float(value))


def contains_python_tags(yaml_string):
    return '!!python/' in yaml_string or '!!binary' in yaml_string


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
