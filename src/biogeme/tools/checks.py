"""Utilities to perform various checks

Michel Bierlaire
Fri Apr 5 11:09:39 2024
"""

from typing import AbstractSet, Any, TypeVar


def check_consistency_of_named_dicts(
    named_dicts: dict[str, dict[Any, Any]],
) -> tuple[bool, str | None]:
    """Verify that all dictionaries have the same set of keys. If not, report the inconsistencies.

    :param named_dicts: A dictionary where each key is a name (str) and each value is a dictionary to check.
    :return: True if keys are consistent across all dictionaries, False otherwise. If not consistent,
             returns a report detailing the inconsistencies.
    """
    # Initialize variables
    dict_names = list(named_dicts.keys())
    keys = [set(d.keys()) for d in named_dicts.values()]
    consistent = True
    inconsistencies = []

    # Reference keys for comparison
    reference_keys = keys[0]
    reference_name = dict_names[0]

    # Compare each set of keys with the first to check for consistency
    for i, (name, key_set) in enumerate(zip(dict_names[1:], keys[1:]), start=1):
        if key_set != reference_keys:
            consistent = False
            missing_in_ref = reference_keys - key_set
            extra_in_ref = key_set - reference_keys
            if missing_in_ref:
                inconsistencies.append(
                    f'{name} missing keys compared to {reference_name}: {missing_in_ref}.'
                )
            if extra_in_ref:
                inconsistencies.append(
                    f'{name} has extra keys compared to {reference_name}: {extra_in_ref}.'
                )

    # Compile the report if there are inconsistencies
    report = None
    if not consistent:
        report = "Inconsistencies found among dictionaries:\n" + "\n".join(
            inconsistencies
        )

    return consistent, report


def validate_dict_types(
    parameter: dict, name: str, value_type: type, key_type: type = int
) -> None:
    """Validate the types of keys and values in a dictionary.

    :param parameter: The dictionary parameter to validate.
    :param name: The name of the parameter for error messages.
    :param value_type: The expected type of the dictionary values.
    :param key_type: The expected type of the dictionary keys. Default is int.
    :raises TypeError: If key or value types do not match expectations.
    """
    if not isinstance(parameter, dict):
        raise TypeError(
            f"{name} must be a dict, got {type(parameter).__name__} instead."
        )
    for key, value in parameter.items():
        if not isinstance(key, key_type):
            raise TypeError(
                f"Keys in {name} must be of type {key_type.__name__}, found type {type(key).__name__}."
            )
        if not (isinstance(value, value_type) or value is None):
            raise TypeError(
                f"Values in {name} must be of type {value_type.__name__} or None, found type {type(value).__name__}."
            )


T = TypeVar("T")


def assert_sets_equal(
    name_a: str,
    set_a: AbstractSet[T],
    name_b: str,
    set_b: AbstractSet[T],
) -> None:
    """Raise an informative exception if two sets differ.

    :param name_a: Label for the first set (used in the error message).
    :param set_a: First set-like collection.
    :param name_b: Label for the second set (used in the error message).
    :param set_b: Second set-like collection.
    :raises ValueError: If the two sets do not contain exactly the same elements.
    """
    # Convert to real sets in case we get other Set implementations
    sa = set(set_a)
    sb = set(set_b)

    missing_in_b = sa - sb
    missing_in_a = sb - sa

    if missing_in_b or missing_in_a:
        msg_lines = ["Sets differ:"]
        if missing_in_b:
            msg_lines.append(
                f"  Elements in {name_a} but not in {name_b}: "
                f"{[repr(x) for x in sorted(missing_in_b, key=repr)]}"
            )
        if missing_in_a:
            msg_lines.append(
                f"  Elements in {name_b} but not in {name_a}: "
                f"{[repr(x) for x in sorted(missing_in_a, key=repr)]}"
            )
        raise ValueError("\n".join(msg_lines))
