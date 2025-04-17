"""Object containing the output of numerical expression  calculation"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Sequence, TypeVar

import numpy as np

GenericType = TypeVar('GenericType')


def convert_to_dict(
    the_sequence: Sequence[GenericType], the_map: dict[str, int]
) -> dict[str, GenericType]:
    """Convert an indexed sequence into a dict mapping a name to the elements.

    :param the_sequence: sequence of element to convert
    :param the_map: dict mapping the names with their index in the sequence.
    :return: the sequence converted into dict
    """
    # Check for any index out of range errors before creating the dictionary
    for name, index in the_map.items():
        if index >= len(the_sequence) or index < 0:
            raise IndexError(
                f"Index {index} for variable '{name}' is out of range for sequence of length {len(the_sequence)}."
            )

    # Use dictionary comprehension to create the result
    result = {name: the_sequence[index] for name, index in the_map.items()}
    return result


@dataclass
class FunctionOutput:
    """Output of a function calculation"""

    function: float
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None
    bhhh: np.ndarray | None = None


@dataclass
class DisaggregateFunctionOutput:
    """Output of a function calculation"""

    functions: np.ndarray
    gradients: np.ndarray | None = None
    hessians: np.ndarray | None = None
    bhhhs: np.ndarray | None = None

    def __len__(self):
        return len(self.functions)

    def unique_entry(self) -> FunctionOutput | None:
        """When there is only one entry, we generate the BiogemeFunctionOutput object"""
        if len(self) == 1:
            return FunctionOutput(
                function=float(self.functions[0]),
                gradient=self.gradients[0] if self.gradients is not None else None,
                hessian=self.hessians[0] if self.hessians is not None else None,
                bhhh=self.bhhhs[0] if self.bhhhs is not None else None,
            )
        return None


@runtime_checkable
class NamedFunctionResult(Protocol):
    """Protocol defining the interface for named function outputs."""

    function: float
    gradient: dict[str, float] | None
    hessian: dict[str, dict[str, float]] | None
    bhhh: dict[str, dict[str, float]] | None


class NamedOutputMixin:
    """Mixin providing common methods for named outputs."""

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: function={self.function}, "
            f"gradient={self.gradient}, hessian={self.hessian}, bhhh={self.bhhh}>"
        )


@dataclass
class NamedFunctionOutput(NamedOutputMixin, NamedFunctionResult):
    """Output of a function calculation, with names of variables"""

    function_output: FunctionOutput
    mapping: dict[str, int]

    function: float = None
    gradient: dict[str, float] | None = None
    hessian: dict[str, dict[str, float]] | None = None
    bhhh: dict[str, dict[str, float]] | None = None

    def __post_init__(self):
        self.function = self.function_output.function
        self.gradient = (
            None
            if self.function_output.gradient is None
            else convert_to_dict(self.function_output.gradient, self.mapping)
        )
        self.hessian = (
            None
            if self.function_output.hessian is None
            else convert_to_dict(
                [
                    convert_to_dict(row, self.mapping)
                    for row in self.function_output.hessian
                ],
                self.mapping,
            )
        )
        self.bhhh = (
            None
            if self.function_output.bhhh is None
            else convert_to_dict(
                [
                    convert_to_dict(row, self.mapping)
                    for row in self.function_output.bhhh
                ],
                self.mapping,
            )
        )
