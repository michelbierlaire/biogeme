"""Object containing the output of numerical expression  calculation"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

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
    if any(index >= len(the_sequence) or index < 0 for index in the_map.values()):
        raise IndexError('One or more indices are out of the acceptable range.')

    # Use dictionary comprehension to create the result
    result = {name: the_sequence[index] for name, index in the_map.items()}
    return result


@dataclass
class FunctionOutput:
    """Output of a function calculation"""

    function: float
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None


@dataclass
class BiogemeFunctionOutput(FunctionOutput):
    """Output of a function calculation"""

    bhhh: np.ndarray | None = None


@dataclass
class BiogemeDisaggregateFunctionOutput:
    """Output of a function calculation"""

    functions: np.ndarray
    gradients: np.ndarray | None = None
    hessians: np.ndarray | None = None
    bhhhs: np.ndarray | None = None


class NamedFunctionOutput:
    """Output of a function calculation, with names of variables"""

    def __init__(
        self, function_output: FunctionOutput, mapping: dict[str, int]
    ) -> None:
        """
        Constructor
        :param function_output: function output stored as numpy parray.
        :param mapping: dict mapping the names with their index in the sequence.
        """
        self.function: float = function_output.function
        self.gradient: dict[str, float] | None = (
            None
            if function_output.gradient is None
            else convert_to_dict(function_output.gradient, mapping)
        )
        self.hessian: dict[str, dict[str, float]] | None = (
            None
            if function_output.hessian is None
            else convert_to_dict(
                [convert_to_dict(row, mapping) for row in function_output.hessian],
                mapping,
            )
        )

    def __repr__(self):
        """"""
        ...


class NamedBiogemeFunctionOutput(NamedFunctionOutput):
    """Output of a function calculation, with names of variables"""

    def __init__(
        self, function_output: BiogemeFunctionOutput, mapping: dict[str, int]
    ) -> None:
        super().__init__(function_output=function_output, mapping=mapping)
        self.bhhh: dict[str, dict[str, float]] | None = (
            None
            if function_output.bhhh is None
            else (
                convert_to_dict(
                    [convert_to_dict(row, mapping) for row in function_output.bhhh],
                    mapping,
                )
            )
        )


@dataclass
class NamedBiogemeDisaggregateFunctionOutput:
    """Output of a function calculation"""

    def __init__(
        self,
        function_output: BiogemeDisaggregateFunctionOutput,
        mapping: dict[str, int],
    ) -> None:
        self.functions: list[float] = [value for value in function_output.functions]
        self.gradients: list[dict[str, float]] | None = (
            None
            if function_output.gradients is None
            else [
                convert_to_dict(gradient, mapping)
                for gradient in function_output.gradients
            ]
        )
        self.hessians: list[dict[str, dict[str, float]]] | None = (
            None
            if function_output.hessians is None
            else (
                [
                    convert_to_dict(
                        [convert_to_dict(row, mapping) for row in hessian],
                        mapping,
                    )
                    for hessian in function_output.hessians
                ]
            )
        )
        self.bhhhs: list[dict[str, dict[str, float]]] | None = (
            None
            if function_output.bhhhs is None
            else (
                [
                    convert_to_dict(
                        [convert_to_dict(row, mapping) for row in bhhh],
                        mapping,
                    )
                    for bhhh in function_output.bhhhs
                ]
            )
        )
