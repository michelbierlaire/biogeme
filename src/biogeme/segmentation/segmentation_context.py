"""Class representing the characterization of a discrete segmentation.

A discrete segmentation divides a population based on the values of a socio-economic
variable, mapping each value to a category name. A reference category is defined
for comparison in model estimation.

Michel Bierlaire
Thu Apr 3 10:08:10 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from biogeme.exceptions import BiogemeError

if TYPE_CHECKING:
    from biogeme.expressions import Variable


class DiscreteSegmentationTuple:
    """Characterization of a discrete segmentation.

    This class is used to define how a discrete socio-economic variable
    is mapped to category names for segmentation purposes. One category
    is selected as the reference for use in estimation.
    """

    def __init__(
        self,
        variable: Variable | str,
        mapping: dict[int, str],
        reference: str | None = None,
    ):
        """Initialize a discrete segmentation characterization.

        :param variable: Socio-economic variable used for segmentation, either as a Variable instance or a string name.
        :param mapping: Dictionary mapping integer values of the variable to category names.
        :param reference: Optional name of the reference category. If not provided, the first category in the mapping is used.

        :raises BiogemeError: If the provided reference category is not found in the mapping.
        """
        from biogeme.expressions import Variable

        self.variable: Variable = (
            variable if isinstance(variable, Variable) else Variable(variable)
        )
        self.mapping: dict[int, str] = mapping
        if reference is None:
            self.reference: str = next(iter(mapping.values()))
        elif reference not in mapping.values():
            error_msg = (
                f'Reference category {reference} does not appear in the list '
                f'of categories: {mapping.values()}'
            )
            raise BiogemeError(error_msg)
        else:
            self.reference: str = reference

        if self.reference is None:
            raise BiogemeError('Reference should not be None')

    def __repr__(self) -> str:
        result = f'{self.variable.name}: [{self.mapping}] ref: {self.reference}'
        return result

    def __str__(self) -> str:
        result = f'{self.variable.name}: [{self.mapping}] ref: {self.reference}'
        return result
