from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .segmentation import Segmentation
from .segmentation_context import DiscreteSegmentationTuple

if TYPE_CHECKING:
    from biogeme.expressions import Beta


def segmented_beta(
    beta: Beta,
    segmentation_tuples: Iterable[DiscreteSegmentationTuple],
    prefix: str = 'segmented',
):
    """Obtain the segmented Beta from a unique function call

    :param beta: parameter to be segmented
    :param segmentation_tuples: characterization of the segmentations
    :param prefix: prefix to be used to generate the name of the
        segmented parameter
    :return: expression of the segmented Beta
    """
    the_segmentation = Segmentation(
        beta=beta, segmentation_tuples=segmentation_tuples, prefix=prefix
    )
    return the_segmentation.segmented_beta()
