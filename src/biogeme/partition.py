""" Defines a class that represents a partition of a list of integers

:author: Michel Bierlaire
:date: Thu Oct 26 17:30:36 2023

"""

import logging
from typing import Iterator, Sequence

Segment = set[int]

logger = logging.getLogger(__name__)


class Partition:
    """Class defining a partition of a set"""

    def __init__(self, segments: Sequence[Segment], full_set: Segment | None = None):
        self.segments = segments
        if full_set:
            self.full_set = full_set
        else:
            logger.warning(
                'The full set of the partition has not been specified. It is defined '
                'by default as the union of the provided segment.'
            )
            self.full_set = set().union(*self.segments)
        self.validate_segments()
        self.validate_partition()

    def number_of_segments(self) -> int:
        return len(self.segments)

    def validate_segments(self) -> None:
        """Check the types"""
        for segment in self.segments:
            if not isinstance(segment, set):
                raise ValueError('Each segment must be of type "set".')
            if not segment:
                raise ValueError('Partition contains empty segment')
            for item in segment:
                if not isinstance(item, int):
                    raise ValueError('Each item in the segment must be an integer.')

    def validate_partition(self) -> None:
        """Check that there is no intersection between segments"""
        for s1, segment1 in enumerate(self.segments):
            for s2, segment2 in enumerate(self.segments):
                if s1 != s2:
                    intersection = segment1.intersection(segment2)
                    if intersection:
                        raise ValueError(
                            f'Elements in two different segments: {intersection}.'
                        )

        # Check that the union of all segments is the given set
        union_of_segments = set().union(*self.segments)
        if union_of_segments != self.full_set:
            missing_elements = self.full_set - union_of_segments
            raise ValueError(
                f'The union of segments does not match the given set. '
                f'Missing elements: {missing_elements}.'
            )

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)
