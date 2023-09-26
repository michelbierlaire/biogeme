""" Module in charge of functionalities related to the sampling of alternatives

:author: Michel Bierlaire
:date: Thu Sep  7 10:14:54 2023
"""

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from biogeme.exceptions import BiogemeError
from .sampling_context import SamplingContext, StratumTuple, LOG_PROBA_COL, MEV_WEIGHT


def generate_segment_size(sample_size: int, number_of_segments: int) -> list[int]:
    """This function calculates the size of each segment, so that
    they are as close to each other as possible, and cover the full sample

    :param sample_size: total size of the sample
    :type sample_size: int

    :param number_of_segments: number of segments
    :type number_of_segments: int

    :return: list of length number_of_segments, containing the segment sizes
    :rtype: list[int]

    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative.")
    if number_of_segments <= 0:
        raise ValueError("Number of segments must be positive.")

    # Calculate the base value and the remainder
    base_value = sample_size // number_of_segments
    remainder = sample_size % number_of_segments

    # Distribute the base value across the list
    segment_sizes = [base_value] * number_of_segments

    # Distribute the remainder across the first few elements
    for i in range(remainder):
        segment_sizes[i] += 1

    return segment_sizes


class SamplingOfAlternatives:
    """Class dealing with the various methods needed to estimate
    models with samples of alternatives

    """

    def __init__(self, context: SamplingContext):
        """Constructor

        :param context: contains all the information that is needed to
        perform the sampling of alternatives.
        :type context: SamplingContext
        """

        self.alternatives = context.alternatives
        self.id_column = context.id_column
        self.partition = context.partition
        self.second_partition = context.second_partition

    def sample_alternatives(
        self, partition: list[StratumTuple], chosen: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame]]:
        """Performing the sampling of alternatives

        :param partition: partition characterizing the sampling.
        :param chosen: ID of the chosen alternative, that must be included
            in the choice set. If None, the chosen alternative is ignored.

        :return: two data frames: (i) data frame containing a sample of
            alternatives, organized in such a way that the segment
            containing the chosen alternative is the last one. Moreover,
            if the chosen alternative has been sampled, it will be the
            last one. (ii) one with the chosen alternative, if requested.
        :raise BiogemeError: if the chosen alternative is unknown.

        """
        if chosen is None:
            chosen_alternative = None
        else:
            chosen_alternative = self.alternatives[
                self.alternatives[self.id_column] == chosen
            ].copy()
            if len(chosen_alternative) < 1:
                error_msg = f'Unknown alternative: {chosen}'
                raise BiogemeError(error_msg)
            if len(chosen_alternative) > 1:
                error_msg = f'Duplicate alternative: {chosen}'
                raise BiogemeError(error_msg)

        results = []

        reordered = None

        for stratum in partition:
            stratum_size = len(stratum.subset)
            sample_size = stratum.sample_size
            logproba = np.log(sample_size) - np.log(stratum_size)
            mev_weight = stratum_size / sample_size
            subset = self.alternatives[
                self.alternatives[self.id_column].isin(stratum.subset)
            ]
            sample = subset.sample(
                n=sample_size, replace=False, axis='index', ignore_index=True
            )
            sample[LOG_PROBA_COL] = logproba
            sample[MEV_WEIGHT] = mev_weight

            if chosen is not None and chosen in stratum.subset:
                # Include the correction terms
                chosen_alternative[LOG_PROBA_COL] = logproba
                chosen_alternative[MEV_WEIGHT] = mev_weight
                # Move the chosen alternative to the end of the sample
                chosen_row = sample[sample[self.id_column] == chosen]
                unchosen_rows = sample[sample[self.id_column] != chosen]
                if chosen_row.empty:
                    reordered = unchosen_rows
                else:
                    reordered = pd.concat([unchosen_rows, chosen_row])
            else:
                results.append(sample)

        if chosen is not None:
            if reordered is None:
                error_msg = (
                    f'The chosen alternative {chosen} has not been found in any '
                    f'segment of the partition.'
                )
                raise BiogemeError(error_msg)
            results.append(reordered)
        return pd.concat(results, ignore_index=True), chosen_alternative
