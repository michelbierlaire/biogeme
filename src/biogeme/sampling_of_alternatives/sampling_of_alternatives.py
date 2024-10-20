""" Module in charge of functionalities related to the sampling of alternatives

:author: Michel Bierlaire
:date: Thu Sep  7 10:14:54 2023
"""

import copy
import logging

import numpy as np
import pandas as pd

from biogeme.exceptions import BiogemeError
from .sampling_context import SamplingContext, LOG_PROBA_COL, MEV_WEIGHT, CNL_PREFIX

logger = logging.getLogger(__name__)


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

        """

        self.alternatives = context.alternatives
        self.id_column = context.id_column
        self.partition = context.sampling_protocol
        self.second_partition = context.mev_sampling_protocol
        self.cnl_nests = context.cnl_nests

    def add_alphas(self, the_sample: pd.DataFrame) -> pd.DataFrame:
        """Add the alpha parameters in the sampled database"""

        if self.cnl_nests is None:
            raise BiogemeError(
                f'No nests have been defined for the cross-nested logit model'
            )

        # We add the alpha parameters in the sample
        def get_alphas(alternative_id: int) -> pd.Series:
            """Prepare the alphas for insertion in the data frame"""
            assert self.cnl_nests is not None
            the_dict = self.cnl_nests.get_alpha_values(alternative_id)
            return pd.Series(the_dict)

        new_columns = the_sample[self.id_column].apply(get_alphas)
        new_columns = new_columns.rename(columns=lambda x: CNL_PREFIX + x)

        the_sample = pd.concat([the_sample, new_columns], axis="columns")
        return the_sample

    def sample_alternatives(self, chosen: int) -> pd.DataFrame:
        """Performing the sampling of alternatives

        :param chosen: ID of the chosen alternative, that must be included
            in the choice set.

        :return: data frame containing a sample of
            alternatives. The first one is the chosen alternative
        :raise BiogemeError: if the chosen alternative is unknown.

        """
        chosen_alternative = self.alternatives[
            self.alternatives[self.id_column] == chosen
        ].copy()
        if len(chosen_alternative) < 1:
            error_msg = f"Unknown alternative: {chosen}"
            raise BiogemeError(error_msg)
        if len(chosen_alternative) > 1:
            error_msg = f"Duplicate alternative: {chosen}"
            raise BiogemeError(error_msg)

        results = []

        for stratum in self.partition:
            # stratum.subset is a set of int
            # We create a copy because we'll have to drop the chosen alternative
            the_subset_of_alternatives = copy.deepcopy(stratum.subset)
            stratum_size = len(stratum.subset)
            sample_size = stratum.sample_size
            logproba = np.log(sample_size) - np.log(stratum_size)
            if chosen in stratum.subset:
                # Discard the chosen alternative
                the_subset_of_alternatives.discard(chosen)
                # And we sample one alternative less
                sample_size -= 1
                # Include the correction terms
                chosen_alternative[LOG_PROBA_COL] = logproba

            # subset is a pandas data frame containing the description
            # of all alternatives in the subset
            subset = self.alternatives[
                self.alternatives[self.id_column].isin(the_subset_of_alternatives)
            ]
            # Perform the sampling
            sample = subset.sample(
                n=sample_size, replace=False, axis="index", ignore_index=True
            )

            sample[LOG_PROBA_COL] = logproba

            results.append(sample)

        the_sample = pd.concat(results, ignore_index=True)
        # Add the chosen alternative. By construction, it is not in the sample.
        the_sample = pd.concat([chosen_alternative, the_sample], ignore_index=True)
        if self.cnl_nests:
            self.add_alphas(the_sample)
        return the_sample

    def sample_mev_alternatives(self) -> pd.DataFrame:
        """Performing the sampling of alternatives for the MEV
        terms. Here, the chosen alternative is ignored.

        :return: data frame containing a sample of alternatives

        """
        results = []

        for stratum in self.second_partition:
            stratum_size = len(stratum.subset)
            sample_size = stratum.sample_size
            mev_weight = stratum_size / sample_size
            subset = self.alternatives[
                self.alternatives[self.id_column].isin(stratum.subset)
            ]
            sample = subset.sample(
                n=sample_size, replace=False, axis="index", ignore_index=True
            )
            sample[MEV_WEIGHT] = mev_weight

            results.append(sample)

        the_sample = pd.concat(results, ignore_index=True)

        if self.cnl_nests:
            self.add_alphas(the_sample)

        return the_sample
