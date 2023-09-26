""" Module in charge of functionalities related to the sampling of alternatives

:author: Michel Bierlaire
:date: Wed Sep  7 15:54:55 2022
"""

from collections import defaultdict
from typing import NamedTuple, Set, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from biogeme.exceptions import BiogemeError
import biogeme.expressions as expr
from biogeme.sampling_context import SamplingContext


class StratumTuple(NamedTuple):
    subset: Set[int]
    sample_size: int


LOG_PROBA_COL = '_log_proba'


def choice_sets_generation(sampling_context: SamplingContext) -> pd.DataFrame:
    """Generation of databases with samples of alternatives. Two strategies can be considered.


    :param sampling_context: contains all the data associated with the sampling of alternatives

    :return: data frame containing the data ready for Biogeme.
    :rtype: pandas.DataFrame

    """
    # Input validation
    appended_rows = []

    for _, individual_row in tqdm(individuals.iterrows(), total=individuals.shape[0]):
        choice = individual_row[choice_column]

        first_sample, chosen_alternative = sample_alternatives(
            alternatives, id_column, partition, chosen=choice
        )

        # Rename columns for chosen_alternative
        chosen_alternative = chosen_alternative.add_suffix('_0')
        chosen_dict = chosen_alternative.iloc[0].to_dict()

        # Rename columns for first_sample without multi-level index
        flattened_first_dict = {}
        for k, v in first_sample.iterrows():
            for col_name, col_value in v.items():
                flattened_first_dict[f'{col_name}_{k+1}'] = col_value

        row_data = individual_row.to_dict()
        row_data.update(chosen_dict)
        row_data.update(flattened_first_dict)

        if second_partition is not None:
            second_sample, _ = sample_alternatives(
                alternatives, id_column, second_partition, chosen=None
            )

            # Rename columns for second_sample without multi-level index
            flattened_second_dict = {}
            for k, v in second_sample.iterrows():
                for col_name, col_value in v.items():
                    flattened_second_dict[f'MEV_{col_name}_{k}'] = col_value

            row_data.update(flattened_second_dict)

        appended_rows.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    updated_individuals = pd.DataFrame(appended_rows)
    return updated_individuals


def mev_cnl_sampling(V, availability, sampling_log_probability, nests):
    """Generate the expression of the CNL G_i function in the context
    of sampling of alternatives.

    It is assumed that the following variables are available in the
        data: for each nest m and each alternative i, a variable m_i
        that is the level of membership of alternative i to nest m.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.
    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type availability: dict(int:biogeme.expressions.expr.Expression)

    :param sampling_log_probability: if not None, it means that the
        choice set is actually a subset that has been sampled from the
        full choice set. In that case, this is a dictionary mapping
        each alternative with the logarithm of its probability to be
        selected in the sample.
    :type sampling_log_probability: dict(int: biogeme.expressions.Expression)

    :param nests: a dictionary where the keys are the names of the
        nests, and the values are the nest parameters.
    :type nests: dict(str: biogeme.expressions.Beta)

    """

    Gi_terms = defaultdict(list)
    biosum = {}
    for nest, mu in nests.items():
        if availability is None:
            biosum = expr.bioMultSum(
                [
                    expr.exp(sampling_log_probability[i])
                    * expr.Variable(f'{nest}_{i}') ** mu
                    * expr.exp(mu * util)
                    for i, util in V.items()
                ]
            )
        else:
            biosum = expr.bioMultSum(
                [
                    expr.exp(sampling_log_probability[i])
                    * availability[i]
                    * expr.Variable(f'{nest}_{i}') ** mu
                    * expr.exp(mu * util)
                    for i, util in V.items()
                ]
            )
        for i, util in V.items():
            Gi_terms[i].append(
                expr.Variable(f'{nest}_{i}') ** mu
                * expr.exp((mu - 1) * (V[i]))
                * biosum ** ((1.0 / mu) - 1.0)
            )
    log_gi = {
        k: expr.logzero(expr.bioMultSum(G)) if G else expr.Numeric(0)
        for k, G in Gi_terms.items()
    }
    log_gi = {
        k: G if G == 0 else G - sampling_log_probability[k] for k, G in log_gi.items()
    }
    return log_gi
