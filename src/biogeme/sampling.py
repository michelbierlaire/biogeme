""" Module in charge of functionalities related to the sampling of alternatives

:author: Michel Bierlaire
:date: Wed Sep  7 15:54:55 2022
"""

from collections import defaultdict
from typing import NamedTuple, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import biogeme.exceptions as excep
import biogeme.expressions as expr


class StratumTuple(NamedTuple):
    subset: Set[int]
    sample_size: int


LOG_PROBA_COL = '_log_proba'


def sample_alternatives(alternatives, id_column, partition, chosen=None):
    """Performing the sampling of alternatives

    :param alternatives: Pandas data frame containing all the
        alternatives as rows. One column must contain a unique ID
        identifying the alternatives. The other columns contain
        variables to include in the data file.
    :type alternatives: pandas.DataFrame

    :param id_column: name of the columns where the IDs of the
        alternatives are stored.
    :type id_column: str

    :param partition: each StratumTuple contains a set of IDs
        characterizing the subset, and the sample size, that is the
        number of alternatives to randomly draw from the subset.
    :type partition: tuple(StratumTuple)

    :param chosen: ID of the chosen alternative, that must be included
        in the choice set. If None, no alternative is added
        deterministically to the choice set.
    :type chosen: int

    :raise BiogemeError: if one alternative belongs to several subsets
        of the partition.

    :raise BiogemeError: if a set in the partition is empty.

    :raise BiogemeError: if the chosen alternative is unknown.

    :raise BiogemeError: if the requested sample size for a stratum if
        larger than the size of the stratum

    :raise BiogemeError: if some alternative do not appear in the partition

    """
    # Verify that we have a partition
    nbr_unique_elements = len(set.union(*[s.subset for s in partition]))
    total_nbr = sum(list(len(s.subset) for s in partition))
    if nbr_unique_elements != total_nbr:
        error_msg = (
            f'This is not a partition. There are {nbr_unique_elements} '
            f'unique elements, and the total size of the partition '
            f'is {total_nbr}. Some elements are therefore present '
            f'in more than one subset.'
        )
        raise excep.BiogemeError(error_msg)

    if nbr_unique_elements != alternatives.shape[0]:
        error_msg = (
            f'The partitions contain {nbr_unique_elements} alternatives '
            f'while there are {alternatives.shape[0]} in the database'
        )
        raise excep.BiogemeError(error_msg)
    # Verify that all requested alternatives appear in the database of alternatives
    for stratum in partition:
        for alt in stratum.subset:
            if alt not in alternatives[id_column]:
                error_msg = (
                    f'Alternative {alt} does not appear in the database of alternaitves'
                )
                raise excep.BiogemeError(error_msg)

    results = []

    for stratum in partition:
        n = len(stratum.subset)
        if n == 0:
            error_msg = 'A stratum is empty'
            raise excep.BiogemeError(error_msg)

        k = stratum.sample_size
        if k > n:
            error_msg = f'Cannot draw {k} elements in a stratum of size {n}'
            raise excep.BiogemeError(error_msg)

        logproba = np.log(k) - np.log(n)
        subset = alternatives[alternatives[id_column].isin(stratum.subset)]
        if chosen is not None and chosen in stratum.subset:
            chosen_alternative = alternatives[alternatives[id_column] == chosen].copy()
            if len(chosen_alternative) < 1:
                error_msg = f'Unknown alternative: {chosen}'
                raise excep.BiogemeError(error_msg)
            if len(chosen_alternative) > 1:
                error_msg = f'Duplicate alternative: {chosen}'
                raise excep.BiogemeError(error_msg)
            chosen_alternative[LOG_PROBA_COL] = logproba
            results.append(chosen_alternative)

            subset = subset.drop(
                subset[subset[id_column] == chosen].index, axis='index'
            )
            n -= 1
            k -= 1

        if k > 0:
            sample = subset.sample(n=k, replace=False, axis='index', ignore_index=True)
            sample[LOG_PROBA_COL] = logproba
            results.append(sample)

    return pd.concat(results, ignore_index=True)


def sampling_of_alternatives(
    partition,
    individuals,
    choice_column,
    alternatives,
    id_column,
    always_include_chosen=True,
):
    """Generation of databases with samples of alternatives

      :param partition: each StratumTuple contains a set of IDs
        characterizing the subset, and the sample size, that is the
        number of alternatives to randomly draw from the subset.
    :type partition: tuple(StratumTuple)

    :param individuals: Pandas data frame containing all the
        individuals as rows. One column must contain the choice of
        each individual.
    :type individuals: pandas.DataFrame

    :param choice_column: name of the column containing the choice of
        each individual.
    :type choice_column: str

    :param alternatives: Pandas data frame containing all the
        alternatives as rows. One column must contain a unique ID
        identifying the alternatives. The other columns contain
        variables to include in the data file.
    :type alternatives: pandas.DataFrame

    :param id_column: name of the column containing the Ids of the
        alternatives.
    :type id_column: str

    :param always_include_chosen: if True, the chosen alternative is
        always included in the choice set with label 0.
    :type always_include_chosen: bool

    :return: data frame containing the data ready for Biogeme.
    :rtype: pandas.DataFrame

    """
    for index_ind, the_individual_row in tqdm(
        individuals.iterrows(), total=individuals.shape[0]
    ):
        choice = the_individual_row[choice_column]
        the_alternatives = alternatives.copy(deep=True)
        if always_include_chosen:
            chosen = choice
        else:
            chosen = None

        sample = sample_alternatives(
            the_alternatives, id_column, partition, chosen=chosen
        )
        sample.reset_index(inplace=True, drop=True)

        if always_include_chosen:
            # Position the chosen alternative at the first row
            chosen_alternative = sample.index[sample[id_column] == choice].tolist()[0]
            new_index = [chosen_alternative] + [
                i for i in range(len(sample)) if i != chosen_alternative
            ]
            sample = sample.reindex(new_index).reset_index(drop=True)

        row_elements = {
            f'{c}_{index}': alt[c]
            for index, alt in sample.iterrows()
            for c in sample.columns
        }
        individuals.loc[index_ind, row_elements.keys()] = row_elements.values()

    return individuals


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
