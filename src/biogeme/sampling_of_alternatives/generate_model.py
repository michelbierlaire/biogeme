"""Generation of models estimated with samples of alternatives

:author: Michel Bierlaire
:date: Fri Sep 22 12:14:59 2023
"""

import copy
import logging

from biogeme.expressions import (
    BelongsTo,
    ConditionalSum,
    ConditionalTermTuple,
    Expression,
    Variable,
    exp,
    log,
    logzero,
)
from biogeme.models import loglogit
from biogeme.nests import NestsForNestedLogit
from .sampling_context import CNL_PREFIX, LOG_PROBA_COL, MEV_WEIGHT, SamplingContext
from ..expressions.add_prefix_suffix import add_prefix_suffix_to_all_variables

logger = logging.getLogger(__name__)


class GenerateModel:
    """Class in charge of generating the biogeme expression for the
    loglikelihood function
    """

    def __init__(self, context: SamplingContext):
        """Constructor

        :param context: contains all the information that is needed to
            perform the sampling of alternatives.

        """

        self.context = context
        self.utility_function = context.utility_function
        self.total_sample_size = context.total_sample_size
        self.total_mev_sample_size = context.total_mev_sample_size
        self.attributes = context.attributes
        self.mev_prefix = context.mev_prefix

        self.utilities = {
            alt_id: self.generate_utility(prefix="", suffix=f"_{alt_id}")
            for alt_id in range(self.total_sample_size)
        }
        if self.context.mev_partition is None:
            self.mev_utilities = {
                alt_id: self.utilities[alt_id]
                for alt_id in range(1, self.total_sample_size)
            }
            logger.debug(
                f'No specific MEV partition. {self.total_sample_size} are sampled for MEV terms.'
            )
        else:
            self.mev_utilities = {
                alt_id: self.generate_utility(
                    prefix=self.mev_prefix, suffix=f"_{alt_id}"
                )
                for alt_id in range(self.context.total_mev_sample_size)
            }
            logger.debug(
                f'Specific MEV partition provided. {self.total_mev_sample_size} are sampled for MEV terms.'
            )

    def generate_utility(self, prefix: str, suffix: str) -> Expression:
        """Generate the utility function for one alternative

        :param prefix: prefix to add to the attributes

        :param suffix: suffix to add to the attributes

        """
        copy_utility = copy.deepcopy(self.utility_function)
        add_prefix_suffix_to_all_variables(
            expr=copy_utility, prefix=prefix, suffix=suffix
        )
        return copy_utility

    def get_logit(self) -> Expression:
        """Returns the expression for the log likelihood of the logit model"""

        corrected_utilities = {
            i: utility - Variable(f"{LOG_PROBA_COL}_{i}")
            for i, utility in self.utilities.items()
        }

        return loglogit(corrected_utilities, None, 0)

    def get_nested_logit(self, nests: NestsForNestedLogit) -> Expression:
        """Returns the expression for the log likelihood of the nested logit model

        :param nests: A tuple containing as many items as nests.
            Each item is also a tuple containing two items:

        - an object of type biogeme.expressions.expr.Expression representing
          the nest parameter,
        - a list containing the list of identifiers of the alternatives
          belonging to the nest.

        Example::

            nesta = MUA ,[1, 2, 3]
            nestb = MUB ,[4, 5, 6]
            nests = nesta, nestb


        """

        dict_of_mev_sums = {}

        # We first compute the MEV partial sum for each nest
        for nest in nests:
            mu_param = nest.nest_param
            list_of_alternatives = nest.list_of_alternatives
            self.context.check_valid_alternatives(set(list_of_alternatives))
            # We first build the MEV term using a sample of
            # alternatives. To build this term, we iterate from 1, as
            # we ignore the chosen alternative.
            list_of_terms = []
            for i, utility in self.mev_utilities.items():
                alternative_id = Variable(
                    f"{self.mev_prefix}{self.context.id_column}_{i}"
                )
                belong_to_nest = BelongsTo(alternative_id, set(list_of_alternatives))
                weight = Variable(f"{self.mev_prefix}{MEV_WEIGHT}_{i}")
                the_term = ConditionalTermTuple(
                    condition=belong_to_nest, term=weight * exp(mu_param * utility)
                )
                list_of_terms.append(the_term)
            dict_of_mev_sums[tuple(list_of_alternatives)] = ConditionalSum(
                list_of_terms
            )

        # We now add all relevant MEV terms to the utilities

        dict_of_mev_terms = {}
        for i, the_utility in self.utilities.items():
            alternative_id = Variable(f"{self.context.id_column}_{i}")
            list_of_terms = []
            for nest in nests:
                mu_param = nest.nest_param
                mev_sum = dict_of_mev_sums[tuple(nest.list_of_alternatives)]
                mev_term = (mu_param - 1.0) * the_utility + (
                    (1.0 / mu_param) - 1.0
                ) * log(mev_sum)
                belong_to_nest = BelongsTo(
                    alternative_id, set(nest.list_of_alternatives)
                )
                the_term = ConditionalTermTuple(condition=belong_to_nest, term=mev_term)

                list_of_terms.append(the_term)
            dict_of_mev_terms[i] = ConditionalSum(list_of_terms)

        corrected_utilities = {
            key: util - Variable(f"{LOG_PROBA_COL}_{key}") + dict_of_mev_terms[key]
            for key, util in self.utilities.items()
        }
        return loglogit(corrected_utilities, None, 0)

    def get_cross_nested_logit(self) -> Expression:
        """Returns the expression for the log likelihood of the nested logit model"""
        nests = self.context.cnl_nests

        # We  compute the MEV partial sum for each nest
        dict_of_mev_sums = {}

        for nest in nests:
            mu_param = nest.nest_param
            # We first build the MEV term using a sample of
            # alternatives. To build this term, we iterate from 1, as
            # we ignore the chosen alternative.
            list_of_terms = []
            for i, utility in self.mev_utilities.items():
                alpha_name = f'{self.mev_prefix}{CNL_PREFIX}{nest.name}_{i}'
                alpha = Variable(alpha_name)
                weight = Variable(f"{self.mev_prefix}{MEV_WEIGHT}_{i}")
                the_term = ConditionalTermTuple(
                    condition=alpha != 0.0,
                    term=weight * alpha**mu_param * exp(mu_param * utility),
                )
                list_of_terms.append(the_term)

            dict_of_mev_sums[nest.name] = ConditionalSum(list_of_terms)

        # We now add all relevant MEV terms to the utilities
        dict_of_mev_terms = {}
        for i, the_utility in self.utilities.items():
            list_of_terms = []
            for nest in nests:
                # Note that we need an alpha for each alternative in the main sample.
                alpha = Variable(f"{CNL_PREFIX}{nest.name}_{i}")
                mu_param = nest.nest_param
                mev_sum = dict_of_mev_sums[nest.name] ** ((1.0 / mu_param) - 1.0)
                mev_term = alpha**mu_param * exp((mu_param - 1) * the_utility) * mev_sum
                the_term = ConditionalTermTuple(condition=alpha != 0.0, term=mev_term)
                list_of_terms.append(the_term)
            dict_of_mev_terms[i] = logzero(ConditionalSum(list_of_terms))

        corrected_utilities = {
            key: util - Variable(f"{LOG_PROBA_COL}_{key}") + dict_of_mev_terms[key]
            for key, util in self.utilities.items()
        }
        return loglogit(corrected_utilities, None, 0)
