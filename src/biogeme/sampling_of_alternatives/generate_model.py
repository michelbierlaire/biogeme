"""Generation of models estimated with samples of alternatives

:author: Michel Bierlaire
:date: Fri Sep 22 12:14:59 2023
"""

import copy
import logging

from biogeme.expressions import (
    Expression,
    LogSampleCrossNested,
    LogSampledLogit,
    LogSampledNested,
    Variable,
)
from biogeme.nests import NestsForNestedLogit

from ..expressions.add_prefix_suffix import add_prefix_suffix_to_all_variables
from .sampling_context import CNL_PREFIX, LOG_PROBA_COL, MEV_WEIGHT, SamplingContext

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
            alt_id: self.generate_utility(prefix='', suffix=f'_{alt_id}')
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
                    prefix=self.mev_prefix, suffix=f'_{alt_id}'
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

        log_probabilities = {
            i: Variable(f'{LOG_PROBA_COL}_{i}') for i in self.utilities
        }

        return LogSampledLogit(
            utilities=self.utilities,
            log_probabilities=log_probabilities,
            choice=0,
        )

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

        log_probabilities = {
            i: Variable(f'{LOG_PROBA_COL}_{i}') for i in self.utilities
        }

        alternative_ids = {
            i: Variable(f'{self.context.id_column}_{i}') for i in self.utilities
        }

        mev_alternative_ids = {
            i: Variable(f'{self.mev_prefix}{self.context.id_column}_{i}')
            for i in self.mev_utilities
        }

        mev_weights = {
            i: Variable(f'{self.mev_prefix}{MEV_WEIGHT}_{i}')
            for i in self.mev_utilities
        }

        return LogSampledNested(
            utilities=self.utilities,
            log_probabilities=log_probabilities,
            alternative_ids=alternative_ids,
            mev_utilities=self.mev_utilities,
            mev_weights=mev_weights,
            mev_alternative_ids=mev_alternative_ids,
            nests=nests,
            choice=0,
        )

    def get_cross_nested_logit(self) -> Expression:
        """Returns the expression for the log likelihood of the nested logit model"""
        nests = self.context.cnl_nests

        log_probabilities = {
            i: Variable(f'{LOG_PROBA_COL}_{i}') for i in self.utilities
        }

        alphas = {
            nest.name: {
                i: Variable(f'{CNL_PREFIX}{nest.name}_{i}') for i in self.utilities
            }
            for nest in nests
        }

        mev_weights = {
            i: Variable(f'{self.mev_prefix}{MEV_WEIGHT}_{i}')
            for i in self.mev_utilities
        }

        mev_alphas = {
            nest.name: {
                i: Variable(f'{self.mev_prefix}{CNL_PREFIX}{nest.name}_{i}')
                for i in self.mev_utilities
            }
            for nest in nests
        }

        return LogSampleCrossNested(
            utilities=self.utilities,
            log_probabilities=log_probabilities,
            alphas=alphas,
            mev_utilities=self.mev_utilities,
            mev_weights=mev_weights,
            mev_alphas=mev_alphas,
            nests=nests,
            choice=0,
        )
