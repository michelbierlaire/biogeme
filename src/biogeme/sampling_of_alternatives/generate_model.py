""" Generation of models estimated with samples of alternatives

:author: Michel Bierlaire
:date: Fri Sep 22 12:14:59 2023
"""
import logging
import copy
from biogeme.models import loglogit
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Variable,
    Expression,
    BelongsTo,
    ConditionalTermTuple,
    ConditionalSum,
    bioMultSum,
    exp,
    log,
    Elem,
    Numeric,
)
from .sampling_context import SamplingContext, LOG_PROBA_COL, MEV_WEIGHT

logger = logging.getLogger(__name__)


class GenerateModel:
    """Class in charge of generating the biogeme expression for the
        loglikelihood function
    """

    def __init__(self, context: SamplingContext):
        """Constructor

        :param context: contains all the information that is needed to
        perform the sampling of alternatives.
        :type context: SamplingContext
        """

        self.context = context
        self.utility_function = context.utility_function
        self.sample_size = context.sample_size
        self.attributes = context.attributes
        self.mev_prefix = context.mev_prefix

    def generate_utility(self, alt_id: int, for_mev: bool = False) -> Expression:
        """Generate the utility function for one alternative

        :param alt_id: id of the alternative
        
        :param for_mev: True if the utility is generated for the
            calculation of the MEV terms

        """
        copy_utility = copy.deepcopy(self.utility_function)
        if for_mev:
            if alt_id == 0:
                error_msg = 'Alternative 0 cannot be involved in the MEV term.'
                raise BiogemeError(error_msg)
            copy_utility.rename_elementary(
                self.attributes, suffix=f'_{alt_id}', prefix=self.mev_prefix
            )
        else:
            copy_utility.rename_elementary(self.attributes, suffix=f'_{alt_id}')
        return copy_utility

    def get_logit(self) -> Expression:
        """Returns the expression for the log likelihood of the logit model"""

        utilities = {
            i: self.generate_utility(i) - Variable(f'{LOG_PROBA_COL}_{i}')
            for i in range(self.sample_size)
        }

        return loglogit(utilities, None, 0)

    def get_nested_logit(
        self, nests: tuple[tuple[Expression, list[int]], ...]
    ) -> Expression:
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
        utilities = {
            i: self.generate_utility(i)
            for i in range(self.sample_size)
        }
        correction_terms = {
            i: -Variable(f'{LOG_PROBA_COL}_{i}') for i in range(self.sample_size)
        }

        dict_of_mev_terms = {}

        for nest in nests:
            mu_param = nest[0]
            list_of_alternatives = nest[1]
            self.context.check_valid_alternatives(set(list_of_alternatives))
            # We first build the MEV term using a sample of
            # alternatives. To build this term, we iterate from 1, as
            # we ignore the chosen alternative.
            list_of_terms = []
            for i in range(1, self.sample_size + 1):
                the_utility = self.generate_utility(i, for_mev=True)
                alternative_id = Variable(
                    f'{self.mev_prefix}{self.context.id_column}_{i}'
                )
                belong_to_nest = BelongsTo(alternative_id, set(list_of_alternatives))
                weight = Variable(f'{self.mev_prefix}{MEV_WEIGHT}_{i}')
                the_term = ConditionalTermTuple(
                    condition=belong_to_nest, term=weight * exp(mu_param * the_utility)
                )
                list_of_terms.append(the_term)
            mev_sum = ConditionalSum(list_of_terms)
            # We then add the term to relevant utilities. Note that we iterate here from 0.

            for i in range(self.sample_size):
                the_utility = utilities.get(i)
                if the_utility is None:
                    raise BiogemeError('Could not retrieve utility for alternative {i}')
                alternative_id = Variable(f'{self.context.id_column}_{i}')
                belong_to_nest = BelongsTo(alternative_id, set(list_of_alternatives))
                mev_term = (mu_param - 1.0) * the_utility + (
                    (1.0 / mu_param) - 1.0
                ) * log(mev_sum)
                the_mev_term = Elem({1: mev_term, 0: Numeric(0)}, belong_to_nest)
                if i in dict_of_mev_terms:
                    dict_of_mev_terms[i].append(the_mev_term)
                else:
                    dict_of_mev_terms[i] = [the_mev_term]

        corrected_utilities = {
            key: util + correction_terms[key] + bioMultSum(dict_of_mev_terms[key])
            for key, util in utilities.items()
        }
        return loglogit(corrected_utilities, None, 0)
