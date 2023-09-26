"""Module design for the estimation of models using with samples of alternatives.

It assumes that the generation of the choice set has been performed be
the module :mod:'choice_set_generation'

:author: Michel Bierlaire
:date: Mon Sep 11 18:27:36 2023

"""
import os
import logging
import copy
import pandas as pd
import biogeme.database as db
from biogeme.biogeme import BIOGEME
from biogeme import models
from biogeme.expressions import Variable, Expression
from biogeme.choice_set_generation import ChoiceSetsGeneration
from biogeme.sampling_context import SamplingContext, LOG_PROBA_COL

logger = logging.getLogger(__name__)


def logit(
        sampling_context:SamplingContext,
        datafile: str,
        regenerate_choice_sets: bool=False) -> BIOGEME:

    if not os.path.exists(datafile) or regenerate_choice_sets:
        generation = ChoiceSetsGeneration(sampling_context)
        database = generation.sample_and_merge(
            datafile=datafile,
            overwrite=regenerate_choice_sets
        )
    else:
        df = pd.read_csv(datafile)
        database = db.Database('sample_of_alternatives', df)
        
        
    def generate_utility(index: int) -> Expression:
        copy_utility = copy.deepcopy(sampling_context.utility_function)
        copy_utility.rename_elementary(sampling_context.alternatives.columns, suffix=f'_{index}')
        combined_variables = [
            combined_variable.name for combined_variable in sampling_context.combined_variables
        ]
        copy_utility.rename_elementary(combined_variables, suffix=f'_{index}')
            
        return copy_utility - Variable(f'{LOG_PROBA_COL}_{index}')
    
    V = {index: generate_utility(index) for index in range(sampling_context.sample_size)}

    logprob = models.loglogit(V, None, 0)

    the_biogeme = BIOGEME(database, logprob)

    return the_biogeme
