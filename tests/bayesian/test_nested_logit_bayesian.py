"""Bayesian estimation smoke tests for nested logit models."""

import arviz as az
import numpy as np
import pandas as pd

import biogeme.biogeme as biogeme_module
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models.nested import lognested
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
from biogeme.parameters import Parameters


def test_bayesian_nested_logit_smoke(monkeypatch):
    """Bayesian estimation builds and evaluates a nested-logit PyTensor graph."""
    dataframe = pd.DataFrame({'choice': [1, 2, 3, 1]})
    database = Database('nested_logit_smoke', dataframe)

    nest_parameter = Beta(
        'lambda_nested',
        value=0.7,
        lowerbound=0.1,
        upperbound=1.0,
        status=0,
    )
    nests = NestsForNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForNestedLogit(
                nest_param=nest_parameter,
                list_of_alternatives=[1, 3],
                name='nested',
            ),
        ),
    )
    log_probability = lognested(
        util={1: 0.1, 2: 0.2, 3: -0.1},
        availability=None,
        nests=nests,
        choice=Variable('choice'),
    )

    biogeme = BIOGEME(
        database=database,
        formulas=log_probability,
        parameters=Parameters(),
        mcmc_sampling_strategy='pymc',
        sample_from_prior=False,
        bayesian_draws=4,
        warmup=1,
        chains=2,
        calculate_likelihood=False,
        calculate_waic=False,
        calculate_loo=False,
        generate_html=False,
        generate_yaml=False,
        generate_netcdf=False,
        save_iterations=False,
        seed=123,
    )

    def fake_run_sampling(
        *, model, draws, tune, chains, config, starting_values=None
    ):
        # Compiling the model here verifies the full BIOGEME -> PyMC -> PyTensor
        # path without making this unit test depend on an MCMC runtime.
        model.compile_logp()(model.initial_point())
        samples = np.arange(chains * draws, dtype=float).reshape(chains, draws)
        samples = samples / 10.0 + 0.4
        posterior = {
            name: samples for name in biogeme.free_betas_names
        }
        return az.from_dict({'posterior': posterior}), False

    monkeypatch.setattr(biogeme_module, 'run_sampling', fake_run_sampling)

    results = biogeme.bayesian_estimation()

    assert 'lambda_nested' in results.parameters
    assert results.parameters['lambda_nested'].mean == 0.75
