"""Bayesian estimation smoke tests for cross-nested logit models."""

import arviz as az
import numpy as np
import pandas as pd

import biogeme.biogeme as biogeme_module
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models import logcnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.parameters import Parameters


def test_bayesian_cross_nested_logit_smoke(monkeypatch):
    """Bayesian estimation builds and evaluates a cross-nested PyTensor graph."""
    dataframe = pd.DataFrame(
        {
            'choice': [1, 2, 3, 1],
            'av1': [1, 1, 1, 1],
            'av2': [1, 1, 1, 1],
            'av3': [0, 0, 1, 0],
        }
    )
    database = Database('cross_nested_logit_smoke', dataframe)

    nest_parameter = Beta(
        'mu_existing',
        value=1.4,
        lowerbound=1.0,
        upperbound=2.0,
        status=0,
    )
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=nest_parameter,
                dict_of_alpha={1: 1.0, 2: 1.0},
                name='public_transport',
            ),
            OneNestForCrossNestedLogit(
                nest_param=1.0,
                dict_of_alpha={3: 1.0},
                name='car',
            ),
        ),
    )
    log_probability = logcnl(
        util={1: 0.1, 2: 0.2, 3: -0.1},
        availability={
            1: Variable('av1'),
            2: Variable('av2'),
            3: Variable('av3'),
        },
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
        point = model.initial_point()
        assert np.isfinite(model.compile_logp()(point))
        assert np.isfinite(model.compile_dlogp()(point)).all()
        assert np.isfinite(model.compile_d2logp()(point)).all()
        samples = np.arange(chains * draws, dtype=float).reshape(chains, draws)
        samples = samples / 10.0 + 0.4
        posterior = {name: samples for name in biogeme.free_betas_names}
        return az.from_dict({'posterior': posterior}), False

    monkeypatch.setattr(biogeme_module, 'run_sampling', fake_run_sampling)

    results = biogeme.bayesian_estimation()

    assert 'mu_existing' in results.parameters
    assert results.parameters['mu_existing'].mean == 0.75
