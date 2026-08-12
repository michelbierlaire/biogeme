from pathlib import Path

from tools import docs_examples


def test_fast_profile_is_small_and_dependency_selection_is_explicit():
    specs = docs_examples.discover_specs(docs_examples.load_config())

    fast = docs_examples.select_specs(specs, 'fast', [])
    assert set(fast) == {
        'hybrid_choice_specs/plot_b00_normalization.py',
        'hybrid_choice_specs/plot_b01_model_specification.py',
        'montecarlo/plot_b01simple_integral.py',
        'programmers/plot_biogeme_logging.py',
        'programmers/plot_database.py',
        'programmers/plot_distributions.py',
        'programmers/plot_draws.py',
        'programmers/plot_expressions.py',
        'programmers/plot_filenames.py',
        'programmers/plot_loglikelihood.py',
        'programmers/plot_nests.py',
        'programmers/plot_segmentation.py',
        'programmers/plot_tools.py',
        'programmers/plot_version.py',
        'tutorials/plot_b01_first_model.py',
        'tutorials/plot_b03_importing_specification.py',
    }
    assert specs['tutorials/plot_b03_importing_specification.py'].expected_outputs == (
        'imported_model.yaml',
        'imported_model.html',
    )

    dependent = docs_examples.select_specs(
        specs, None, ['tutorials/plot_b05_simulation.py']
    )
    assert set(dependent) == {
        'tutorials/plot_b01_first_model.py',
        'tutorials/plot_b05_simulation.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(dependent)] == [
        'tutorials/plot_b01_first_model.py',
        'tutorials/plot_b05_simulation.py',
    ]


def test_copy_source_excludes_generated_outputs(tmp_path: Path):
    source = tmp_path / 'source'
    destination = tmp_path / 'workspace'
    (source / 'saved_results').mkdir(parents=True)
    (source / 'plot_example.py').write_text('print("ok")\n')
    (source / 'data.dat').write_text('input\n')
    (source / 'old.yaml').write_text('stale\n')
    (source / 'saved_results' / 'old.yaml').write_text('stale\n')

    docs_examples.copy_source(source, destination)

    assert (destination / 'plot_example.py').is_file()
    assert (destination / 'data.dat').is_file()
    assert not (destination / 'old.yaml').exists()
    assert not (destination / 'saved_results').exists()


def test_indicators_dependency_chain_and_artifact_contract():
    specs = docs_examples.discover_specs(docs_examples.load_config())
    selected = docs_examples.select_specs(specs, None, ['indicators/plot_b09wtp.py'])

    assert set(selected) == {
        'indicators/plot_b02estimation.py',
        'indicators/plot_b09wtp.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(selected)] == [
        'indicators/plot_b02estimation.py',
        'indicators/plot_b09wtp.py',
    ]
    assert specs['indicators/plot_b02estimation.py'].expected_outputs == (
        'b02estimation.yaml',
        'b02estimation.html',
    )
    assert specs['indicators/plot_b05revenues.py'].expected_outputs == (
        'revenue_1.00.txt',
    )


def test_generated_files_include_indicator_revenue_reports(tmp_path: Path):
    revenue = tmp_path / 'revenue_1.00.txt'
    revenue.write_text('1 % 2 % 3\n')

    assert docs_examples.generated_files(tmp_path) == [revenue]


def test_archived_estimation_outputs_are_harvested_and_validated(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    (workspace / 'saved_results').mkdir(parents=True)
    (workspace / 'saved_results' / 'model.yaml').write_text('result\n')
    spec = docs_examples.ExampleSpec(
        script='plot_model.py',
        source_directory=tmp_path,
        script_name='plot_model.py',
        mode='self_contained',
        profile='full',
        dependencies=(),
        required_inputs=(),
        expected_outputs=('model.yaml',),
        expected_output_globs=(),
        requires_artifacts=True,
        gallery=True,
    )

    harvested = docs_examples.harvest(workspace)

    assert harvested == ['saved_results/model.yaml']
    assert docs_examples.validate_outputs(spec, workspace, harvested) == []


def test_swissmetro_logit_dependency_chain():
    specs = docs_examples.discover_specs(docs_examples.load_config())
    selected = docs_examples.select_specs(
        specs, None, ['swissmetro/plot_b01d_logit_simul.py']
    )

    assert set(selected) == {
        'swissmetro/plot_b01a_logit.py',
        'swissmetro/plot_b01d_logit_simul.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(selected)] == [
        'swissmetro/plot_b01a_logit.py',
        'swissmetro/plot_b01d_logit_simul.py',
    ]
    assert specs['swissmetro/plot_b01a_logit.py'].expected_outputs == (
        'b01a_logit.yaml',
        'b01a_logit.html',
    )


def test_swissmetro_normal_mixture_dependency_chain():
    specs = docs_examples.discover_specs(docs_examples.load_config())
    selected = docs_examples.select_specs(
        specs, None, ['swissmetro/plot_b05c_normal_mixture_simul.py']
    )

    assert set(selected) == {
        'swissmetro/plot_b05a_normal_mixture.py',
        'swissmetro/plot_b05c_normal_mixture_simul.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(selected)] == [
        'swissmetro/plot_b05a_normal_mixture.py',
        'swissmetro/plot_b05c_normal_mixture_simul.py',
    ]
    assert specs['swissmetro/plot_b05a_normal_mixture.py'].expected_outputs == (
        'b05a_normal_mixture.yaml',
        'b05a_normal_mixture.html',
    )


def test_swissmetro_cnl_dependency_chain():
    specs = docs_examples.discover_specs(docs_examples.load_config())
    selected = docs_examples.select_specs(
        specs, None, ['swissmetro/plot_b11b_cnl_simul.py']
    )

    assert set(selected) == {
        'swissmetro/plot_b11a_cnl.py',
        'swissmetro/plot_b11b_cnl_simul.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(selected)] == [
        'swissmetro/plot_b11a_cnl.py',
        'swissmetro/plot_b11b_cnl_simul.py',
    ]
    assert specs['swissmetro/plot_b11a_cnl.py'].expected_outputs == (
        'b11a_cnl.yaml',
        'b11a_cnl.html',
    )


def test_swissmetro_panel_dependency_chain():
    specs = docs_examples.discover_specs(docs_examples.load_config())
    selected = docs_examples.select_specs(
        specs, None, ['swissmetro/plot_b13_panel_simul.py']
    )

    assert set(selected) == {
        'swissmetro/plot_b12_panel.py',
        'swissmetro/plot_b13_panel_simul.py',
    }
    assert [spec.script for spec in docs_examples.topological_order(selected)] == [
        'swissmetro/plot_b12_panel.py',
        'swissmetro/plot_b13_panel_simul.py',
    ]
    assert specs['swissmetro/plot_b12_panel.py'].expected_outputs == (
        'b12_panel.yaml',
        'b12_panel.html',
    )


def test_full_profile_has_artifact_contracts_for_deterministic_jobs():
    config = docs_examples.load_config()
    specs = docs_examples.discover_specs(config)
    jobs = docs_examples.load_jed_module().discover_jobs(config)

    contracted = {
        name
        for name, spec in specs.items()
        if spec.profile == 'full'
        and (spec.expected_outputs or spec.expected_output_globs)
    }
    assert len(contracted) >= 90
    assert specs['bayesian_swissmetro/plot_b01a_logit.py'].expected_outputs == (
        'b01a_logit.yaml',
        'b01a_logit.html',
        'b01a_logit.nc',
    )
    assert specs['bayesian_swissmetro/plot_b01b_logit.py'].expected_outputs == (
        'b01b_logit.yaml',
        'b01b_logit.html',
    )
    assert specs['bayesian_swissmetro/plot_b04_validation.py'].expected_outputs == (
        'b04validation.yaml',
        'b04validation.html',
    )
    assert specs['bayesian_swissmetro/plot_b05_normal_mixture.py'].expected_outputs == (
        'b05_normal_mixture.yaml',
        'b05_normal_mixture.html',
        'b05_normal_mixture.nc',
    )
    declared_netcdf = {
        output
        for spec in specs.values()
        for output in spec.expected_outputs
        if output.endswith('.nc')
    }
    assert declared_netcdf == {'b01a_logit.nc', 'b05_normal_mixture.nc'}
    assert specs[
        'swissmetro/plot_b05d_normal_mixture_all_algos.py'
    ].expected_output_globs == ('b05normal_mixture_algo_*.yaml',)
    assert specs['swissmetro/plot_b20_multiple_models.py'].expected_output_globs == (
        'b20multiple_models_*.yaml',
        'b20multiple_models_*.html',
    )

    # These jobs produce dynamic reports, CSV-only reports, or in-memory
    # post-processing results and therefore are intentionally not represented
    # by an exact importer contract yet.
    without_contract = {
        name
        for name, job in jobs.items()
        if job.requires_artifacts
        and not (specs[name].expected_outputs or specs[name].expected_output_globs)
    }
    assert without_contract == {
        'assisted/plot_b09post_processing.py',
        'swissmetro/plot_b01e_logit_all_algos.py',
        'swissmetro/plot_b21c_process_pareto.py',
        'swissmetro/plot_b22c_process_pareto.py',
    }
