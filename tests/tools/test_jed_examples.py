import json
import os
from pathlib import Path
from types import SimpleNamespace

from jed_runs import jed_examples
from jed_runs.jed_examples import (
    Job,
    discover_jobs,
    generated_script,
    job_finish,
    job_start,
    load_config,
    select_jobs,
    status_label,
)
from jed_runs.jed_fresh_start import collect_targets, root_smoke_artifacts


def test_high_draw_panel_simulation_uses_memory_profile(tmp_path: Path):
    config = load_config()
    job = discover_jobs(config)['swissmetro/plot_b13_panel_simul.py']

    script = generated_script(config, job, tmp_path)

    assert job.profile == 'panel_simulation'
    assert '#SBATCH --cpus-per-task=36' in script
    assert '#SBATCH --mem-per-cpu=7000M' in script
    assert '--xla_force_host_platform_device_count=36' in script


def test_root_smoke_artifacts_are_explicitly_allowlisted(tmp_path: Path):
    error = tmp_path / 'biogeme-smoke-123.err'
    output = tmp_path / 'biogeme-smoke-123.out'
    unrelated = tmp_path / 'other.err'
    error.write_text('error')
    output.write_text('output')
    unrelated.write_text('keep')

    assert root_smoke_artifacts(tmp_path) == [error, output]


def test_h04_uses_high_memory_profile(tmp_path: Path):
    config = load_config()
    job = discover_jobs(config)['hybrid_choice_models/plot_h04_mode_lv_gauss_simult.py']

    script = generated_script(config, job, tmp_path)

    assert job.profile == 'hybrid'
    assert '#SBATCH --cpus-per-task=18' in script
    assert '#SBATCH --mem-per-cpu=7000M' in script
    assert 'export OPENBLAS_NUM_THREADS=1' in script
    assert 'export OMP_NUM_THREADS=1' in script
    assert 'WORK_DIRECTORY="$JOB_TMP/work"' in script
    assert 'rsync -a' in script
    assert 'rsync -a --delete --delete-excluded \\\n' in script
    assert "    --exclude='__pycache__/' \\\n" in script
    assert '--work-directory "$WORK_DIRECTORY"' in script


def test_slow_selection_excludes_light_jobs_and_keeps_dependencies():
    jobs = discover_jobs(load_config())

    selected = select_jobs(jobs, slow_only=True)

    assert selected
    assert all(job.profile != 'light' for job in selected.values())
    assert 'tutorials/plot_b05_simulation.py' not in selected
    assert 'indicators/plot_b02estimation.py' in selected


def test_explicit_selection_rejects_slow_flag():
    jobs = discover_jobs(load_config())

    try:
        select_jobs(jobs, ['tutorials/plot_b05_simulation.py'], slow_only=True)
    except ValueError as error:
        assert str(error) == 'Use either --only or --slow, not both.'
    else:  # pragma: no cover - assertion keeps the failure message explicit.
        raise AssertionError('Expected --only/--slow conflict to be rejected')


def test_explicit_selection_can_omit_dependencies_for_retries():
    jobs = discover_jobs(load_config())

    selected = select_jobs(
        jobs,
        ['hybrid_choice_models/plot_h03_mode_lv_gauss_seq.py'],
        include_dependencies=False,
    )

    assert list(selected) == ['hybrid_choice_models/plot_h03_mode_lv_gauss_seq.py']


def test_hybrid_sequential_job_declares_predecessor_and_input():
    jobs = discover_jobs(load_config())
    job = jobs['hybrid_choice_models/plot_h03_mode_lv_gauss_seq.py']

    assert job.dependencies == ('hybrid_choice_models/plot_h02_lv_mimic_gauss.py',)
    assert job.required_inputs == ('saved_results/plot_h02_lv_mimic_gauss.yaml',)


def test_dependency_contract_rejects_mismatched_producer_output():
    config = load_config()
    config['docs']['examples']['hybrid_choice_models/plot_h02_lv_mimic_gauss.py'][
        'expected_outputs'
    ] = ['wrong.yaml']

    try:
        discover_jobs(config)
    except ValueError as error:
        assert 'plot_h03_mode_lv_gauss_seq.py requires' in str(error)
    else:  # pragma: no cover - assertion keeps the failure message explicit.
        raise AssertionError('Expected the dependency contract to be rejected')


def test_status_labels_are_compact():
    assert status_label('finished without error') == 'OK'
    assert status_label('finished with errors') == 'ERROR'
    assert status_label('running') == 'RUNNING'
    assert status_label('scheduled and pending') == 'PENDING'
    assert status_label('not scheduled') == 'NOT_SCHEDULED'


def test_status_output_highlights_errors(monkeypatch, tmp_path: Path, capsys):
    run_directory = tmp_path / 'run-1'
    (run_directory / 'jobs' / 'plot_ok').mkdir(parents=True)
    (run_directory / 'jobs' / 'plot_ok' / 'completion.json').write_text(
        json.dumps({'exit_code': 0})
    )
    (run_directory / 'jobs' / 'plot_error').mkdir(parents=True)
    (run_directory / 'run.json').write_text(
        json.dumps(
            {
                'run_id': 'run-1',
                'created_at': '2026-08-10T16:45:45+00:00',
                'jobs': {
                    'plot_ok.py': {'job_id': '1', 'script': 'plot_ok.py'},
                    'plot_error.py': {'job_id': '2', 'script': 'plot_error.py'},
                },
            }
        )
    )
    monkeypatch.setattr(
        jed_examples,
        'global_statuses',
        lambda: {
            'plot_ok.py': {
                'label': 'OK',
                'record': {'job_id': '1'},
                'run_directory': run_directory,
                'detail': 'outputs validated',
            },
            'plot_error.py': {
                'label': 'ERROR',
                'record': {'job_id': '2'},
                'run_directory': run_directory,
                'detail': 'failed in test',
            },
        },
    )
    monkeypatch.setattr(
        jed_examples,
        'slurm_state',
        lambda job_id: (
            ('COMPLETED', '0:0|00:01:00||None')
            if job_id == '1'
            else ('FAILED', '1:0|00:00:01||None')
        ),
    )

    assert jed_examples.command_status(SimpleNamespace(run_id=None, verbose=False)) == 0
    output = capsys.readouterr().out
    assert 'Summary: ERROR=1 | OK=1' in output
    assert '  ERROR 2: plot_error.py' in output
    assert f'{"OK":10} {"1":12}' in output
    assert f'{"ERROR":10} {"2":12}' in output
    assert 'finished without error' not in output


def test_job_lifecycle_only_harvests_isolated_work_directory(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
    )

    assert job_start(job, state, work) == 0
    (work / 'model.yaml').write_text('result')
    # This artifact belongs to another job in the shared source directory and
    # must not be picked up by this job's harvest.
    (source / 'other_job.nc').write_bytes(b'foreign')

    assert job_finish(job, state, 0, work) == 0
    assert (source / 'saved_results' / 'model.yaml').read_text() == 'result'
    assert not (source / 'saved_results' / 'other_job.nc').exists()


def test_job_lifecycle_harvests_output_written_in_saved_results(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
    )

    assert job_start(job, state, work) == 0
    (work / 'saved_results').mkdir()
    (work / 'saved_results' / 'model.yaml').write_text('result')
    # Simulate a filesystem with coarser timestamp resolution than the
    # lifecycle clock, as on Windows. The snapshot comparison must still
    # recognize the newly created output.
    started_at_ns = json.loads((state / 'start.json').read_text())['started_at_ns']
    os.utime(
        work / 'saved_results' / 'model.yaml',
        ns=(started_at_ns - 1, started_at_ns - 1),
    )

    assert job_finish(job, state, 0, work) == 0
    assert (source / 'saved_results' / 'model.yaml').read_text() == 'result'


def test_job_lifecycle_does_not_archive_undeclared_netcdf(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml',),
    )

    assert job_start(job, state, work) == 0
    (work / 'model.yaml').write_text('result')
    (work / 'unused.nc').write_bytes(b'unused posterior draws')

    assert job_finish(job, state, 0, work) == 0
    assert (source / 'saved_results' / 'model.yaml').is_file()
    assert not (source / 'saved_results' / 'unused.nc').exists()


def test_job_lifecycle_archives_declared_netcdf(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml', 'model.nc'),
    )

    assert job_start(job, state, work) == 0
    (work / 'model.yaml').write_text('result')
    (work / 'model.nc').write_bytes(b'posterior draws')

    assert job_finish(job, state, 0, work) == 0
    assert (source / 'saved_results' / 'model.yaml').is_file()
    assert (source / 'saved_results' / 'model.nc').is_file()


def test_job_lifecycle_archives_declared_root_text_report(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_revenues.py',
        path=source / 'plot_revenues.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('revenue_1.00.txt',),
    )

    assert job_start(job, state, work) == 0
    (work / 'revenue_1.00.txt').write_text('revenue % lower % upper\n')

    assert job_finish(job, state, 0, work) == 0
    assert (source / 'revenue_1.00.txt').is_file()
    assert not (source / 'saved_results' / 'revenue_1.00.txt').exists()


def test_job_lifecycle_requires_all_declared_outputs(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml', 'model.html'),
    )

    assert job_start(job, state, work) == 0
    (work / 'model.yaml').write_text('result')

    assert job_finish(job, state, 0, work) != 0
    completion = json.loads((state / 'completion.json').read_text())
    assert 'model.html' in completion['diagnostics'][0]


def test_completed_job_is_not_ok_when_declared_archive_is_missing(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    run = tmp_path / 'run'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml',),
    )
    assert job_start(job, state, work) == 0
    (work / 'model.yaml').write_text('result')
    # Simulate a runner that reports completion before its archive copy.
    monkeypatch.setattr(jed_examples, 'discover_jobs', lambda: {job.script: job})
    assert job_finish(job, state, 0, work) == 0
    (source / 'saved_results' / 'model.yaml').unlink()
    job_state = run / 'jobs' / 'plot_model'
    job_state.mkdir(parents=True)
    for name in ('completion.json', 'diagnostic.json'):
        (job_state / name).write_text((state / name).read_text())
    monkeypatch.setattr(jed_examples, 'EXAMPLES_ROOT', source)
    monkeypatch.setattr(jed_examples, 'slurm_state', lambda _: ('COMPLETED', '0:0'))
    record = {'script': job.script, 'job_id': '123'}
    status, detail = jed_examples.classify_job(record, run)
    assert status == 'not done'
    assert 'model.yaml' in detail


def test_mark_ok_requires_declared_outputs(monkeypatch, tmp_path: Path):
    source = tmp_path / 'example'
    source.mkdir()
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml',),
    )
    monkeypatch.setattr(jed_examples, 'load_config', lambda: {'jobs': {}})
    monkeypatch.setattr(jed_examples, 'discover_jobs', lambda config: {job.script: job})
    monkeypatch.setenv('BIOGEME_JED_STATE_DIRECTORY', str(tmp_path / 'state'))

    try:
        jed_examples.command_mark_ok(
            SimpleNamespace(script=[job.script], source='laptop', note=None)
        )
    except ValueError as error:
        assert 'no result/report artifact' in str(error)
    else:  # pragma: no cover
        raise AssertionError('Expected mark-ok to validate declared outputs')


def test_invalidate_removes_declared_outputs(monkeypatch, tmp_path: Path):
    source = tmp_path / 'example'
    source.mkdir()
    output = source / 'saved_results' / 'model.yaml'
    output.parent.mkdir()
    output.write_text('old')
    job = Job(
        script='plot_model.py',
        path=source / 'plot_model.py',
        source='print("model")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
        expected_outputs=('model.yaml',),
    )
    monkeypatch.setattr(jed_examples, 'load_config', lambda: {'jobs': {}})
    monkeypatch.setattr(jed_examples, 'discover_jobs', lambda config: {job.script: job})
    monkeypatch.setenv('BIOGEME_JED_STATE_DIRECTORY', str(tmp_path / 'state'))

    assert (
        jed_examples.command_invalidate(
            SimpleNamespace(
                script=[job.script],
                all=False,
                no_dependents=False,
                reason='repair',
            )
        )
        == 0
    )
    assert not output.exists()


def test_job_lifecycle_harvests_markdown_diagnostic_report(tmp_path: Path):
    source = tmp_path / 'example'
    work = tmp_path / 'work'
    state = tmp_path / 'state'
    source.mkdir()
    work.mkdir()
    job = Job(
        script='plot_diagnostic.py',
        path=source / 'plot_diagnostic.py',
        source='print("diagnostic")',
        profile='light',
        dependencies=(),
        required_inputs=(),
        requires_artifacts=True,
    )

    assert job_start(job, state, work) == 0
    (work / 'model_monte_carlo_diagnostic.md').write_text('# Report')

    assert job_finish(job, state, 0, work) == 0
    assert (
        source / 'saved_results' / 'model_monte_carlo_diagnostic.md'
    ).read_text() == '# Report'


def test_fresh_start_targets_generated_files_but_preserves_sources(tmp_path: Path):
    examples = tmp_path / 'examples'
    examples.mkdir()
    (examples / 'plot_model.py').write_text('print("model")')
    (examples / 'data.csv').write_text('choice\n1\n')
    (examples / 'README.md').write_text('source documentation')
    (examples / 'model.yaml').write_text('generated')
    (examples / 'model_monte_carlo_diagnostic.md').write_text('generated report')
    (examples / 'plot_model.run').write_text('#SBATCH --job-name=test')
    (examples / 'model_slurm.out').write_text('output')
    (examples / 'saved_results').mkdir()
    (examples / 'saved_results' / 'model.yaml').write_text('archived')
    (examples / 'model').mkdir()
    (examples / 'model' / 'trace.png').write_bytes(b'plot')
    (examples / '__pycache__').mkdir()
    (examples / '__pycache__' / 'plot_model.pyc').write_bytes(b'cache')
    jed_state = tmp_path / '.jed_runs'
    (jed_state / 'run-1').mkdir(parents=True)
    (jed_state / 'run-1' / 'run.json').write_text('{}')

    files, directories = collect_targets(examples, jed_state)
    relative_files = {path.relative_to(tmp_path) for path in files}
    relative_directories = {path.relative_to(tmp_path) for path in directories}

    assert Path('examples/model.yaml') in relative_files
    assert Path('examples/model_monte_carlo_diagnostic.md') in relative_files
    assert Path('examples/plot_model.run') in relative_files
    assert Path('examples/model_slurm.out') in relative_files
    assert Path('examples/saved_results/model.yaml') in relative_files
    assert Path('examples/model/trace.png') in relative_files
    assert Path('examples/data.csv') not in relative_files
    assert Path('examples/README.md') not in relative_files
    assert Path('examples/plot_model.py') not in relative_files
    assert Path('examples/__pycache__') in relative_directories
    assert Path('.jed_runs') in relative_directories
