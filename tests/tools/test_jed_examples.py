from pathlib import Path

from jed_runs.jed_examples import (
    Job,
    discover_jobs,
    generated_script,
    job_finish,
    job_start,
    load_config,
)
from jed_runs.jed_fresh_start import collect_targets


def test_high_draw_panel_simulation_uses_memory_profile(tmp_path: Path):
    config = load_config()
    job = discover_jobs(config)['swissmetro/plot_b13_panel_simul.py']

    script = generated_script(config, job, tmp_path)

    assert job.profile == 'panel_simulation'
    assert '#SBATCH --cpus-per-task=36' in script
    assert '#SBATCH --mem-per-cpu=7000M' in script
    assert '--xla_force_host_platform_device_count=36' in script


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
