from pathlib import Path

from tools.jed_examples import discover_jobs, generated_script, load_config
from tools.jed_fresh_start import collect_targets


def test_high_draw_panel_simulation_uses_memory_profile(tmp_path: Path):
    config = load_config()
    job = discover_jobs(config)['swissmetro/plot_b13_panel_simul.py']

    script = generated_script(config, job, tmp_path)

    assert job.profile == 'panel_simulation'
    assert '#SBATCH --cpus-per-task=36' in script
    assert '#SBATCH --mem-per-cpu=7000M' in script
    assert '--xla_force_host_platform_device_count=36' in script


def test_fresh_start_targets_generated_files_but_preserves_sources(tmp_path: Path):
    examples = tmp_path / 'examples'
    examples.mkdir()
    (examples / 'plot_model.py').write_text('print("model")')
    (examples / 'data.csv').write_text('choice\n1\n')
    (examples / 'model.yaml').write_text('generated')
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
    assert Path('examples/plot_model.run') in relative_files
    assert Path('examples/model_slurm.out') in relative_files
    assert Path('examples/saved_results/model.yaml') in relative_files
    assert Path('examples/model/trace.png') in relative_files
    assert Path('examples/data.csv') not in relative_files
    assert Path('examples/plot_model.py') not in relative_files
    assert Path('examples/__pycache__') in relative_directories
    assert Path('.jed_runs') in relative_directories
