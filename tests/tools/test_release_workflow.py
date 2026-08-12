from pathlib import Path
from types import SimpleNamespace

from jed_runs import release_examples, release_phase2
from jed_runs.release_common import run_command


def test_release_example_inventory_establishes_baseline(tmp_path: Path, monkeypatch):
    examples = tmp_path / 'examples'
    examples.mkdir()
    script = examples / 'plot_first.py'
    script.write_text('""".. _plot_first:\n\nFirst example\n"""\n')
    inventory = tmp_path / 'inventory.json'
    monkeypatch.setattr(release_examples, 'EXAMPLES_ROOT', examples)
    monkeypatch.setattr(release_examples, 'INVENTORY_PATH', inventory)
    monkeypatch.setattr(
        release_examples,
        'load_config',
        lambda: {'docs': {'examples': {}}, 'jobs': {}},
    )
    monkeypatch.setattr(
        release_examples,
        'discover_jobs',
        lambda config: {
            'plot_first.py': SimpleNamespace(
                profile='light',
                requires_artifacts=False,
                dependencies=(),
                required_inputs=(),
            )
        },
    )

    first = release_examples.inspect()
    assert first['inventory_exists'] is False
    assert first['new'] == []
    release_examples.write_inventory(first['current'])

    second = release_examples.inspect()
    assert second['inventory_exists'] is True
    assert second['new'] == []

    new_script = examples / 'plot_second.py'
    new_script.write_text('""".. _plot_second:\n\nSecond example\n"""\n')
    monkeypatch.setattr(
        release_examples,
        'discover_jobs',
        lambda config: {
            'plot_first.py': SimpleNamespace(
                profile='light',
                requires_artifacts=False,
                dependencies=(),
                required_inputs=(),
            ),
            'plot_second.py': SimpleNamespace(
                profile='light',
                requires_artifacts=False,
                dependencies=(),
                required_inputs=(),
            ),
        },
    )
    report = release_examples.inspect()
    assert [record['script'] for record in report['new']] == ['plot_second.py']
    assert report['unresolved'] == []


def test_release_example_inventory_flags_missing_output_contract(
    tmp_path: Path, monkeypatch
):
    examples = tmp_path / 'examples'
    examples.mkdir()
    script = examples / 'plot_estimation.py'
    script.write_text('""".. _plot_estimation:\n\nEstimator\n"""\n')
    inventory = tmp_path / 'inventory.json'
    inventory.write_text('{"scripts": {}}')
    monkeypatch.setattr(release_examples, 'EXAMPLES_ROOT', examples)
    monkeypatch.setattr(release_examples, 'INVENTORY_PATH', inventory)
    monkeypatch.setattr(
        release_examples,
        'load_config',
        lambda: {'docs': {'examples': {}}, 'jobs': {}},
    )
    monkeypatch.setattr(
        release_examples,
        'discover_jobs',
        lambda config: {
            'plot_estimation.py': SimpleNamespace(
                profile='standard',
                requires_artifacts=True,
                dependencies=(),
                required_inputs=(),
            )
        },
    )

    report = release_examples.inspect()
    assert report['unresolved']
    assert report['unresolved'][0]['script'] == 'plot_estimation.py'


def test_dry_run_command_does_not_execute(monkeypatch, capsys):
    def fail(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError('dry run executed a subprocess')

    monkeypatch.setattr('subprocess.run', fail)
    assert run_command(['sbatch', 'example.run'], apply=False) == 0
    assert '[PLAN] sbatch example.run' in capsys.readouterr().out


def test_phase2_parser_accepts_common_options_after_subcommand():
    parser = release_phase2.build_parser()
    args = parser.parse_args(
        ['run', '--source', 'user@host:/examples', '--stage', '/tmp/stage']
    )
    assert args.source == 'user@host:/examples'
    assert args.stage == '/tmp/stage'


def test_rsync_transfer_is_resumable():
    command = release_phase2.transfer_command(
        'user@host:/home/user/examples', Path('/tmp/stage')
    )
    assert '--partial' in command
    assert '--exclude=*.nc' in command
    assert command[-2] == 'user@host:/home/user/examples/'
