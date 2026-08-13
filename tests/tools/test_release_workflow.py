from pathlib import Path
from types import SimpleNamespace

from jed_runs import release_examples, release_phase1, release_phase2, release_reset
from jed_runs.release_common import (
    DirtyWorkingTreeError,
    ensure_clean_tree,
    ensure_release,
    run_command,
)


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


def test_dirty_tree_explains_how_to_preserve_archived_results(monkeypatch):
    monkeypatch.setattr(
        'jed_runs.release_common.git_status',
        lambda: ['?? docs/source/examples/indicators/saved_results/model.yaml'],
    )

    try:
        ensure_clean_tree()
    except RuntimeError as error:
        message = str(error)
    else:  # pragma: no cover - assertion keeps the failure message explicit.
        raise AssertionError('Expected dirty-tree protection to reject the run')

    assert 'jed_commit_results.py --dry-run' in message
    assert 'git stash push --include-untracked' in message
    assert 'Do not use release_reset.py or jed_fresh_start.py' in message


def test_generated_artifacts_are_allowed(monkeypatch, capsys):
    monkeypatch.setattr(
        'jed_runs.release_common.git_status',
        lambda: ['?? docs/source/examples/indicators/saved_results/model.yaml'],
    )

    ensure_clean_tree(allow_generated=True)
    assert 'ignoring 1 generated release artifact' in capsys.readouterr().out


def test_generated_smoke_diagnostic_is_allowed(monkeypatch, capsys):
    monkeypatch.setattr(
        'jed_runs.release_common.git_status',
        lambda: ['?? biogeme-smoke-65991538.err'],
    )

    ensure_clean_tree(allow_generated=True)
    assert 'ignoring 1 generated release artifact' in capsys.readouterr().out


def test_authored_change_still_blocks_with_generated_artifacts(monkeypatch):
    monkeypatch.setattr(
        'jed_runs.release_common.git_status',
        lambda: [
            '?? biogeme-smoke-65991538.err',
            ' M src/biogeme/biogeme.py',
        ],
    )

    try:
        ensure_clean_tree(allow_generated=True)
    except DirtyWorkingTreeError as error:
        message = str(error)
    else:  # pragma: no cover - assertion keeps the failure message explicit.
        raise AssertionError('Expected authored changes to block the run')

    assert 'src/biogeme/biogeme.py' in message


def test_existing_jed_attempts_are_detected(monkeypatch):
    monkeypatch.setattr(
        release_phase1,
        'global_statuses',
        lambda: {
            'family/plot_example.py': {'record': {'job_id': '12345'}},
            'family/plot_other.py': {'record': {}},
        },
    )

    assert release_phase1.has_existing_jed_attempts() is True


def test_incomplete_phase2_state_from_older_revision_is_replaced(monkeypatch, capsys):
    monkeypatch.setattr('jed_runs.release_common.git_revision', lambda: 'new-revision')
    monkeypatch.setattr('jed_runs.release_common.manifest_hash', lambda: 'manifest')
    monkeypatch.setattr(
        'jed_runs.release_common.load_current_release',
        lambda: {
            'release_id': 'old',
            'revision': 'old-revision',
            'manifest_sha256': 'manifest',
            'phase': 'phase2',
            'phase1': {},
            'phase2': {'transferred': True},
        },
    )

    release = ensure_release(apply=False, phase='phase2')

    assert release['revision'] == 'new-revision'
    assert release['phase1'] == {}
    assert 'starting a new local Phase 2 attempt' in capsys.readouterr().out


def test_active_phase1_state_from_older_revision_is_protected(monkeypatch):
    monkeypatch.setattr('jed_runs.release_common.git_revision', lambda: 'new-revision')
    monkeypatch.setattr('jed_runs.release_common.manifest_hash', lambda: 'manifest')
    monkeypatch.setattr(
        'jed_runs.release_common.load_current_release',
        lambda: {
            'release_id': 'old',
            'revision': 'old-revision',
            'manifest_sha256': 'manifest',
            'phase1': {'launched': True},
            'phase2': {},
        },
    )

    try:
        ensure_release(apply=False, phase='phase2')
    except RuntimeError as error:
        assert 'different Git revision' in str(error)
    else:  # pragma: no cover - assertion keeps the failure explicit.
        raise AssertionError('An active Phase 1 release must not be replaced')


def test_reset_allows_laptop_without_slurm(monkeypatch, capsys):
    def missing_squeue(*args, **kwargs):
        raise FileNotFoundError('squeue')

    monkeypatch.setattr(release_reset.subprocess, 'run', missing_squeue)

    assert release_reset.slurm_jobs_running() is False
    assert 'squeue is not available' in capsys.readouterr().err


def test_phase1_adopts_existing_attempts_without_resetting(monkeypatch, capsys):
    monkeypatch.setattr(
        release_phase1, 'ensure_clean_tree', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        release_phase1,
        'check_examples',
        lambda **kwargs: {'new': [], 'changed': [], 'removed': [], 'unresolved': []},
    )
    monkeypatch.setattr(release_phase1, 'has_existing_jed_attempts', lambda: True)
    monkeypatch.setattr(
        release_phase1,
        'ensure_release',
        lambda **kwargs: {'phase1': {}},
    )
    monkeypatch.setattr(release_phase1, 'print_launch_plan', lambda: 0)

    args = SimpleNamespace(apply=False, skip_slurm_check=False)
    assert release_phase1.phase1_run(args) == 0
    assert (
        'adopting them and skipping the fresh-start cleanup' in capsys.readouterr().out
    )


def test_phase1_dirty_tree_next_steps_distinguish_existing_release(monkeypatch, capsys):
    monkeypatch.setattr(
        release_phase1,
        'ensure_clean_tree',
        lambda **kwargs: (_ for _ in ()).throw(DirtyWorkingTreeError('dirty tree')),
    )

    assert release_phase1.main(['run']) == 2
    output = capsys.readouterr()
    assert 'authored or unrecognized files' in output.out
    assert 'release_phase1.py status instead' in output.out


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
    assert '--delete' in command
    assert '--exclude=*.nc' in command
    assert '--include=*.yaml' in command
    assert '--include=*.html' in command
    assert command[-2] == 'user@host:/home/user/examples/'
    assert '--stats' in command


def test_phase2_reports_missing_artifacts_by_producer(monkeypatch, capsys):
    gaps = {'tutorials/plot_b01_first_model.py': ['first_model.yaml']}
    monkeypatch.setattr(release_phase2, 'staged_artifact_gaps', lambda _: gaps)
    release_phase2.print_artifact_recovery(gaps)
    output = capsys.readouterr().out
    assert 'Missing staged artifacts by producer:' in output
    assert 'tutorials/plot_b01_first_model.py: first_model.yaml' in output


def test_incomplete_recorded_stage_forces_refresh(monkeypatch, tmp_path):
    stage = tmp_path / 'examples'
    source = 'user@host:/home/user/examples'
    release = {
        'phase2': {
            'source': source,
            'transferred': True,
            'imported': True,
            'built': True,
            'artifact_root': str(stage),
        }
    }
    calls = []
    monkeypatch.setattr(release_phase2, 'ensure_clean_tree', lambda **_: None)
    monkeypatch.setattr(release_phase2, 'ensure_release', lambda **_: release)
    monkeypatch.setattr(release_phase2, 'save_release', lambda _: None)
    monkeypatch.setattr(release_phase2, 'next_steps', lambda _: None)
    monkeypatch.setattr(release_phase2, 'ensure_source', lambda *args, **kwargs: calls.append(args) or stage)
    monkeypatch.setattr(release_phase2, 'staged_artifacts_complete', lambda _: False)
    monkeypatch.setattr(release_phase2, 'clean_docs', lambda **_: 0)
    monkeypatch.setattr(release_phase2, 'import_artifacts', lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(release_phase2, 'build_docs', lambda **_: 0)

    args = SimpleNamespace(source=source, stage=str(stage), apply=True)
    assert release_phase2.phase2_run(args) == 0
    assert calls == [(source, stage.resolve())]


def test_phase2_refreshes_incomplete_transfer_before_import(monkeypatch, tmp_path):
    stage = tmp_path / 'examples'
    source = 'user@host:/home/user/examples'
    calls = []
    events = []
    release = {
        'phase2': {
            'source': source,
            'transferred': True,
            'artifact_root': str(stage),
        }
    }
    monkeypatch.setattr(release_phase2, 'ensure_clean_tree', lambda **_: None)
    monkeypatch.setattr(release_phase2, 'ensure_release', lambda **_: release)
    monkeypatch.setattr(release_phase2, 'save_release', lambda _: None)
    monkeypatch.setattr(release_phase2, 'next_steps', lambda _: None)
    monkeypatch.setattr(release_phase2, 'clean_docs', lambda **_: 0)

    def refresh(remote, destination, *, apply):
        calls.append((remote, destination, apply))
        events.append('transfer')
        return destination

    monkeypatch.setattr(release_phase2, 'ensure_source', refresh)
    monkeypatch.setattr(release_phase2, 'clean_docs', lambda **_: events.append('clean') or 0)
    monkeypatch.setattr(
        release_phase2,
        'import_artifacts',
        lambda *_args, **_kwargs: events.append('import') or 1,
    )
    args = SimpleNamespace(source=source, stage=str(stage), apply=True)

    assert release_phase2.phase2_run(args) == 1
    assert calls == [(source, stage.resolve(), True)]
    assert events == ['transfer', 'clean', 'import']
