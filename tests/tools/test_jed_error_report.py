import json
from pathlib import Path

from jed_runs import jed_error_report


def write_run(root: Path, run_id: str, created_at: str, jobs: dict) -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / 'run.json').write_text(
        json.dumps({'run_id': run_id, 'created_at': created_at, 'jobs': jobs})
    )


def test_aggregate_report_combines_partial_runs(monkeypatch, tmp_path: Path):
    write_run(
        tmp_path,
        'run-1',
        '2026-08-10T10:00:00+00:00',
        {
            'ok.py': {'script': 'ok.py', 'state': 'ok'},
            'retry.py': {'script': 'retry.py', 'state': 'error'},
        },
    )
    write_run(
        tmp_path,
        'run-2',
        '2026-08-10T11:00:00+00:00',
        {
            'retry.py': {'script': 'retry.py', 'state': 'ok'},
        },
    )

    monkeypatch.setattr(
        jed_error_report.runner,
        'discover_jobs',
        lambda: {'ok.py': object(), 'retry.py': object(), 'never.py': object()},
    )
    monkeypatch.setattr(
        jed_error_report.runner,
        'classify_job',
        lambda record, run_directory: (
            ('finished without error', 'outputs validated')
            if record['state'] == 'ok'
            else ('finished with errors', 'failed in test')
        ),
    )

    report, unresolved = jed_error_report.build_aggregate_report(tmp_path)

    assert unresolved == 1
    assert '**OK** `ok.py` (run `run-1`)' in report
    assert '**OK** `retry.py` (run `run-2`)' in report
    assert '**NOT_DONE** `never.py` (run `-`)' in report


def test_dry_run_does_not_replace_real_attempt(tmp_path: Path):
    write_run(
        tmp_path,
        'run-1',
        '2026-08-10T10:00:00+00:00',
        {'ok.py': {'script': 'ok.py', 'state': 'ok'}},
    )
    write_run(
        tmp_path,
        'run-2',
        '2026-08-10T11:00:00+00:00',
        {
            'ok.py': {
                'script': 'ok.py',
                'status': 'not scheduled',
                'diagnostic': 'dry run',
            }
        },
    )

    _, attempts = jed_error_report.aggregate_attempts(tmp_path)

    assert attempts['ok.py'][0].name == 'run-1'
