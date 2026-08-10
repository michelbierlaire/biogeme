#!/usr/bin/env python3
"""Create a digest and evidence report for failed JED example jobs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import jed_examples as runner

ERROR_MARKERS = (
    'traceback',
    'exception',
    'error:',
    'fatal',
    'failed',
    'no such file',
    'not found',
    'modulenotfounderror',
    'importerror',
    'filenotfounderror',
    'memoryerror',
    'out of memory',
    'oom',
    'killed',
    'timeout',
    'cancelled',
)
JOB_OUTPUT_NAMES = ('slurm.err', 'slurm.out')
JOB_METADATA_NAMES = ('diagnostic.json', 'completion.json', 'start.json')


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError as error:
        return f'[Unable to read {path.name}: {error}]'


def unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def job_directory(run_directory: Path, script: str) -> Path:
    slug = script.replace('/', '__').removesuffix('.py')
    return run_directory / 'jobs' / slug


def generated_script_path(run_directory: Path, script: str) -> Path:
    filename = script.replace('/', '__') + '.run'
    return run_directory / 'jobs' / filename


def slurm_evidence(job_id: str | None) -> str:
    if not job_id:
        return 'No Slurm job ID was recorded.'
    command = [
        'sacct',
        '--noheader',
        '--parsable2',
        '--allocations',
        '--jobs',
        job_id,
        '--format=JobID,State,ExitCode,Elapsed,MaxRSS,Reason',
    ]
    code, stdout, stderr = runner.run_external(command)
    if code != 0:
        return f'sacct was not available (exit code {code}): {stderr or stdout}'
    return stdout or 'sacct returned no accounting record.'


def output_files(job_path: Path) -> list[Path]:
    files = [job_path / name for name in (*JOB_OUTPUT_NAMES, *JOB_METADATA_NAMES)]
    if job_path.is_dir():
        files.extend(
            path
            for path in sorted(job_path.iterdir())
            if path.is_file()
            and path not in files
            and path.suffix.lower() in {'.err', '.json', '.log', '.out', '.txt'}
        )
    return files


def output_texts(job_path: Path) -> list[tuple[str, str]]:
    return [
        (path.name, read_text(path))
        for path in output_files(job_path)
        if path.is_file()
    ]


def digest_reasons(
    status: str,
    status_detail: str,
    completion: dict[str, Any],
    diagnostic: dict[str, Any],
    outputs: list[tuple[str, str]],
) -> list[str]:
    reasons: list[str] = []
    for source in (diagnostic, completion):
        diagnostics = source.get('diagnostics', [])
        if isinstance(diagnostics, list):
            reasons.extend(str(item) for item in diagnostics)
        reason = source.get('reason')
        if reason:
            reasons.append(str(reason))
        missing = source.get('missing_inputs', [])
        if missing:
            reasons.append('Missing required inputs: ' + ', '.join(map(str, missing)))

    exit_code = completion.get('exit_code')
    if exit_code not in (None, 0):
        reasons.append(f'Job completion exit code: {exit_code}.')
    if status_detail and status_detail != 'outputs validated':
        reasons.append(f'Slurm detail: {status_detail}')

    # Add a few high-signal lines when the wrapper could not record a Python
    # diagnostic (for example, an import failure before job-finish runs).
    for filename, text in outputs:
        for line in text.splitlines():
            candidate = line.strip()
            lowered = candidate.lower()
            if candidate and any(marker in lowered for marker in ERROR_MARKERS):
                reasons.append(f'{filename}: {candidate}')
                if sum(1 for item in reasons if item.startswith(f'{filename}:')) >= 5:
                    break

    if not reasons:
        reasons.append(f'Job classified as {status}; inspect the evidence below.')
    return unique(reasons)


def code_block(text: str) -> str:
    longest = max(
        (len(match.group(0)) for match in re.finditer(r'`+', text)),
        default=0,
    )
    fence = '`' * max(3, longest + 1)
    body = text.rstrip() or '(empty)'
    return f'{fence}text\n{body}\n{fence}'


def select_run(root: Path, run_id: str | None) -> Path:
    if run_id:
        selected = root / run_id
        if not (selected / 'run.json').is_file():
            raise ValueError(f'No run state found for run ID {run_id!r}.')
        return selected
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.name != 'resets' and (candidate / 'run.json').is_file():
            return candidate
    raise ValueError('No run state was found. Launch a run first.')


def build_report(run_directory: Path) -> tuple[str, int]:
    run_record = read_json(run_directory / 'run.json')
    jobs = run_record.get('jobs', {})
    if not isinstance(jobs, dict):
        raise ValueError(f'Invalid jobs record in {run_directory / "run.json"}.')

    failed: list[dict[str, Any]] = []
    for script, record in sorted(jobs.items()):
        if not isinstance(record, dict):
            record = {}
        directory = job_directory(run_directory, script)
        status, detail = runner.classify_job(record, run_directory)
        completion = read_json(directory / 'completion.json')
        diagnostic = read_json(directory / 'diagnostic.json')
        is_failed = (
            status == 'finished with errors'
            or diagnostic.get('category') == 'finished with errors'
            or completion.get('exit_code') not in (None, 0)
        )
        if is_failed:
            outputs = output_texts(directory)
            failed.append(
                {
                    'script': script,
                    'record': record,
                    'directory': directory,
                    'status': status,
                    'status_detail': detail,
                    'completion': completion,
                    'diagnostic': diagnostic,
                    'outputs': outputs,
                    'sacct': slurm_evidence(str(record.get('job_id', '')) or None),
                }
            )

    lines = [
        '# JED error report',
        '',
        f'- Run: `{run_record.get("run_id", run_directory.name)}`',
        f'- Created: `{run_record.get("created_at", "unknown")}`',
        f'- Failed jobs: **{len(failed)}**',
        '',
        '## Digest',
        '',
    ]
    if not failed:
        lines.append('No jobs were classified as `finished with errors`.')
        lines.append('')
        return chr(10).join(lines), 0

    for item in failed:
        lines.extend(
            [
                f'### `{item["script"]}`',
                '',
                f'- Status: `{item["status"]}`',
                f'- Job ID: `{item["record"].get("job_id", "-")}`',
            ]
        )
        reasons = digest_reasons(
            item['status'],
            item['status_detail'],
            item['completion'],
            item['diagnostic'],
            item['outputs'],
        )
        lines.extend(['- Digest:'])
        lines.extend(f'  - {reason}' for reason in reasons)
        lines.append('')

    lines.extend(['## Comprehensive evidence', ''])
    for item in failed:
        lines.extend([f'### `{item["script"]}`', ''])
        metadata = {
            'status': item['status'],
            'status_detail': item['status_detail'],
            'job_record': item['record'],
            'diagnostic': item['diagnostic'],
            'completion': item['completion'],
            'sacct': item['sacct'],
        }
        lines.extend(
            [
                '#### Runner and Slurm metadata',
                '',
                code_block(json.dumps(metadata, indent=2, sort_keys=True)),
                '',
            ]
        )
        for filename, text in item['outputs']:
            lines.extend([f'#### `{filename}`', '', code_block(text), ''])
        generated = generated_script_path(run_directory, item['script'])
        if generated.is_file():
            lines.extend(
                [
                    '#### Generated Slurm script',
                    '',
                    code_block(read_text(generated)),
                    '',
                ]
            )

    return chr(10).join(lines), len(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compile digest and full output for failed JED jobs.'
    )
    parser.add_argument('--run-id', help='run ID; defaults to the newest run')
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'report path; defaults to .jed_runs/<run-id>/error-report.md; '
            'use - for stdout'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = runner.load_config()
        root = runner.state_root(config)
        run_directory = select_run(root, args.run_id)
        report, failed_count = build_report(run_directory)
        if args.output == '-':
            print(report)
        else:
            output = (
                run_directory / 'error-report.md'
                if args.output is None
                else Path(args.output)
            )
            if not output.is_absolute():
                output = runner.PROJECT_ROOT / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report + chr(10), encoding='utf-8')
            print(f'Wrote {output}')
            print(f'Failed jobs included: {failed_count}')
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
