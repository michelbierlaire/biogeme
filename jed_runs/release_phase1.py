#!/usr/bin/env python3
"""Incrementally prepare, submit, monitor, and finalize a JED release run.

The command is a dry run unless ``--apply`` is supplied.  Release state is
stored below the ignored ``.jed_runs`` directory; the user normally works
without specifying a release identifier.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
import time
from typing import Any

try:
    from .release_common import (
        PROJECT_ROOT,
        ensure_clean_tree,
        ensure_release,
        next_steps,
        now_iso,
        python_command,
        run_command,
        save_release,
    )
except ImportError:  # pragma: no cover - direct script execution
    from release_common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        ensure_clean_tree,
        ensure_release,
        next_steps,
        now_iso,
        python_command,
        run_command,
        save_release,
    )

sys.path.insert(0, str(PROJECT_ROOT))

from jed_runs import release_examples  # noqa: E402
from jed_runs.jed_examples import (  # noqa: E402
    discover_jobs,
    global_statuses,
    load_config,
    select_not_done_jobs,
    topological_jobs,
)


def slurm_jobs_running() -> bool:
    result = subprocess.run(
        ['squeue', '--noheader', '--user', getpass.getuser()],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print('WARNING: squeue is unavailable; skipping the running-job check.')
        return False
    return bool(result.stdout.strip())


def check_examples(*, apply: bool) -> dict[str, Any]:
    report = release_examples.inspect()
    if not report['inventory_exists']:
        if apply:
            release_examples.write_inventory(report['current'])
            print('Initialized the example inventory for this release.')
        else:
            print('The example inventory would be initialized on apply.')
        return report
    if report['new'] or report['changed'] or report['removed']:
        release_examples.print_report(report)
        raise RuntimeError(
            'The example inventory changed. Run release_examples.py --apply '
            'after reviewing the changes before continuing.'
        )
    if report['unresolved']:
        release_examples.print_report(report)
        raise RuntimeError('The example suite has unresolved release metadata.')
    return report


def print_launch_plan() -> int:
    config = load_config()
    jobs = discover_jobs(config)
    statuses = global_statuses(config)
    selected = select_not_done_jobs(jobs, statuses)
    if not selected:
        print('No NOT_DONE jobs are currently runnable.')
        return 0
    print(f'Jobs that would be submitted: {len(selected)}')
    for job in topological_jobs(selected):
        dependencies = ', '.join(job.dependencies) or '-'
        print(f'  {job.script} [{job.profile}] <- {dependencies}')
    return 0


def status_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in global_statuses().values():
        label = item['label']
        counts[label] = counts.get(label, 0) + 1
    return counts


def print_status_summary() -> dict[str, int]:
    counts = status_counts()
    summary = ' | '.join(f'{key}={value}' for key, value in sorted(counts.items()))
    print(f'Summary: {summary or "NO_JOBS"}')
    return counts


def phase1_run(args: argparse.Namespace) -> int:
    ensure_clean_tree(args.allow_dirty)
    report = check_examples(apply=args.apply)
    if report['new'] or report['unresolved']:
        return 1
    release = ensure_release(apply=args.apply, phase='phase1')
    phase = release.setdefault('phase1', {})

    if not phase.get('prepared'):
        if not args.skip_slurm_check and slurm_jobs_running():
            raise RuntimeError(
                'Slurm jobs are still running; refusing to prepare a fresh release.'
            )
        command = python_command('jed_runs/jed_fresh_start.py')
        if args.apply:
            command.append('--apply')
        code = run_command(command, apply=args.apply)
        if code:
            next_steps(
                [
                    'Review the cleanup diagnostics.',
                    'Rerun release_phase1.py run after correcting the problem.',
                ]
            )
            return code
        if args.apply:
            phase['prepared'] = True
            save_release(release)
    else:
        print('Preparation was already completed; reusing the existing state.')

    if args.apply:
        code = run_command(
            python_command('jed_runs/jed_examples.py', 'launch', '--not-done'),
            apply=True,
        )
        if code:
            next_steps(
                [
                    'Inspect the submission error.',
                    'Rerun release_phase1.py run --apply to submit only unfinished jobs.',
                ]
            )
            return code
        phase['last_submission_at'] = now_iso()
        save_release(release)
    else:
        print_launch_plan()

    if args.apply:
        next_steps(
            [
                'Monitor the submitted jobs with release_phase1.py status.',
                'After repairs, invalidate affected jobs and rerun release_phase1.py run --apply.',
            ]
        )
    else:
        next_steps(
            [
                'Review this plan and the example metadata.',
                'Run release_phase1.py run --apply to prepare and submit the jobs.',
            ]
        )
    return 0


def phase1_status(args: argparse.Namespace) -> int:
    code = run_command(
        python_command('jed_runs/jed_examples.py', 'status', '--verbose'),
        apply=True,
    )
    if code:
        next_steps(
            ['Rerun release_phase1.py status after correcting the status command.']
        )
        return code
    counts = print_status_summary()
    if counts.get('ERROR'):
        next_steps(
            [
                'Inspect the diagnostics with jed_runs/jed_error_report.py.',
                'Repair and invalidate failed examples, then rerun release_phase1.py run --apply.',
            ]
        )
    elif counts.get('RUNNING') or counts.get('PENDING'):
        next_steps(['Run release_phase1.py status again after the jobs progress.'])
    elif counts.get('NOT_DONE') or counts.get('NOT_SCHEDULED'):
        next_steps(['Run release_phase1.py run --apply to submit the remaining jobs.'])
    else:
        next_steps(
            [
                'Phase 1 is complete; run release_phase1.py finalize --apply.',
                'Then run release_phase2.py on the laptop.',
            ]
        )
    return 0


def phase1_monitor(args: argparse.Namespace) -> int:
    if not args.wait:
        return phase1_status(args)
    while True:
        counts = print_status_summary()
        if not (counts.get('RUNNING') or counts.get('PENDING')):
            return phase1_status(args)
        time.sleep(args.poll_seconds)


def phase1_finalize(args: argparse.Namespace) -> int:
    counts = status_counts()
    if not counts or any(label != 'OK' for label in counts):
        print_status_summary()
        next_steps(
            [
                'Resolve all ERROR, RUNNING, PENDING, and NOT_DONE jobs.',
                'Run release_phase1.py finalize --apply again.',
            ]
        )
        return 1
    release = ensure_release(apply=args.apply, phase='phase1')
    phase = release.setdefault('phase1', {})
    if phase.get('finalized'):
        print('Phase 1 was already finalized.')
    else:
        command = python_command('jed_runs/jed_cleanup.py')
        if args.apply:
            command.append('--apply')
        code = run_command(command, apply=args.apply)
        if code:
            next_steps(
                [
                    'Review the cleanup diagnostics.',
                    'Rerun release_phase1.py finalize --apply.',
                ]
            )
            return code
        if args.apply:
            phase['finalized'] = True
            save_release(release)
    next_steps(
        [
            'On the laptop, run release_phase2.py run for the transfer, import, and documentation build.',
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run', help='prepare and submit unfinished jobs')
    run.add_argument('--apply', action='store_true')
    run.add_argument('--allow-dirty', action='store_true')
    run.add_argument('--skip-slurm-check', action='store_true')
    run.set_defaults(function=phase1_run)

    status = subparsers.add_parser('status', help='show the incremental release status')
    status.set_defaults(function=phase1_status)

    monitor = subparsers.add_parser(
        'monitor', help='show status or wait for completion'
    )
    monitor.add_argument('--wait', action='store_true')
    monitor.add_argument('--poll-seconds', type=int, default=60)
    monitor.set_defaults(function=phase1_monitor)

    finalize = subparsers.add_parser(
        'finalize', help='clean root artifacts after all jobs are OK'
    )
    finalize.add_argument('--apply', action='store_true')
    finalize.set_defaults(function=phase1_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        next_steps(
            [
                'Resolve the reported issue without resetting successful jobs.',
                'Rerun the same release_phase1.py command; it will resume incrementally.',
            ]
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
