#!/usr/bin/env python3
"""Remove root-level generated artifacts after a JED example run.

JED jobs copy their results into ``saved_results`` and ``saved_html`` while
leaving the original files in place.  Run this command only after all jobs in
the run have completed and the archived results have been reviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jed_examples import (
    ARTIFACT_SUFFIXES,
    EXAMPLES_ROOT,
    INPUT_CSV_NAMES,
    PROJECT_ROOT,
    RESULT_DIRECTORIES,
    copy_output,
    discover_jobs,
)


def generated_root_artifacts() -> list[Path]:
    """Return generated artifacts outside ``saved_results``/``saved_html``."""
    candidates: list[Path] = []
    for path in EXAMPLES_ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(EXAMPLES_ROOT).parts
        if any(part in RESULT_DIRECTORIES for part in relative_parts):
            continue
        if path.name.endswith('.run'):
            continue
        if path.suffix in ARTIFACT_SUFFIXES and not (
            path.suffix == '.csv' and path.name in INPUT_CSV_NAMES
        ):
            candidates.append(path)
        elif path.name.startswith('slurm-') or path.name.endswith('_slurm.out'):
            candidates.append(path)
        elif path.name.startswith('revenue_') and path.suffix == '.txt':
            candidates.append(path)
    for pattern in ('biogeme-smoke-*.err', 'biogeme-smoke-*.out'):
        candidates.extend(
            path
            for path in PROJECT_ROOT.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    return sorted(candidates)


def archive_declared_root_outputs() -> list[Path]:
    """Copy declared root outputs into their persistent archive first.

    Older jobs may have completed before the runner started archiving every
    declared output.  Promote those root-level files before cleanup so a
    successful job is not made impossible to import merely by finalization.
    """
    promoted: list[Path] = []
    for job in discover_jobs().values():
        for expected in job.expected_outputs:
            relative = Path(expected)
            if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
                continue
            source = job.directory / relative
            if not source.is_file():
                continue
            destination_directory = (
                job.directory / 'saved_html'
                if source.suffix.lower() == '.html'
                else job.directory / 'saved_results'
            )
            destination = destination_directory / source.name
            if source.resolve() != destination.resolve():
                copy_output(source, destination)
                promoted.append(destination)
    return promoted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Remove generated root-level example artifacts while preserving '
            'saved_results and saved_html.'
        )
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='delete the listed files; without this flag, only inspect them',
    )
    args = parser.parse_args(argv)

    candidates = generated_root_artifacts()
    print(f'Found {len(candidates)} root-level generated artifact(s).')
    for path in candidates:
        print(f'  {path.relative_to(PROJECT_ROOT)}')

    if not args.apply:
        print('Dry run only. Re-run with --apply to delete these files.')
        return 0

    promoted = archive_declared_root_outputs()
    if promoted:
        print(f'Archived {len(promoted)} declared root-level output(s).')
    for path in candidates:
        path.unlink()
    print(f'Deleted {len(candidates)} root-level generated artifact(s).')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except OSError as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2) from error
