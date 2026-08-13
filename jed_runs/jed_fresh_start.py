#!/usr/bin/env python3
"""Remove generated JED/example state before a completely fresh run.

The cleaner is deliberately a dry run unless ``--apply`` is supplied.  It
removes generated files below ``docs/source/examples`` (including archived
results and generated ``.run`` files), Python/pytest caches, and the ignored
JED state directory.  Model/data/configuration source files are not targets.

Run this only after all Slurm jobs from the previous run have finished.  The
JED state directory contains the job records and diagnostics and is removed
without a backup; use ``jed_runs/jed_examples.py reset --apply`` instead when a
recoverable reset is wanted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:  # Support both ``python -m jed_runs.jed_fresh_start`` and direct execution.
    from .jed_examples import (
        ARTIFACT_SUFFIXES,
        DIAGNOSTIC_MARKDOWN_SUFFIX,
        EXAMPLES_ROOT,
        INPUT_CSV_NAMES,
        PROJECT_ROOT,
        load_config,
        state_root,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from jed_examples import (  # type: ignore[no-redef]
        ARTIFACT_SUFFIXES,
        DIAGNOSTIC_MARKDOWN_SUFFIX,
        EXAMPLES_ROOT,
        INPUT_CSV_NAMES,
        PROJECT_ROOT,
        load_config,
        state_root,
    )


CACHE_DIRECTORIES = {'__pycache__', '.pytest_cache'}
GENERATED_PLOT_FILES = {'autocorr.png', 'energy.png', 'rank.png', 'trace.png'}


def is_result_directory(name: str) -> bool:
    """Return whether *name* is an archive directory created for results."""
    return (
        name == 'saved_results'
        or name == 'saved_html'
        or name.startswith(('saved_results_', 'saved_html_'))
    )


def is_generated_file(path: Path, in_result_directory: bool) -> bool:
    """Return whether *path* is a generated example artifact."""
    if in_result_directory:
        return True
    if path.name in {'.DS_Store', *GENERATED_PLOT_FILES}:
        return True
    if path.name.endswith('.run'):
        return True
    if path.name.startswith('slurm-') or path.name.endswith('_slurm.out'):
        return True
    if path.name.startswith('revenue_') and path.suffix == '.txt':
        return True
    if path.suffix == '.csv' and path.name in INPUT_CSV_NAMES:
        return False
    if path.name.endswith(DIAGNOSTIC_MARKDOWN_SUFFIX):
        return True
    return path.suffix in ARTIFACT_SUFFIXES or path.suffix in {'.prof', '.pyc'}


def collect_targets(
    examples_root: Path, jed_state: Path
) -> tuple[list[Path], list[Path]]:
    """Collect generated files and directories without following symlinks.

    Archive directories are intentionally retained as empty directories after
    their files are removed: estimators and the JED harvester recreate them as
    needed.  Cache directories and the ignored JED state directory are removed
    entirely.
    """
    files: list[Path] = []
    directories: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    if jed_state.exists():
        if jed_state.is_symlink():
            raise ValueError(f'Refusing to remove symlinked JED state: {jed_state}')
        if not jed_state.is_dir():
            raise ValueError(f'JED state path is not a directory: {jed_state}')
        directories.append(jed_state)

    for current, directory_names, file_names in os.walk(
        examples_root,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        in_result_directory = any(
            is_result_directory(part)
            for part in current_path.relative_to(examples_root).parts
        )

        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                # A symlink may point outside the example tree; never follow
                # or unlink it automatically.
                kept_directories.append(directory_name)
            elif directory_name in CACHE_DIRECTORIES:
                directories.append(directory)
            else:
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                continue
            if is_generated_file(path, in_result_directory):
                files.append(path)

    return sorted(set(files)), sorted(
        set(directories), key=lambda path: len(path.parts), reverse=True
    )


def root_smoke_artifacts(project_root: Path) -> list[Path]:
    """Return narrowly named root-level Slurm smoke diagnostics."""
    return sorted(
        path
        for pattern in ('biogeme-smoke-*.err', 'biogeme-smoke-*.out')
        for path in project_root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    )


def validate_jed_state_path(jed_state: Path) -> None:
    """Refuse to remove a configured state directory outside this checkout."""
    project_root = PROJECT_ROOT.resolve()
    resolved_state = jed_state.resolve()
    try:
        resolved_state.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f'Refusing to remove JED state outside the repository: {jed_state}'
        ) from error
    if resolved_state == project_root:
        raise ValueError('Refusing to remove the repository itself as JED state.')


def relative(path: Path) -> str:
    """Format a path relative to the repository for stable output."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Remove generated example outputs, .run files, caches, and JED '
            'diagnostics before a fresh run.'
        )
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='delete the listed files/directories; otherwise only inspect them',
    )
    args = parser.parse_args(argv)

    configured_state = state_root(load_config())
    validate_jed_state_path(configured_state)
    files, directories = collect_targets(EXAMPLES_ROOT, configured_state)
    files = sorted(set(files + root_smoke_artifacts(PROJECT_ROOT)))
    print(f'Found generated files: {len(files)}')
    for path in files:
        print(f'  {relative(path)}')
    print(f'Found generated directories: {len(directories)}')
    for path in directories:
        print(f'  {relative(path)}/')

    if not args.apply:
        print('Dry run only. Re-run with --apply to delete these targets.')
        return 0

    for path in files:
        path.unlink()
    for path in directories:
        shutil.rmtree(path)
    print(f'Deleted {len(files)} files and {len(directories)} directories.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2) from error
