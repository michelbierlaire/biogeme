#!/usr/bin/env python3
"""Remove disposable outputs written into the documentation examples tree.

Sphinx-Gallery executes examples in ``docs/source/examples`` and some
estimators write their intermediate files there.  Those files are useful only
while the build is running.  Archived release fixtures in ``saved_results``
and ``saved_html`` are deliberately preserved.

The command is dry-run by default.  ``--apply`` removes only files that are
not present in ``HEAD``; this also safely removes generated files that were
accidentally staged, while leaving authored files untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / 'docs' / 'source' / 'examples'
RESULT_DIRECTORIES = {'saved_results', 'saved_html'}
CACHE_DIRECTORIES = {'__pycache__', '.pytest_cache'}
GENERATED_PLOT_FILES = {'autocorr.png', 'energy.png', 'rank.png', 'trace.png'}
GENERATED_SUFFIXES = {
    '.F12',
    '.csv',
    '.err',
    '.html',
    '.iter',
    '.log',
    '.nc',
    '.out',
    '.pareto',
    '.pickle',
    '.pkl',
    '.prof',
    '.pyc',
    '.tex',
    '.yaml',
}
INPUT_CSV_NAMES = {'data.csv', 'optima.csv'}


def _is_result_directory(part: str) -> bool:
    return part in RESULT_DIRECTORIES or part.startswith(
        ('saved_results_', 'saved_html_')
    )


def _head_paths() -> set[str]:
    """Return example paths tracked by the committed revision.

    Looking at ``HEAD`` rather than the index means a newly generated file
    that was accidentally staged is still recognized as disposable.
    """
    result = subprocess.run(
        [
            'git',
            'ls-tree',
            '-r',
            '--name-only',
            'HEAD',
            '--',
            'docs/source/examples',
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = 'docs/source/examples/'
    return {
        line.removeprefix(prefix)
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    }


def _staged_paths() -> list[str]:
    """Return paths currently staged below the examples tree."""
    result = subprocess.run(
        [
            'git',
            'diff',
            '--cached',
            '--name-only',
            '--diff-filter=ACMRT',
            '--',
            'docs/source/examples',
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def is_disposable(
    path: Path,
    tracked_paths: set[str],
    examples_root: Path = EXAMPLES_ROOT,
) -> bool:
    """Return whether *path* is a generated output, not a source or input."""
    relative = path.relative_to(examples_root)
    if relative.as_posix() in tracked_paths:
        return False
    if any(_is_result_directory(part) for part in relative.parts):
        return False
    if path.name in {'.DS_Store', *GENERATED_PLOT_FILES}:
        return True
    if path.name == 'biogeme.toml' or path.name.endswith('.run'):
        return True
    if path.name.startswith('slurm-') or path.name.endswith('_slurm.out'):
        return True
    if path.name.startswith('revenue_') and path.suffix == '.txt':
        return True
    if path.name.endswith('_monte_carlo_diagnostic.md'):
        return True
    if path.name.startswith('test~') and path.suffix == '.dat':
        return True
    if path.suffix not in GENERATED_SUFFIXES:
        return False
    return not (path.suffix == '.csv' and path.name in INPUT_CSV_NAMES)


def collect_targets(
    examples_root: Path = EXAMPLES_ROOT,
    tracked_paths: set[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Collect disposable files and cache directories below *examples_root*."""
    if examples_root != EXAMPLES_ROOT:
        # Tests and callers using a temporary tree can explicitly provide the
        # paths relative to that tree.  Production callers use HEAD paths.
        tracked_paths = tracked_paths or set()
    else:
        tracked_paths = _head_paths() if tracked_paths is None else tracked_paths

    files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(
        examples_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink() or directory_name not in CACHE_DIRECTORIES:
                kept_directories.append(directory_name)
            else:
                directories.append(directory)
        directory_names[:] = kept_directories
        for file_name in file_names:
            path = current_path / file_name
            if not path.is_symlink() and is_disposable(
                path, tracked_paths, examples_root
            ):
                files.append(path)
    if examples_root == EXAMPLES_ROOT:
        for staged in _staged_paths():
            relative = staged.removeprefix('docs/source/examples/')
            path = EXAMPLES_ROOT / relative
            if relative not in tracked_paths and is_disposable(
                path, tracked_paths, examples_root
            ):
                files.append(path)
    return sorted(set(files)), sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    )


def _unstage_if_needed(path: Path) -> None:
    relative = str(path.relative_to(PROJECT_ROOT))
    staged_check = subprocess.run(
        ['git', 'diff', '--cached', '--quiet', '--', relative],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if staged_check.returncode == 0:
        return
    if staged_check.returncode != 1:
        raise RuntimeError(
            f'Could not inspect the Git index for {relative}: '
            f'{staged_check.stderr.strip()}'
        )
    result = subprocess.run(
        ['git', 'restore', '--staged', '--', relative],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f'Cannot unstage generated file {relative}. '
            'Unstage it in Git/GitHub Desktop, then rerun the cleaner. '
            f'{result.stderr.strip()}'
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='delete listed outputs')
    args = parser.parse_args(argv)

    files, directories = collect_targets()
    print(f'Found {len(files)} disposable example file(s).')
    for path in files:
        print(f'  {path.relative_to(PROJECT_ROOT)}')
    print(f'Found {len(directories)} example cache directorie(s).')
    for path in directories:
        print(f'  {path.relative_to(PROJECT_ROOT)}/')

    if not args.apply:
        print('Dry run only. Re-run with --apply to remove these targets.')
        return 0

    for path in files:
        _unstage_if_needed(path)
        if path.exists() or path.is_symlink():
            path.unlink()
    for path in directories:
        shutil.rmtree(path)
    print(f'Removed {len(files)} file(s) and {len(directories)} directorie(s).')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f'error: {error}')
        raise SystemExit(2) from error
