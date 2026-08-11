#!/usr/bin/env python3
"""Commit archived JED example results to the repository.

Only files below ``docs/source/examples/**/saved_results`` and
``docs/source/examples/**/saved_html`` are staged.  The command refuses to
run when unrelated files are already staged, which prevents an accidental
commit of other work in the checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / 'docs' / 'source' / 'examples'
RESULT_DIRECTORIES = {'saved_results', 'saved_html'}
DEFAULT_COMMIT_MESSAGE = 'Update JED example results'


def run_git(
    repository_root: Path, arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run Git in *repository_root* and capture its text output."""
    return subprocess.run(
        ['git', *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )


def git_output(repository_root: Path, arguments: list[str]) -> str:
    """Run Git and return stdout, raising a useful error on failure."""
    result = run_git(repository_root, arguments)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f'git {" ".join(arguments)} failed with exit code '
            f'{result.returncode}: {detail}'
        )
    return result.stdout


def verify_repository(repository_root: Path) -> None:
    """Ensure that *repository_root* is the top-level Git checkout."""
    root = repository_root.resolve()
    reported_root = Path(
        git_output(root, ['rev-parse', '--show-toplevel']).strip()
    ).resolve()
    if reported_root != root:
        raise RuntimeError(
            f'Expected Git repository root {root}, but Git reported {reported_root}.'
        )


def result_directories(repository_root: Path) -> list[Path]:
    """Find archived-result directories below the documentation examples."""
    examples_root = repository_root / 'docs' / 'source' / 'examples'
    if not examples_root.is_dir():
        raise RuntimeError(f'Examples directory does not exist: {examples_root}')
    return sorted(
        path
        for path in examples_root.rglob('*')
        if path.is_dir()
        and not path.is_symlink()
        and path.name in RESULT_DIRECTORIES
    )


def relative_targets(repository_root: Path, directories: list[Path]) -> list[str]:
    """Return Git pathspecs for the archived-result directories."""
    return [path.relative_to(repository_root).as_posix() for path in directories]


def staged_paths(repository_root: Path) -> set[str]:
    """Return paths currently staged in the Git index."""
    output = git_output(repository_root, ['diff', '--cached', '--name-only', '-z'])
    return {path for path in output.split('\0') if path}


def is_under_target(path: str, targets: list[str]) -> bool:
    """Return whether a Git path belongs to one of the target directories."""
    return any(path == target or path.startswith(f'{target}/') for target in targets)


def ensure_no_unrelated_staged_changes(
    staged: set[str], targets: list[str]
) -> None:
    """Refuse to commit if the index already contains unrelated changes."""
    unrelated = sorted(path for path in staged if not is_under_target(path, targets))
    if unrelated:
        formatted = '\n'.join(f'  {path}' for path in unrelated)
        raise RuntimeError(
            'Refusing to commit because unrelated files are already staged:\n'
            f'{formatted}\nUnstage them first, then rerun this command.'
        )


def print_status(repository_root: Path, targets: list[str]) -> None:
    """Print the working-tree status restricted to archived results."""
    result = run_git(repository_root, ['status', '--short', '--', *targets])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f'git status failed: {detail}')
    print(result.stdout.rstrip() or '  No changes found.')


def commit_results(
    repository_root: Path,
    message: str = DEFAULT_COMMIT_MESSAGE,
    dry_run: bool = False,
) -> int:
    """Stage and commit all archived JED results.

    The return value is zero when no changes are found or a commit succeeds.
    """
    root = repository_root.resolve()
    if not message.strip():
        raise ValueError('The commit message cannot be empty.')
    verify_repository(root)
    directories = result_directories(root)
    targets = relative_targets(root, directories)
    if not targets:
        print(f'No {sorted(RESULT_DIRECTORIES)} directories were found.')
        return 0

    ensure_no_unrelated_staged_changes(staged_paths(root), targets)
    print('Archived-result directories:')
    for target in targets:
        print(f'  {target}')
    if dry_run:
        print('Dry run; no files were staged or committed.')
        print_status(root, targets)
        return 0

    # Some archived result formats, notably NetCDF files, are ignored by the
    # repository's general-purpose generated-file rules.  Force-add is safe
    # here because the pathspec is restricted to the two archive directories.
    add_result = run_git(root, ['add', '--all', '--force', '--', *targets])
    if add_result.returncode != 0:
        detail = add_result.stderr.strip() or add_result.stdout.strip()
        raise RuntimeError(f'git add failed: {detail}')

    staged = staged_paths(root)
    ensure_no_unrelated_staged_changes(staged, targets)
    target_changes = {path for path in staged if is_under_target(path, targets)}
    if not target_changes:
        print('No changes found in archived-result directories; nothing to commit.')
        return 0

    commit_result = run_git(
        root,
        ['commit', '-m', message, '--', *targets],
    )
    output = commit_result.stdout.strip() or commit_result.stderr.strip()
    if commit_result.returncode != 0:
        raise RuntimeError(f'git commit failed: {output}')
    if output:
        print(output)
    print(f'Committed {len(target_changes)} archived-result file(s).')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and commit archived results."""
    parser = argparse.ArgumentParser(
        description=(
            'Commit files in docs/source/examples/**/saved_results and '
            'saved_html.'
        )
    )
    parser.add_argument(
        '--message',
        default=DEFAULT_COMMIT_MESSAGE,
        help=f'commit message (default: {DEFAULT_COMMIT_MESSAGE!r})',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='show the target directories and their status without committing',
    )
    args = parser.parse_args(argv)
    return commit_results(
        repository_root=PROJECT_ROOT,
        message=args.message,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2) from error
