#!/usr/bin/env python3
"""Reset generated release state for a completely fresh attempt.

This command is intentionally conservative and dry-run by default.  It uses
the allowlisted JED cleaner for generated example artifacts and only removes
explicitly named local release directories.  Source files, input data, the
Git repository, and the release manifest are never targets.
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .release_common import PROJECT_ROOT, next_steps, relative
except ImportError:  # pragma: no cover - direct script execution
    from release_common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        next_steps,
        relative,
    )

sys.path.insert(0, str(PROJECT_ROOT))

from jed_runs.jed_fresh_start import (  # noqa: E402
    EXAMPLES_ROOT,
    collect_targets,
    load_config,
    state_root,
    validate_jed_state_path,
)


def slurm_jobs_running() -> bool:
    try:
        username = getpass.getuser()
    except (ImportError, OSError):
        # ``getpass.getuser`` falls back to the POSIX ``pwd`` module when
        # none of the usual environment variables is set.  That module is
        # unavailable on Windows.  In that situation Slurm cannot be
        # queried reliably, so apply the same conservative laptop behavior
        # as when the ``squeue`` executable is absent.
        print(
            'WARNING: the current username could not be determined; Slurm '
            'jobs cannot be checked from this machine. Continue only if you '
            'have confirmed that no JED jobs are running.',
            file=sys.stderr,
        )
        return False
    try:
        result = subprocess.run(
            ['squeue', '--noheader', '--user', username],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        # A laptop normally has no Slurm client.  The explicit --confirm flag
        # is the user's acknowledgement that a reset is safe in that case.
        print(
            'WARNING: squeue is not available; Slurm jobs cannot be checked '
            'from this machine. Continue only if you have confirmed that no '
            'JED jobs are running.',
            file=sys.stderr,
        )
        return False
    if result.returncode != 0:
        print(
            'WARNING: squeue could not query Slurm; continue only if you have '
            'confirmed that no JED jobs are running.',
            file=sys.stderr,
        )
        return False
    return bool(result.stdout.strip())


def local_targets(scope: str) -> list[Path]:
    targets: list[Path] = []
    if scope in {'jed', 'all'}:
        configured_state = state_root(load_config())
        validate_jed_state_path(configured_state)
        files, directories = collect_targets(EXAMPLES_ROOT, configured_state)
        targets.extend(files)
        targets.extend(directories)
    if scope in {'laptop', 'all'}:
        # The laptop scope also removes imported fixtures and root-level
        # generated artifacts, but deliberately does not remove the JED state
        # directory (that belongs to the JED/all scopes).
        files, directories = collect_targets(
            EXAMPLES_ROOT, PROJECT_ROOT / '.release-reset-no-state'
        )
        targets.extend(files)
        targets.extend(directories)
        for name in ('.docs_runs', '.release_staging'):
            path = PROJECT_ROOT / name
            if path.exists():
                targets.append(path)
        for name in ('build', '.cache', 'warnings.log', 'linkcheck.log', 'doctest.log'):
            path = PROJECT_ROOT / 'docs' / name
            if path.exists():
                targets.append(path)
    for pattern in ('biogeme-smoke-*.err', 'biogeme-smoke-*.out'):
        targets.extend(
            path
            for path in PROJECT_ROOT.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    # Remove nested targets when a parent directory is already selected.
    unique = sorted(set(targets), key=lambda path: (len(path.parts), str(path)))
    result: list[Path] = []
    for path in unique:
        if not any(path != parent and parent in path.parents for parent in result):
            result.append(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['jed', 'laptop', 'all'], default='all')
    parser.add_argument('--apply', action='store_true', help='remove listed targets')
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='required with --apply; confirms that all release artifacts may be removed',
    )
    args = parser.parse_args(argv)
    targets = local_targets(args.scope)
    print(f'Reset scope: {args.scope}')
    print(f'Targets: {len(targets)}')
    for path in targets:
        print(f'  {relative(path)}')

    if not args.apply:
        next_steps(
            [
                'Review the targets above.',
                'Re-run with --apply --confirm to remove them.',
            ]
        )
        return 0
    if not args.confirm:
        raise ValueError('--confirm is required together with --apply')
    if args.scope in {'jed', 'all'} and slurm_jobs_running():
        raise RuntimeError(
            'Slurm jobs are still running; refusing to reset JED release state.'
        )

    for path in targets:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    print(f'Removed {len(targets)} target(s).')
    next_steps(
        [
            'Run release_examples.py --apply to establish a new example inventory.',
            'Start a new release with release_phase1.py run --apply.',
        ]
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        next_steps(
            [
                'Review the reset error and ensure no Slurm jobs are running.',
                'Rerun the reset dry run before applying it.',
            ]
        )
        raise SystemExit(2) from error
