#!/usr/bin/env python3
"""Prepare detached source worktrees and one uv environment per release.

This command is intentionally dry-run by default. It does not remove an
existing worktree or environment. If a target already exists, it reports it
and leaves it untouched so an interrupted preparation can be resumed safely.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RELEASES = ('3.2.14', '3.3.3', '3.3.4')


def run(command: list[str], *, cwd: Path, apply: bool) -> int:
    print(f'[{"RUN" if apply else "PLAN"}] ' + ' '.join(command))
    if not apply:
        return 0
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree-root', type=Path, required=True)
    parser.add_argument('--environment-root', type=Path, required=True)
    parser.add_argument('--python', default='3.12')
    parser.add_argument('--uv', default='uv')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)

    if args.apply:
        args.worktree_root.mkdir(parents=True, exist_ok=True)
        args.environment_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for release in RELEASES:
        suffix = release.replace('.', '_')
        worktree = args.worktree_root / f'biogeme_{suffix}'
        environment = args.environment_root / f'biogeme_{suffix}'
        if worktree.exists():
            print(f'[KEEP] worktree already exists: {worktree}')
        else:
            code = run(
                ['git', 'worktree', 'add', '--detach', str(worktree), f'v{release}'],
                cwd=PROJECT_ROOT,
                apply=args.apply,
            )
            if code:
                failures.append(f'worktree {release}')
                continue

        if environment.exists():
            print(f'[KEEP] environment already exists: {environment}')
            continue

        code = run(
            [args.uv, 'venv', '--python', args.python, str(environment)],
            cwd=PROJECT_ROOT,
            apply=args.apply,
        )
        if code:
            failures.append(f'venv {release}')
            continue
        python = environment / 'bin' / 'python'
        code = run(
            [args.uv, 'pip', 'install', '--python', str(python), '-e', str(worktree)],
            cwd=PROJECT_ROOT,
            apply=args.apply,
        )
        if code:
            failures.append(f'install {release}')

    if not args.apply:
        print(
            '\nDry run only. Re-run with --apply to create missing worktrees and environments.'
        )
        return 0
    if failures:
        print('Preparation failures:', file=sys.stderr)
        for failure in failures:
            print(f'  {failure}', file=sys.stderr)
        return 1
    print('All missing release environments were prepared.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
