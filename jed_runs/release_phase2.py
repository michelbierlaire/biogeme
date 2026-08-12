#!/usr/bin/env python3
"""Incrementally transfer JED artifacts, import them, and build the docs.

The command is a dry run unless ``--apply`` is supplied.  Transfers use
``rsync --partial`` and imports are manifest-limited and strict, so an
interrupted operation can be retried safely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .release_common import (
        PROJECT_ROOT,
        ensure_clean_tree,
        ensure_release,
        next_steps,
        python_command,
        relative,
        run_command,
        save_release,
    )
except ImportError:  # pragma: no cover - direct script execution
    from release_common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        ensure_clean_tree,
        ensure_release,
        next_steps,
        python_command,
        relative,
        run_command,
        save_release,
    )

sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STAGE = PROJECT_ROOT / '.release_staging' / 'examples'


def is_remote(source: str) -> bool:
    return ':' in source and not source.startswith(('/', './', '../'))


def transfer_command(source: str, stage: Path) -> list[str]:
    return [
        'rsync',
        '-a',
        '--partial',
        '--progress',
        '--whole-file',
        '-e',
        'ssh -o Compression=no',
        '--include=*/',
        '--include=bayesian_swissmetro/saved_results/b01a_logit.nc',
        '--include=bayesian_swissmetro/saved_results/b05_normal_mixture.nc',
        '--exclude=*.nc',
        '--include=*/saved_results/***',
        '--include=*/saved_html/***',
        '--include=revenue_*.txt',
        '--exclude=*',
        source.rstrip('/') + '/',
        str(stage) + '/',
    ]


def ensure_source(source: str, stage: Path, *, apply: bool) -> Path:
    if is_remote(source):
        command = transfer_command(source, stage)
        if apply:
            stage.mkdir(parents=True, exist_ok=True)
        code = run_command(command, apply=apply)
        if code:
            raise RuntimeError(f'Artifact transfer failed with exit code {code}.')
        return stage
    path = Path(source).expanduser().resolve()
    checkout = path / 'docs' / 'source' / 'examples'
    if checkout.is_dir():
        return checkout
    if path.name == 'examples' and path.is_dir():
        return path
    if not path.exists() and not apply:
        print(f'[PLAN] local source is not present yet: {path}')
        return path
    raise ValueError(
        f'Source must be a repository checkout, an examples directory, or a '
        f'remote examples path: {path}'
    )


def import_artifacts(source: Path, *, apply: bool) -> int:
    command = python_command(
        'tools/import_jed_results.py',
        '--source',
        str(source),
        '--profile',
        'all',
        '--strict',
    )
    code = run_command(command, apply=apply)
    if not apply:
        run_command([*command, '--replace-results', '--apply'], apply=False)
        return 0
    if code:
        return code
    apply_command = [*command, '--replace-results', '--apply']
    return run_command(apply_command, apply=True)


def build_docs(*, apply: bool) -> int:
    code = run_command(['make', '-C', 'docs', 'html', 'PROFILE=full'], apply=apply)
    if code:
        return code
    return run_command(['make', '-C', 'docs', 'check-html'], apply=apply)


def phase2_run(args: argparse.Namespace) -> int:
    ensure_clean_tree(args.allow_dirty)
    release = ensure_release(apply=args.apply, phase='phase2')
    phase = release.setdefault('phase2', {})
    stage = Path(args.stage).expanduser().resolve()
    source_key = args.source
    if phase.get('source') and phase['source'] != source_key:
        raise RuntimeError(
            'This release was started with a different artifact source. '
            'Use the same --source or reset the release.'
        )
    phase['source'] = source_key

    source = stage
    if not phase.get('transferred'):
        source = ensure_source(args.source, stage, apply=args.apply)
        if args.apply:
            phase['transferred'] = True
            phase['artifact_root'] = str(source)
            save_release(release)
    else:
        source = Path(phase.get('artifact_root', str(stage)))
        print(f'Reusing the completed transfer at {relative(source)}.')

    if not phase.get('imported'):
        code = import_artifacts(source, apply=args.apply)
        if code:
            next_steps(
                [
                    'Complete or repair the artifact transfer and strict import.',
                    'Rerun release_phase2.py run --apply; completed stages are reused.',
                ]
            )
            return code
        if args.apply:
            phase['imported'] = True
            save_release(release)
    else:
        print('Artifact import was already completed; reusing it.')

    if not phase.get('built'):
        code = build_docs(apply=args.apply)
        if code:
            next_steps(
                [
                    'Inspect docs/warnings.log and correct the documentation issue.',
                    'Rerun release_phase2.py build --apply.',
                ]
            )
            return code
        if args.apply:
            phase['built'] = True
            save_release(release)
    else:
        print('Documentation build was already completed; reusing it.')

    if args.apply:
        next_steps(
            [
                'Review git status, git diff --check, and the generated documentation.',
                'Commit the reviewed release changes manually.',
            ]
        )
    else:
        next_steps(
            [
                'Review the transfer, strict import, and documentation-build plan.',
                'Run release_phase2.py run --apply to execute it incrementally.',
            ]
        )
    return 0


def phase2_step(args: argparse.Namespace) -> int:
    release = ensure_release(apply=args.apply, phase='phase2')
    phase = release.setdefault('phase2', {})
    stage = Path(args.stage).expanduser().resolve()
    if args.command == 'transfer':
        if phase.get('source') and phase['source'] != args.source:
            raise RuntimeError(
                'This release was started with a different artifact source. '
                'Use the same --source or reset the release.'
            )
        source = ensure_source(args.source, stage, apply=args.apply)
        if args.apply:
            phase['source'] = args.source
            phase['transferred'] = True
            phase['artifact_root'] = str(source)
            save_release(release)
        next_steps(['Run release_phase2.py import --apply.'])
        return 0
    if args.command == 'import':
        if phase.get('transferred'):
            source = Path(phase.get('artifact_root', str(stage)))
            print(f'Reusing the completed transfer at {relative(source)}.')
        else:
            source = ensure_source(args.source, stage, apply=False)
        code = import_artifacts(source, apply=args.apply)
        if code:
            next_steps(
                [
                    'Complete or repair the artifact transfer and strict import.',
                    'Rerun release_phase2.py import --apply.',
                ]
            )
            return code
        if args.apply:
            phase['source'] = args.source
            phase['imported'] = True
            save_release(release)
        next_steps(['Run release_phase2.py build --apply.'])
        return 0
    code = build_docs(apply=args.apply)
    if code:
        next_steps(
            [
                'Inspect docs/warnings.log and correct the documentation issue.',
                'Rerun release_phase2.py build --apply.',
            ]
        )
        return code
    if args.apply:
        phase['built'] = True
        save_release(release)
    next_steps(['Review git status and commit the reviewed release changes manually.'])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common(step: argparse.ArgumentParser) -> None:
        step.add_argument('--source', default=str(DEFAULT_STAGE))
        step.add_argument('--stage', default=str(DEFAULT_STAGE))
        step.add_argument('--allow-dirty', action='store_true')

    run = subparsers.add_parser('run', help='transfer, import, and build')
    add_common(run)
    run.add_argument('--apply', action='store_true')
    run.set_defaults(function=phase2_run)
    for name in ('transfer', 'import', 'build'):
        step = subparsers.add_parser(name)
        add_common(step)
        step.add_argument('--apply', action='store_true')
        step.set_defaults(function=phase2_step)
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
                'Resolve the reported transfer, import, or build issue.',
                'Rerun the same release_phase2.py command; completed stages are reused.',
            ]
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
