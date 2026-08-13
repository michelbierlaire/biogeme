#!/usr/bin/env python3
"""Incrementally transfer JED artifacts, clean/import them, and build the docs.

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
        DirtyWorkingTreeError,
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
        DirtyWorkingTreeError,
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
        '--stats',
        '--whole-file',
        '--delete',
        '-e',
        'ssh -o Compression=no',
        '--include=*/',
        '--include=bayesian_swissmetro/saved_results/b01a_logit.nc',
        '--include=bayesian_swissmetro/saved_results/b05_normal_mixture.nc',
        '--exclude=*.nc',
        '--include=*/saved_results/***',
        '--include=*/saved_html/***',
        # Some examples write their declared result files at the example
        # directory root.  The JED harvester accepts both layouts; retain
        # root-level reports in the transfer so the strict importer can see
        # them and move them into the canonical saved_* directories.
        '--include=*.yaml',
        '--include=*.html',
        '--include=*.pareto',
        '--include=revenue_*.txt',
        '--exclude=*',
        source.rstrip('/') + '/',
        str(stage) + '/',
    ]


def staged_artifacts_complete(stage: Path) -> bool:
    """Return whether the stage contains every manifest-declared artifact."""
    if not stage.is_dir():
        return False
    from tools import docs_examples, import_jed_results

    config = docs_examples.load_config()
    specs = docs_examples.discover_specs(config)
    selected = docs_examples.select_specs(specs, None, [])
    with_outputs = [
        spec
        for spec in selected.values()
        if spec.expected_outputs or spec.expected_output_globs
    ]
    if not with_outputs:
        return False
    items = import_jed_results.build_plan(with_outputs, stage, stage)
    return bool(items) and all(item.source is not None for item in items)


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


def clean_docs(*, apply: bool) -> int:
    """Remove generated Sphinx/gallery build state before the first import."""
    return run_command(['make', '-C', 'docs', 'clean'], apply=apply)


def build_docs(*, apply: bool) -> int:
    code = run_command(['make', '-C', 'docs', 'html', 'PROFILE=full'], apply=apply)
    if code:
        return code
    return run_command(['make', '-C', 'docs', 'check-html'], apply=apply)


def phase2_run(args: argparse.Namespace) -> int:
    # Imported fixtures and documentation outputs are generated release
    # artifacts.  Permit those while continuing to reject authored changes.
    ensure_clean_tree(allow_generated=True)
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
    # The release state is deliberately resumable, but the staging directory
    # can be deleted or partially copied independently of that state. Never
    # trust a recorded completed import when its declared inputs are absent.
    if phase.get('imported') and not staged_artifacts_complete(stage):
        print(
            'Recorded Phase 2 import is incomplete: refreshing the JED '
            'artifact transfer.'
        )
        phase['imported'] = False
        phase['built'] = False
    # A transfer can succeed while the strict import still reports missing
    # artifacts (for example, when a JED job finished after the first rsync).
    # Until import has succeeded, always refresh a remote source.  rsync is
    # resumable, so this does not discard an interrupted staging transfer and
    # it ensures that rerunning the same command actually sees newly archived
    # JED results.  Once import is complete, the immutable staged snapshot is
    # reused for the documentation build.
    if not phase.get('imported'):
        source = ensure_source(args.source, stage, apply=args.apply)
        if args.apply:
            phase['transferred'] = True
            phase['artifact_root'] = str(source)
            save_release(release)
    else:
        source = Path(phase.get('artifact_root', str(stage)))
        print(f'Reusing the completed transfer at {relative(source)}.')

    if not phase.get('cleaned') and not phase.get('imported'):
        code = clean_docs(apply=args.apply)
        if code:
            next_steps(
                [
                    'Inspect the documentation cleanup error.',
                    'Rerun release_phase2.py run --apply.',
                ]
            )
            return code
        if args.apply:
            phase['cleaned'] = True
            save_release(release)

    if not phase.get('imported'):
        code = import_artifacts(source, apply=args.apply)
        if code:
            next_steps(
                [
                    'If JED reports all jobs OK, rerun this same command; it refreshes the '
                    'persistent staging transfer before retrying strict import.',
                    'If an artifact is still missing, inspect that example on JED and '
                    'rerun release_phase1.py status there.',
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
    # Keep the individual transfer/import/build commands consistent with the
    # combined command: generated release outputs are allowed, authored edits
    # are still rejected by the clean-tree guard.
    ensure_clean_tree(allow_generated=True)
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
        if phase.get('imported') and not staged_artifacts_complete(stage):
            print(
                'Recorded Phase 2 import is incomplete: refreshing the JED '
                'artifact transfer.'
            )
            phase['imported'] = False
            phase['built'] = False
        if phase.get('imported'):
            source = Path(phase.get('artifact_root', str(stage)))
        else:
            # Refresh a remote stage until strict import succeeds.  A prior
            # transfer may have completed before a late JED result appeared.
            source = ensure_source(args.source, stage, apply=args.apply)
            if args.apply:
                phase['source'] = args.source
                phase['transferred'] = True
                phase['artifact_root'] = str(source)
                save_release(release)
        if phase.get('imported'):
            print(f'Reusing the completed transfer at {relative(source)}.')
        if not phase.get('cleaned'):
            code = clean_docs(apply=args.apply)
            if code:
                next_steps(
                    [
                        'Inspect the documentation cleanup error.',
                        'Rerun release_phase2.py import --apply.',
                    ]
                )
                return code
            if args.apply:
                phase['cleaned'] = True
                save_release(release)
        code = import_artifacts(source, apply=args.apply)
        if code:
            next_steps(
                [
                    'Rerun release_phase2.py import --apply; remote staging is refreshed '
                    'until strict import succeeds.',
                    'If an artifact is still missing, inspect that example on JED and '
                    'rerun release_phase1.py status there.',
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
    except DirtyWorkingTreeError as error:
        print(f'error: {error}', file=sys.stderr)
        next_steps(
            [
                'Inspect git status --short and keep only intentional source changes.',
                'For disposable generated outputs, run release_reset.py --scope all '
                'as a dry run, then rerun it with --apply --confirm.',
                'If archived results must be kept, commit them with '
                'jed_commit_results.py or stash them before cleaning.',
                'Rerun the same release_phase2.py command after the checkout is clean.',
            ]
        )
        return 2
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
