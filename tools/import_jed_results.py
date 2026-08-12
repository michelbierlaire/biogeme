#!/usr/bin/env python3
"""Import declared documentation artifacts from a completed JED checkout.

The JED machine and the release laptop do not need to share a filesystem.  A
completed checkout (or a staged ``docs/source/examples`` directory) can be
passed to this command and only artifacts declared in
``jed_runs/jed_examples.toml`` are considered.  Exact ``expected_outputs`` and
runtime-generated ``expected_output_globs`` are both supported.  The command
is a dry run by default; ``--apply`` is required before anything is copied.

Examples::

    uv run --locked --group docs python tools/import_jed_results.py \
        --source /tmp/biogeme-jed-examples --profile all
    uv run --locked --group docs python tools/import_jed_results.py \
        --source /tmp/biogeme-jed-examples --profile all --strict --apply

    The source may be a repository checkout or its ``docs/source/examples``
    directory.  Existing target artifacts are backed up below
``.docs_runs/imports`` before they are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_EXAMPLES_ROOT = PROJECT_ROOT / 'docs' / 'source' / 'examples'
RESULT_SUFFIXES = {'.nc', '.pareto', '.yaml'}
ARCHIVE_DIRECTORIES = {'saved_results', 'saved_html'}


@dataclass(frozen=True)
class ImportItem:
    """One expected artifact and the source/target paths selected for it."""

    script: str
    expected: str
    source: Path | None
    target: Path
    candidates: tuple[Path, ...]
    pattern: bool = False


def load_docs_module():
    """Import the shared documentation discovery module from any cwd."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tools import docs_examples

    return docs_examples


def resolve_examples_root(source: Path) -> Path:
    """Return an examples directory from a checkout or staged tree."""
    source = source.expanduser().resolve()
    checkout_examples = source / 'docs' / 'source' / 'examples'
    if checkout_examples.is_dir():
        return checkout_examples
    if source.name == 'examples' and source.is_dir():
        return source
    raise ValueError(
        f'Source must be a repository checkout or docs/source/examples directory: '
        f'{source}'
    )


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected output must be a relative path: {value}')
    return path


def destination_for(expected: str, example_directory: Path) -> Path:
    """Map a manifest output name to its canonical laptop location."""
    relative = _safe_relative(expected)
    if relative.parts and relative.parts[0] in ARCHIVE_DIRECTORIES:
        return example_directory / relative
    if relative.suffix.lower() == '.html':
        return example_directory / 'saved_html' / relative
    if relative.suffix.lower() in RESULT_SUFFIXES:
        return example_directory / 'saved_results' / relative
    # Text reports such as indicators/revenue_1.00.txt are consumed from the
    # example directory itself and therefore remain at its root.
    return example_directory / relative


def source_candidates(expected: str, example_directory: Path) -> tuple[Path, ...]:
    """Return source locations, preferring JED's archived copies."""
    relative = _safe_relative(expected)
    if relative.parts and relative.parts[0] in ARCHIVE_DIRECTORIES:
        candidates = [example_directory / relative]
    elif relative.suffix.lower() == '.html':
        candidates = [
            example_directory / 'saved_html' / relative,
            example_directory / relative,
            example_directory / 'saved_html' / relative,
        ]
    elif relative.suffix.lower() in RESULT_SUFFIXES:
        candidates = [
            example_directory / 'saved_results' / relative,
            example_directory / relative,
            example_directory / 'saved_results' / relative,
        ]
    else:
        candidates = [
            example_directory / relative,
            example_directory / 'saved_results' / relative,
            example_directory / 'saved_html' / relative,
        ]
    # Keep ordering while removing duplicate paths (the archive-first lists
    # intentionally include the direct path as a fallback).
    return tuple(dict.fromkeys(candidates))


def glob_source_candidates(pattern: str, example_directory: Path) -> tuple[Path, ...]:
    """Find files matching a declared dynamic-output pattern.

    JED archives YAML/NetCDF/Pareto output below ``saved_results`` and HTML
    output below ``saved_html``.  The root directory is retained as a fallback
    because the importer may be pointed at a checkout before root cleanup.
    """
    relative = _safe_relative(pattern)
    suffix = relative.suffix.lower()
    if suffix == '.html':
        roots = [example_directory / 'saved_html', example_directory]
    elif suffix in RESULT_SUFFIXES:
        roots = [example_directory / 'saved_results', example_directory]
    else:
        roots = [example_directory, example_directory / 'saved_results']
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(path for path in root.glob(pattern) if path.is_file())
    return tuple(dict.fromkeys(matches))


def expected_from_source(path: Path, example_directory: Path) -> str:
    """Convert an archived source path to its manifest-relative name."""
    relative = path.relative_to(example_directory)
    if relative.parts and relative.parts[0] in ARCHIVE_DIRECTORIES:
        relative = Path(*relative.parts[1:])
    return relative.as_posix()


def build_plan(
    specs: Iterable[Any], source_root: Path, target_root: Path
) -> list[ImportItem]:
    """Build a safe, manifest-limited import plan."""
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    items: list[ImportItem] = []
    seen_targets: dict[Path, str] = {}
    for spec in sorted(specs, key=lambda item: item.script):
        source_directory = source_root / Path(spec.script).parent
        target_directory = target_root / Path(spec.script).parent
        for expected in spec.expected_outputs:
            target = destination_for(expected, target_directory).resolve()
            try:
                target.relative_to(target_root)
            except ValueError as error:
                raise ValueError(
                    f'Import target escapes examples tree: {target}'
                ) from error
            previous = seen_targets.get(target)
            if previous is not None and previous != spec.script:
                raise ValueError(
                    f'Manifest outputs collide at {target}: {previous} and '
                    f'{spec.script}'
                )
            seen_targets[target] = spec.script
            candidates = source_candidates(expected, source_directory)
            source = next((path for path in candidates if path.is_file()), None)
            items.append(
                ImportItem(
                    script=spec.script,
                    expected=expected,
                    source=source.resolve() if source else None,
                    target=target,
                    candidates=tuple(path.resolve() for path in candidates),
                )
            )
        for pattern in getattr(spec, 'expected_output_globs', ()):
            matches = glob_source_candidates(pattern, source_directory)
            if not matches:
                items.append(
                    ImportItem(
                        script=spec.script,
                        expected=pattern,
                        source=None,
                        target=target_directory,
                        candidates=tuple(
                            path.resolve()
                            for path in (
                                source_directory / pattern,
                                source_directory / 'saved_results' / pattern,
                                source_directory / 'saved_html' / pattern,
                            )
                        ),
                        pattern=True,
                    )
                )
                continue
            for source in matches:
                expected = expected_from_source(source, source_directory)
                target = destination_for(expected, target_directory).resolve()
                try:
                    target.relative_to(target_root)
                except ValueError as error:
                    raise ValueError(
                        f'Import target escapes examples tree: {target}'
                    ) from error
                previous = seen_targets.get(target)
                if previous is not None and previous != spec.script:
                    raise ValueError(
                        f'Manifest outputs collide at {target}: {previous} and '
                        f'{spec.script}'
                    )
                if previous == spec.script:
                    # The JED runner intentionally keeps a root-level copy
                    # after archiving it.  A glob can therefore discover the
                    # same logical artifact twice; retain the archive-first
                    # match selected by ``glob_source_candidates``.
                    continue
                seen_targets[target] = spec.script
                items.append(
                    ImportItem(
                        script=spec.script,
                        expected=expected,
                        source=source.resolve(),
                        target=target,
                        candidates=(source.resolve(),),
                        pattern=True,
                    )
                )
    return items


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def new_import_directory(state_root: Path) -> Path:
    base = state_root / 'imports'
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = base / run_id
    counter = 1
    while directory.exists():
        directory = base / f'{run_id}-{counter:02d}'
        counter += 1
    directory.mkdir(parents=True)
    return directory


def apply_plan(
    items: Iterable[ImportItem],
    target_root: Path,
    import_directory: Path,
) -> list[dict[str, Any]]:
    """Copy available artifacts, backing up overwritten targets first."""
    records: list[dict[str, Any]] = []
    backup_root = import_directory / 'backup'
    for item in items:
        record: dict[str, Any] = {
            'script': item.script,
            'expected': item.expected,
            'source': str(item.source) if item.source else None,
            'target': relative_to_project(item.target),
            'status': 'missing',
            'source_candidates': [str(path) for path in item.candidates],
        }
        if item.source is None:
            records.append(record)
            continue
        source_digest = sha256(item.source)
        record['source_sha256'] = source_digest
        if item.target.is_file() and item.target.resolve() == item.source.resolve():
            record.update({'status': 'unchanged', 'target_sha256': source_digest})
            records.append(record)
            continue
        if item.target.is_file() and sha256(item.target) == source_digest:
            record.update({'status': 'unchanged', 'target_sha256': source_digest})
            records.append(record)
            continue
        if item.target.exists():
            try:
                target_relative = item.target.relative_to(target_root.resolve())
            except ValueError as error:
                raise ValueError(
                    f'Refusing to back up unsafe target: {item.target}'
                ) from error
            backup = backup_root / target_relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.target, backup)
            record['backup'] = relative_to_project(backup)
        item.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.target)
        record.update({'status': 'copied', 'target_sha256': sha256(item.target)})
        records.append(record)
    return records


def replace_result_archives(
    items: Iterable[ImportItem], target_root: Path, import_directory: Path
) -> list[Path]:
    """Move archived files not present in the incoming manifest to backup."""

    expected_targets = {item.target.resolve() for item in items}
    backup_root = import_directory / 'backup'
    removed: list[Path] = []
    archive_directories = [
        path
        for path in target_root.rglob('*')
        if path.is_dir() and not path.is_symlink() and path.name in ARCHIVE_DIRECTORIES
    ]
    for directory in sorted(archive_directories):
        for path in sorted(directory.rglob('*')):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.resolve() in expected_targets
            ):
                continue
            relative = path.relative_to(target_root)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup))
            removed.append(path)
    return removed


def print_plan(items: Iterable[ImportItem]) -> tuple[int, int]:
    available = 0
    missing = 0
    for item in items:
        if item.source is None:
            missing += 1
            candidates = ', '.join(str(path) for path in item.candidates)
            print(f'MISSING {item.script}: {item.expected} (looked in {candidates})')
        else:
            available += 1
            print(
                f'COPY    {item.script}: {item.source} -> '
                f'{relative_to_project(item.target)}'
            )
    return available, missing


def command_import(args: argparse.Namespace) -> int:
    docs_examples = load_docs_module()
    config = docs_examples.load_config()
    specs = docs_examples.discover_specs(config)
    selected = docs_examples.select_specs(
        specs,
        None if args.profile == 'all' else args.profile,
        args.script or [],
    )
    selected_with_outputs = [
        spec
        for spec in selected.values()
        if spec.expected_outputs or spec.expected_output_globs
    ]
    source_root = resolve_examples_root(Path(args.source))
    target_root = Path(args.target).expanduser().resolve()
    if not target_root.is_dir():
        raise ValueError(f'Target examples directory does not exist: {target_root}')
    if target_root != TARGET_EXAMPLES_ROOT.resolve():
        # A custom target is useful in tests and staging, but it must still be
        # an examples tree rather than an arbitrary repository directory.
        if target_root.name != 'examples':
            raise ValueError(f'Target must be an examples directory: {target_root}')
    items = build_plan(selected_with_outputs, source_root, target_root)
    if not items:
        raise ValueError('No selected manifest entries declare expected outputs.')

    print(f'Source examples: {source_root}')
    print(f'Target examples: {target_root}')
    print(f'Profile: {args.profile}; selected artifacts: {len(items)}')
    available, missing = print_plan(items)
    if not args.apply:
        print(
            f'Dry run: {available} artifact(s) available, {missing} missing. '
            'Re-run with --apply to copy available artifacts.'
        )
        return 1 if args.strict and missing else 0
    if args.strict and missing:
        print(
            f'error: {missing} declared artifact(s) are missing; '
            'strict apply made no changes.',
            file=sys.stderr,
        )
        return 1

    import_directory = new_import_directory(docs_examples.state_root(config))
    removed: list[Path] = []
    if args.replace_results:
        removed = replace_result_archives(items, target_root, import_directory)
    records = apply_plan(items, target_root, import_directory)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_examples': str(source_root),
        'target_examples': str(target_root),
        'profile': args.profile,
        'strict': args.strict,
        'replace_results': args.replace_results,
        'removed_stale_results': [relative_to_project(path) for path in removed],
        'artifacts': records,
        'counts': {
            'copied': sum(record['status'] == 'copied' for record in records),
            'unchanged': sum(record['status'] == 'unchanged' for record in records),
            'missing': sum(record['status'] == 'missing' for record in records),
        },
    }
    report_path = import_directory / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'Import report: {relative_to_project(report_path)}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        required=True,
        help='JED checkout or staged docs/source/examples directory',
    )
    parser.add_argument(
        '--target',
        default=str(TARGET_EXAMPLES_ROOT),
        help='laptop examples directory (default: repository docs/source/examples)',
    )
    parser.add_argument('--profile', choices=['full', 'fast', 'all'], default='full')
    parser.add_argument(
        '--script', action='append', help='import artifacts for one or more scripts'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='return failure when any declared artifact is missing',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='copy artifacts; without this flag the command is a dry run',
    )
    parser.add_argument(
        '--replace-results',
        action='store_true',
        help=(
            'with --apply, remove archived files not declared by the selected '
            'manifest; removed files are backed up in the import report directory'
        ),
    )
    parser.set_defaults(function=command_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
