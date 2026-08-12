#!/usr/bin/env python3
"""Detect and register examples added to the release suite.

The JED runner discovers every ``plot_*.py`` file automatically.  This tool
maintains a small ignored inventory so that a new example is noticed before a
release starts, and proposes the manifest metadata needed for scheduling and
artifact import.  It is a dry run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .release_common import (
        PROJECT_ROOT,
        RELEASE_ROOT,
        json_dump,
        json_load,
        next_steps,
        relative,
        sha256,
    )
except ImportError:  # pragma: no cover - direct script execution
    from release_common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        RELEASE_ROOT,
        json_dump,
        json_load,
        next_steps,
        relative,
        sha256,
    )

sys.path.insert(0, str(PROJECT_ROOT))

from jed_runs.jed_examples import (  # noqa: E402
    CONFIG_PATH,
    EXAMPLES_ROOT,
    discover_jobs,
    load_config,
)

INVENTORY_PATH = RELEASE_ROOT / 'example-inventory.json'


def source_inventory() -> dict[str, str]:
    return {
        path.relative_to(EXAMPLES_ROOT).as_posix(): sha256(path)
        for path in sorted(EXAMPLES_ROOT.rglob('plot_*.py'))
    }


def load_inventory() -> dict[str, str] | None:
    if not INVENTORY_PATH.is_file():
        return None
    value = json_load(INVENTORY_PATH)
    scripts = value.get('scripts')
    if not isinstance(scripts, dict):
        raise ValueError(f'Invalid example inventory: {INVENTORY_PATH}')
    return {str(name): str(digest) for name, digest in scripts.items()}


def documentation_quality(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        module = ast.parse(path.read_text(errors='replace'))
    except SyntaxError as error:
        return [f'syntax error: {error}']
    docstring = ast.get_docstring(module) or ''
    if not docstring.strip():
        problems.append('missing module documentation')
    if '.. _plot_' not in docstring:
        problems.append('module documentation has no Sphinx example label')
    return problems


def load_manifest() -> dict[str, Any]:
    with CONFIG_PATH.open('rb') as stream:
        return tomllib.load(stream)


def proposal(script: str, job: Any, config: dict[str, Any]) -> str:
    docs_config = config.get('docs', {}).get('examples', {})
    jobs_config = config.get('jobs', {})
    lines = [f'[docs.examples."{script}"]']
    if not job.requires_artifacts:
        lines.extend(['mode = "self_contained"', 'profile = "fast"'])
    else:
        lines.extend(
            [
                'profile = "full"',
                '# TODO: replace these placeholders with the actual output names',
                'expected_outputs = ["TODO.yaml", "TODO.html"]',
            ]
        )
    lines.extend(['', f'[jobs."{script}"]', f'profile = "{job.profile}"'])
    if job.dependencies:
        values = ', '.join(f'"{item}"' for item in job.dependencies)
        lines.append(f'depends_on = [{values}]')
    if job.required_inputs:
        values = ', '.join(f'"{item}"' for item in job.required_inputs)
        lines.append(f'required_inputs = [{values}]')
    # Keep these references explicit so the caller can see why a block was
    # proposed even when an entry is partially configured already.
    if script in docs_config:
        lines.append('# Existing docs entry requires review.')
    if script in jobs_config:
        lines.append('# Existing jobs entry requires review.')
    return '\n'.join(lines)


def inspect() -> dict[str, Any]:
    config = load_config()
    jobs = discover_jobs(config)
    inventory = load_inventory()
    current = source_inventory()
    if inventory is None:
        # The first invocation establishes a baseline.  Treating the whole
        # historical tree as newly added would produce hundreds of false
        # positives in an existing checkout.
        new = []
        changed: list[str] = []
        removed: list[str] = []
    else:
        new = sorted(set(current) - set(inventory))
        changed = sorted(
            name
            for name in set(current) & set(inventory)
            if current[name] != inventory[name]
        )
        removed = sorted(set(inventory) - set(current))

    docs_config = config.get('docs', {}).get('examples', {})
    jobs_config = config.get('jobs', {})
    records: list[dict[str, Any]] = []
    for script in new:
        job = jobs[script]
        path = EXAMPLES_ROOT / script
        records.append(
            {
                'script': script,
                'profile': job.profile,
                'requires_artifacts': job.requires_artifacts,
                'dependencies': list(job.dependencies),
                'required_inputs': list(job.required_inputs),
                'has_docs_entry': script in docs_config,
                'has_job_entry': script in jobs_config,
                'has_output_contract': bool(
                    docs_config.get(script, {}).get('expected_outputs')
                    or docs_config.get(script, {}).get('expected_output_globs')
                ),
                'documentation_problems': documentation_quality(path),
                'proposal': proposal(script, job, config),
            }
        )

    unresolved = [
        record
        for record in records
        if record['documentation_problems']
        or record['requires_artifacts']
        and not record['has_output_contract']
    ]
    return {
        'inventory_exists': inventory is not None,
        'current': current,
        'new': records,
        'changed': changed,
        'removed': removed,
        'unresolved': unresolved,
    }


def print_report(report: dict[str, Any]) -> None:
    if not report['inventory_exists']:
        print('No example inventory exists yet; the current tree is the baseline.')
    new = report['new']
    print(f'New examples: {len(new)}')
    for record in new:
        print(f'  NEW {record["script"]} [{record["profile"]}]')
        if record['requires_artifacts']:
            print('      persistent outputs require an explicit manifest contract')
        for problem in record['documentation_problems']:
            print(f'      documentation: {problem}')
        print('      proposed manifest:')
        for line in record['proposal'].splitlines():
            print(f'        {line}')
    print(f'Changed examples since inventory: {len(report["changed"])}')
    for script in report['changed']:
        print(f'  CHANGED {script}')
    print(f'Removed examples since inventory: {len(report["removed"])}')
    for script in report['removed']:
        print(f'  REMOVED {script}')


def write_inventory(current: dict[str, str]) -> None:
    json_dump(
        INVENTORY_PATH,
        {
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'scripts': current,
        },
    )


def append_manifest(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with CONFIG_PATH.open('a') as stream:
        stream.write('\n\n# Added by release_examples.py\n')
        for record in records:
            stream.write(record['proposal'])
            stream.write('\n\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply',
        action='store_true',
        help='update the inventory and safe manifest entries',
    )
    parser.add_argument(
        '--strict', action='store_true', help='fail if new examples need manual review'
    )
    args = parser.parse_args(argv)
    report = inspect()
    print_report(report)

    if args.strict and (report['unresolved'] or report['changed'] or report['removed']):
        print('\nManual review is required before this release can start.')
        next_steps(
            [
                'Review the proposed manifest blocks above.',
                'For changed examples, update the inventory after reviewing the source fix.',
                'Add actual expected output names for estimator examples.',
                'Run release_examples.py again with --strict.',
            ]
        )
        return 1

    if not args.apply:
        next_steps(
            [
                'Review the report above.',
                'Run release_examples.py --apply to record the current inventory and add safe entries.',
            ]
        )
        return 0

    safe = [record for record in report['new'] if not record['documentation_problems']]
    safe = [record for record in safe if not record['requires_artifacts']]
    if report['unresolved']:
        print('The inventory was not updated because manual review is required.')
        next_steps(
            [
                'Complete the documentation and output contracts for the unresolved examples.',
                'Run release_examples.py --strict again.',
            ]
        )
        return 1
    append_manifest(safe)
    write_inventory(report['current'])
    print(f'Updated example inventory: {relative(INVENTORY_PATH)}')
    next_steps(
        [
            'Run release_examples.py --strict to verify the suite.',
            'Start or resume Phase 1 with release_phase1.py run --apply.',
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
                'Review the error and correct the source or manifest.',
                'Rerun release_examples.py --strict.',
            ]
        )
        raise SystemExit(2) from error
