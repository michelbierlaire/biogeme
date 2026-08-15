#!/usr/bin/env python3
"""Compare benchmark solutions and produce a timing table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

RELEASES = ('3.2.14', '3.3.3', '3.3.4')
MODELS = ('b05a_normal_mixture', 'b11a_cnl', 'b12_panel')


def load_records(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for release in RELEASES:
        for model in MODELS:
            paths = sorted((root / release / model).glob('repeat-*.json'))
            paths = [path for path in paths if not path.name.endswith('.error.json')]
            values = [json.loads(path.read_text()) for path in paths]
            if values:
                records[(release, model)] = values
    return records


def median_time(values: list[dict[str, Any]]) -> float | None:
    times = [record.get('wall_time_seconds') for record in values]
    times = [float(value) for value in times if isinstance(value, (int, float))]
    return statistics.median(times) if times else None


def numeric_difference(left: Any, right: Any) -> tuple[float | None, float | None]:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None, None
    absolute = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return absolute, absolute / scale


def solution_comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_parameters = reference.get('estimated_parameters', {})
    candidate_parameters = candidate.get('estimated_parameters', {})
    names = sorted(set(reference_parameters) | set(candidate_parameters))
    parameter_differences: dict[str, dict[str, float | None]] = {}
    for name in names:
        absolute, relative = numeric_difference(
            reference_parameters.get(name), candidate_parameters.get(name)
        )
        parameter_differences[name] = {
            'absolute': absolute,
            'relative': relative,
        }
    max_absolute = max(
        (
            item['absolute']
            for item in parameter_differences.values()
            if item['absolute'] is not None
        ),
        default=None,
    )
    max_relative = max(
        (
            item['relative']
            for item in parameter_differences.values()
            if item['relative'] is not None
        ),
        default=None,
    )
    ll_absolute, ll_relative = numeric_difference(
        reference.get('final_log_likelihood'), candidate.get('final_log_likelihood')
    )
    return {
        'reference_release': reference.get('release'),
        'candidate_release': candidate.get('release'),
        'reference_converged': reference.get('converged'),
        'candidate_converged': candidate.get('converged'),
        'log_likelihood_absolute_difference': ll_absolute,
        'log_likelihood_relative_difference': ll_relative,
        'maximum_parameter_absolute_difference': max_absolute,
        'maximum_parameter_relative_difference': max_relative,
        'parameter_differences': parameter_differences,
    }


def format_seconds(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.3f}'


def report_markdown(
    records: dict[tuple[str, str], list[dict[str, Any]]],
    comparisons: list[dict[str, Any]],
    *,
    results_root: Path,
) -> str:
    lines = [
        '# Biogeme release benchmark',
        '',
        'The models are the Biogeme 3.3.4 Swissmetro specifications. '
        'Only the compatibility adapter changes across releases.',
        '',
        f'Results directory: `{results_root}`',
        '',
        '## Median wall-clock estimation time (seconds)',
        '',
        '| Model | 3.2.14 | 3.3.3 | 3.3.4 |',
        '|---|---:|---:|---:|',
    ]
    for model in MODELS:
        values = [
            format_seconds(median_time(records.get((release, model), [])))
            for release in RELEASES
        ]
        lines.append(f'| `{model}` | {values[0]} | {values[1]} | {values[2]} |')

    lines.extend(
        [
            '',
            '## Correctness comparisons',
            '',
            'The reference is Biogeme 3.3.4. Differences are reported rather '
            'than silently treated as failures. The two Monte Carlo models '
            'may differ because equivalent seeds do not guarantee identical '
            'draw streams across implementation generations.',
            '',
            '| Model | Candidate | Δ log-likelihood | Max abs parameter difference | Max relative parameter difference |',
            '|---|---|---:|---:|---:|',
        ]
    )
    for comparison in comparisons:
        lines.append(
            f'| `{comparison["model"]}` | {comparison["candidate_release"]} | '
            f'{format_seconds(comparison["log_likelihood_absolute_difference"])} | '
            f'{format_seconds(comparison["maximum_parameter_absolute_difference"])} | '
            f'{format_seconds(comparison["maximum_parameter_relative_difference"])} |'
        )
    lines.extend(
        [
            '',
            '## Run metadata',
            '',
            'Each JSON record contains the executable path, imported Biogeme '
            'module path, package version, seed, draw count, configuration, '
            'convergence flag, and Biogeme-reported optimization diagnostics.',
            '',
        ]
    )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument(
        '--strict', action='store_true', help='fail if any of the nine runs is missing'
    )
    args = parser.parse_args(argv)

    root = args.results_root.resolve()
    records = load_records(root)
    expected = {(release, model) for release in RELEASES for model in MODELS}
    missing = sorted(expected - set(records))
    if missing:
        print('Missing benchmark results:', file=sys.stderr)
        for release, model in missing:
            print(f'  {release}/{model}', file=sys.stderr)
        if args.strict:
            return 2

    comparisons: list[dict[str, Any]] = []
    for model in MODELS:
        reference_values = records.get(('3.3.4', model), [])
        if not reference_values:
            continue
        reference = reference_values[0]
        for release in RELEASES:
            if release == '3.3.4' or not records.get((release, model)):
                continue
            candidate = records[(release, model)][0]
            comparison = solution_comparison(reference, candidate)
            comparison['model'] = model
            comparisons.append(comparison)

    markdown = report_markdown(records, comparisons, results_root=root)
    print(markdown)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + '\n')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['model', *RELEASES])
            for model in MODELS:
                writer.writerow(
                    [
                        model,
                        *[
                            median_time(records.get((release, model), []))
                            for release in RELEASES
                        ],
                    ]
                )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
