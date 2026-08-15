#!/usr/bin/env python3
"""Run the nine release-comparison estimators in fresh processes.

The runner deliberately launches one script at a time. This avoids contention
between JAX/BLAS processes and gives the nine measurements the same allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
GENERATED_ROOT = ROOT / 'generated'
RELEASES = ('3.2.14', '3.3.3', '3.3.4')
MODELS = ('b05a_normal_mixture', 'b11a_cnl', 'b12_panel')


def release_directory_name(release: str) -> str:
    return f'biogeme_{release.replace(".", "_")}'


def script_path(release: str, model: str) -> Path:
    path = GENERATED_ROOT / release_directory_name(release) / f'{model}.py'
    if not path.is_file():
        raise FileNotFoundError(f'Generated estimator not found: {path}')
    return path


def default_python(env_root: Path, release: str) -> Path:
    return env_root / release_directory_name(release) / 'bin' / 'python'


def parse_release_python(value: str | None, *, release: str, env_root: Path) -> Path:
    path = Path(value) if value else default_python(env_root, release)
    if not path.is_file():
        option = f'--python-{release.replace(".", "-")}'
        raise FileNotFoundError(
            f'Python executable for Biogeme {release} does not exist: {path}. '
            f'Pass {option} or prepare the benchmark environments first.'
        )
    # Do not resolve a venv's Python symlink: resolving it can remove the
    # environment directory from the executable path and make Python ignore
    # the venv's ``pyvenv.cfg``.
    return path.absolute()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def case_name(release: str, model: str) -> str:
    return f'{release}/{model}'


def selected_cases(values: list[str] | None) -> list[tuple[str, str]]:
    cases = [(release, model) for release in RELEASES for model in MODELS]
    if not values:
        return cases
    selected: list[tuple[str, str]] = []
    valid = {case_name(*case): case for case in cases}
    for value in values:
        if value not in valid:
            choices = ', '.join(sorted(valid))
            raise ValueError(f'Unknown case {value!r}. Valid cases: {choices}')
        selected.append(valid[value])
    return selected


def run(args: argparse.Namespace) -> int:
    env_root = args.env_root.resolve()
    if args.dry_run:
        pythons = {
            release: Path(
                getattr(args, f'python_{release.replace(".", "_")}')
                or default_python(env_root, release)
            )
            for release in RELEASES
        }
    else:
        pythons = {
            release: parse_release_python(
                getattr(args, f'python_{release.replace(".", "_")}'),
                release=release,
                env_root=env_root,
            )
            for release in RELEASES
        }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args.only)
    failures: list[str] = []

    for repeat in range(1, args.repetitions + 1):
        for release, model in cases:
            case = case_name(release, model)
            case_dir = output_root / release / model
            result_path = case_dir / f'repeat-{repeat:02d}.json'
            stdout_path = case_dir / f'repeat-{repeat:02d}.stdout.log'
            stderr_path = case_dir / f'repeat-{repeat:02d}.stderr.log'
            command = [
                str(pythons[release]),
                str(script_path(release, model)),
                '--output',
                str(result_path),
            ]
            if args.data is not None:
                command.extend(['--data', str(args.data.resolve())])
            print(f'[{"PLAN" if args.dry_run else "RUN"}] {case} repeat {repeat}')
            print('  ' + ' '.join(command))
            if args.dry_run:
                continue

            case_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    'PYTHONNOUSERSITE': '1',
                    'PYTHONHASHSEED': '0',
                    'OMP_NUM_THREADS': str(args.threads),
                    'OPENBLAS_NUM_THREADS': str(args.threads),
                    'MKL_NUM_THREADS': str(args.threads),
                    'VECLIB_MAXIMUM_THREADS': str(args.threads),
                    'NUMEXPR_NUM_THREADS': str(args.threads),
                }
            )
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            if completed.returncode != 0:
                failures.append(case)
                write_json(
                    case_dir / f'repeat-{repeat:02d}.error.json',
                    {
                        'case': case,
                        'repeat': repeat,
                        'returncode': completed.returncode,
                        'stdout': str(stdout_path),
                        'stderr': str(stderr_path),
                    },
                )
                print(f'  FAILED (exit {completed.returncode}); continuing.')
                continue

            if not result_path.is_file():
                failures.append(case)
                print(f'  FAILED: estimator did not create {result_path}')
                continue
            record = json.loads(result_path.read_text())
            reported_version = str(record.get('biogeme_distribution_version', ''))
            if reported_version != release:
                failures.append(case)
                print(
                    f'  FAILED: expected Biogeme {release}, '
                    f'but estimator reported {reported_version or "unknown"}.'
                )
                continue
            print(
                f'  OK: {record.get("wall_time_seconds", "?")} seconds, '
                f'converged={record.get("converged", False)}'
            )

    if args.dry_run:
        print('Dry run only; no estimators were started.')
        return 0
    if failures:
        print('\nFailed cases:')
        for case in failures:
            print(f'  {case}')
        return 1
    print(f'Completed {len(cases) * args.repetitions} benchmark run(s).')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--env-root', type=Path, default=Path('/home/bierlair/venvs/biogeme-benchmark')
    )
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument(
        '--only', action='append', help='case such as 3.3.4/b11a_cnl (repeatable)'
    )
    parser.add_argument('--dry-run', action='store_true')
    for release in RELEASES:
        parser.add_argument(
            f'--python-{release.replace(".", "-")}',
            dest=f'python_{release.replace(".", "_")}',
            type=str,
            help=f'Python executable for Biogeme {release}',
        )
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2) from error
