#!/usr/bin/env python3
"""Manage the Sphinx-gallery examples as dependent JED jobs.

The runner discovers every plot_*.py below docs/source/examples and provides
reset, launch, and status operations. Generated Slurm scripts and job state
live below the ignored .jed_runs directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / 'docs' / 'source' / 'examples'
CONFIG_PATH = Path(__file__).with_name('jed_examples.toml')
ARTIFACT_SUFFIXES = {
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
    '.tex',
    '.yaml',
}
HARVEST_SUFFIXES = {'.html', '.md', '.nc', '.pareto', '.yaml'}
DECLARED_RESULT_SUFFIXES = {'.nc', '.pareto', '.yaml'}
DIAGNOSTIC_MARKDOWN_SUFFIX = '_monte_carlo_diagnostic.md'
RESULT_DIRECTORIES = {'saved_results', 'saved_html'}
INPUT_CSV_NAMES = {'data.csv', 'optima.csv'}
MISSING_OUTPUT_EXIT_CODE = 90
MISSING_INPUT_EXIT_CODE = 91
STATUS_LABELS = {
    'finished without error': 'OK',
    'finished with errors': 'ERROR',
    'running': 'RUNNING',
    'scheduled and pending': 'PENDING',
    'not scheduled': 'NOT_SCHEDULED',
    'not done': 'NOT_DONE',
}
STATUS_ORDER = ('ERROR', 'RUNNING', 'PENDING', 'NOT_DONE', 'NOT_SCHEDULED', 'OK')
RELEASE_STATE_FILENAME = 'release-state.json'


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + chr(10))
    temporary.replace(path)


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open('rb') as config_file:
        return tomllib.load(config_file)


def source_has_estimation(source: str) -> bool:
    methods = (
        '.estimate(',
        '.quick_estimate(',
        '.estimate_or_load(',
        '.estimate_catalog(',
        '.estimate_parameters(',
        '.reestimate(',
        'bayesian_estimation(',
        'bayesian_estimation_panel(',
    )
    return any(method in source for method in methods)


def source_has_output(source: str) -> bool:
    """Return whether an estimator should create a persistent artifact."""
    if 'pareto_file_name' in source or '.to_csv(' in source:
        return True
    if 'generate_html=False' in source and 'generate_yaml=False' in source:
        return False
    return source_has_estimation(source)


def infer_required_inputs(source: str) -> list[str]:
    """Infer literal result inputs; dynamic paths are supplied by the config."""
    if not ('from_yaml_file' in source or 'from_netcdf' in source):
        return []
    paths: list[str] = []
    for match in re.finditer(r"""['"]([^'"]+[.](?:yaml|nc|pareto))['"]""", source):
        value = match.group(1)
        if '{' not in value and value not in paths:
            paths.append(value)
    return paths


@dataclass(frozen=True)
class Job:
    script: str
    path: Path
    source: str
    profile: str
    dependencies: tuple[str, ...]
    required_inputs: tuple[str, ...]
    requires_artifacts: bool
    expected_outputs: tuple[str, ...] = ()
    expected_output_globs: tuple[str, ...] = ()

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def script_name(self) -> str:
        return self.path.name


def default_profile(relative_script: str, source: str) -> str:
    directory = relative_script.split('/', 1)[0]
    if directory == 'bayesian_swissmetro':
        return 'bayesian'
    if directory == 'montecarlo':
        return 'montecarlo'
    if directory == 'hybrid_choice_models' and source_has_estimation(source):
        if relative_script.endswith('plot_h04_mode_lv_gauss_simult.py'):
            return 'hybrid'
        return 'standard'
    if source_has_estimation(source):
        return 'standard'
    return 'light'


def discover_jobs(config: dict[str, Any] | None = None) -> dict[str, Job]:
    config = config or load_config()
    configured_jobs = config.get('jobs', {})
    configured_docs = config.get('docs', {}).get('examples', {})
    jobs: dict[str, Job] = {}
    for path in sorted(EXAMPLES_ROOT.rglob('plot_*.py')):
        relative = path.relative_to(EXAMPLES_ROOT).as_posix()
        source = path.read_text(errors='replace')
        override = configured_jobs.get(relative, {})
        docs_override = configured_docs.get(relative, {})
        dependencies = tuple(override.get('depends_on', []))
        required_inputs = tuple(
            override.get('required_inputs', infer_required_inputs(source))
        )
        jobs[relative] = Job(
            script=relative,
            path=path,
            source=source,
            profile=override.get('profile', default_profile(relative, source)),
            dependencies=dependencies,
            required_inputs=required_inputs,
            requires_artifacts=override.get(
                'requires_artifacts', source_has_output(source)
            ),
            expected_outputs=tuple(docs_override.get('expected_outputs', [])),
            expected_output_globs=tuple(docs_override.get('expected_output_globs', [])),
        )
    for job in jobs.values():
        missing = [
            dependency for dependency in job.dependencies if dependency not in jobs
        ]
        if missing:
            raise ValueError(
                f'{job.script} refers to missing dependency/dependencies: '
                + ', '.join(missing)
            )
    validate_dependency_contracts(config, jobs)
    return jobs


def validate_dependency_contracts(config: dict[str, Any], jobs: dict[str, Job]) -> None:
    """Reject dependency edges whose required artifact is not declared.

    ``[jobs]`` is authoritative for execution order and required inputs, while
    ``[docs.examples]`` declares the producer's output contract.  Checking the
    two views together prevents a consumer from being scheduled after a
    producer that cannot provide the file it needs.
    """
    docs_examples = config.get('docs', {}).get('examples', {})
    for job in jobs.values():
        if not job.dependencies or not job.required_inputs:
            continue
        producer_specs = [docs_examples.get(name, {}) for name in job.dependencies]
        if not all(producer_specs):
            continue
        expected = {
            Path(value).name
            for spec in producer_specs
            for value in spec.get('expected_outputs', [])
        }
        patterns = {
            Path(value).name
            for spec in producer_specs
            for value in spec.get('expected_output_globs', [])
        }
        if not expected and not patterns:
            continue
        missing = [
            Path(value).name
            for value in job.required_inputs
            if Path(value).name not in expected
            and not any(
                fnmatch.fnmatch(Path(value).name, pattern) for pattern in patterns
            )
        ]
        if missing:
            dependencies = ', '.join(job.dependencies)
            available = ', '.join(sorted(expected | patterns)) or '<none>'
            raise ValueError(
                f'{job.script} requires {", ".join(missing)}, but its '
                f'dependency/dependencies ({dependencies}) declare {available}'
            )


def topological_jobs(jobs: dict[str, Job]) -> list[Job]:
    ordered: list[Job] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f'Circular example dependency involving {name}')
        visiting.add(name)
        for dependency in jobs[name].dependencies:
            if dependency in jobs:
                visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(jobs[name])

    for name in sorted(jobs):
        visit(name)
    return ordered


def select_jobs(
    all_jobs: dict[str, Job],
    requested: list[str] | None = None,
    slow_only: bool = False,
    include_dependencies: bool = True,
) -> dict[str, Job]:
    """Select jobs and close the selection over their dependencies.

    ``slow_only`` excludes the ``light`` resource profile.  Explicitly
    requested jobs normally include their prerequisites, even when a
    prerequisite uses the light profile.  A retry can set
    ``include_dependencies`` to false when the required artifacts are already
    archived from an earlier successful run.
    """
    requested_set = set(requested or [])
    if requested_set and slow_only:
        raise ValueError('Use either --only or --slow, not both.')
    unknown = requested_set - all_jobs.keys()
    if unknown:
        raise ValueError('Unknown job(s): ' + ', '.join(sorted(unknown)))
    if requested_set:
        selected = requested_set
    elif slow_only:
        selected = {name for name, job in all_jobs.items() if job.profile != 'light'}
    else:
        selected = set(all_jobs)

    if not include_dependencies:
        return {name: all_jobs[name] for name in selected}

    changed = True
    while changed:
        changed = False
        for name in tuple(selected):
            for dependency in all_jobs[name].dependencies:
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return {name: all_jobs[name] for name in selected}


def state_root(config: dict[str, Any]) -> Path:
    configured = expand(
        os.environ.get(
            'BIOGEME_JED_STATE_DIRECTORY',
            config.get('cluster', {}).get('state_directory', '.jed_runs'),
        )
    )
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def example_artifacts(directory: Path) -> list[Path]:
    """Find generated artifacts without treating bundled input data as output."""
    results: list[Path] = []
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(directory).parts
        if any(part in {'.git', '__pycache__'} for part in relative_parts):
            continue
        if relative_parts and relative_parts[0] in RESULT_DIRECTORIES:
            results.append(path)
            continue
        if path.suffix in ARTIFACT_SUFFIXES:
            if path.suffix == '.csv' and path.name in INPUT_CSV_NAMES:
                continue
            results.append(path)
            continue
        if path.name.endswith(DIAGNOSTIC_MARKDOWN_SUFFIX):
            results.append(path)
            continue
        if path.name.startswith('revenue_') and path.suffix == '.txt':
            results.append(path)
            continue
        if path.name.startswith('slurm-') or path.name.endswith('_slurm.out'):
            results.append(path)
    return sorted(set(results))


def snapshot(directory: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in example_artifacts(directory):
        stat = path.stat()
        result[path.relative_to(directory).as_posix()] = {
            'mtime_ns': stat.st_mtime_ns,
            'size': stat.st_size,
        }
    return result


def changed_artifacts(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> list[str]:
    return sorted(
        name
        for name, metadata in after.items()
        if name not in before or before[name] != metadata
    )


def declared_output_matches(
    job: Job, working_directory: Path, changed: list[str]
) -> list[str]:
    """Return declared outputs that were not created or changed this attempt."""

    changed_set = set(changed)
    missing: list[str] = []

    def candidates(expected: str) -> list[Path]:
        relative = Path(expected)
        if relative.is_absolute() or '..' in relative.parts:
            return []
        if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
            return [working_directory / relative]
        if relative.suffix.lower() == '.html':
            return [
                working_directory / 'saved_html' / relative,
                working_directory / relative,
            ]
        if relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
            return [
                working_directory / 'saved_results' / relative,
                working_directory / relative,
            ]
        return [
            working_directory / relative,
            working_directory / 'saved_results' / relative,
        ]

    def changed_path(path: Path) -> bool:
        try:
            return path.relative_to(working_directory).as_posix() in changed_set
        except ValueError:
            return False

    for expected in job.expected_outputs:
        if not any(
            path.is_file() and changed_path(path) for path in candidates(expected)
        ):
            missing.append(expected)

    for pattern in job.expected_output_globs:
        relative = Path(pattern)
        if relative.is_absolute() or '..' in relative.parts:
            missing.append(pattern)
            continue
        if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
            roots = [working_directory]
        elif relative.suffix.lower() == '.html':
            roots = [working_directory / 'saved_html', working_directory]
        elif relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
            roots = [working_directory / 'saved_results', working_directory]
        else:
            roots = [working_directory, working_directory / 'saved_results']
        matches = [
            path
            for root in roots
            if root.is_dir()
            for path in root.glob(pattern)
            if path.is_file() and changed_path(path)
        ]
        if not matches:
            missing.append(pattern)
    return missing


def declared_output_files(job: Job, directory: Path) -> list[Path]:
    """Return existing files matching a job's declared output contract."""

    result: set[Path] = set()

    def safe_path(value: str) -> Path | None:
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts:
            return None
        return relative

    def exact_candidates(value: str) -> list[Path]:
        relative = safe_path(value)
        if relative is None:
            return []
        if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
            return [directory / relative]
        if relative.suffix.lower() == '.html':
            return [directory / 'saved_html' / relative, directory / relative]
        if relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
            return [directory / 'saved_results' / relative, directory / relative]
        return [directory / relative, directory / 'saved_results' / relative]

    for expected in job.expected_outputs:
        result.update(path for path in exact_candidates(expected) if path.is_file())

    for pattern in job.expected_output_globs:
        relative = safe_path(pattern)
        if relative is None:
            continue
        if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
            roots = [directory]
        elif relative.suffix.lower() == '.html':
            roots = [directory / 'saved_html', directory]
        elif relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
            roots = [directory / 'saved_results', directory]
        else:
            roots = [directory, directory / 'saved_results']
        for root in roots:
            if root.is_dir():
                result.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(result)


def archived_output_candidates(directory: Path, expected: str) -> list[Path]:
    """Return the possible archived locations for one declared output."""
    relative = Path(expected)
    if relative.is_absolute() or '..' in relative.parts:
        return []
    if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
        return [directory / relative]
    if relative.suffix.lower() == '.html':
        return [directory / 'saved_html' / relative, directory / relative]
    if relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
        return [directory / 'saved_results' / relative, directory / relative]
    return [directory / relative, directory / 'saved_results' / relative]


def archived_output_glob_matches(directory: Path, pattern: str) -> list[Path]:
    """Return files matching a declared dynamic-output pattern."""
    relative = Path(pattern)
    if relative.is_absolute() or '..' in relative.parts:
        return []
    if relative.parts and relative.parts[0] in RESULT_DIRECTORIES:
        roots = [directory]
    elif relative.suffix.lower() == '.html':
        roots = [directory / 'saved_html', directory]
    elif relative.suffix.lower() in DECLARED_RESULT_SUFFIXES:
        roots = [directory / 'saved_results', directory]
    else:
        roots = [directory, directory / 'saved_results']
    return [
        path
        for root in roots
        if root.is_dir()
        for path in root.glob(pattern)
        if path.is_file()
    ]


def missing_archived_outputs(
    expected_outputs: tuple[str, ...],
    expected_output_globs: tuple[str, ...],
    directory: Path,
) -> list[str]:
    """Return declared outputs absent from the shared archived example tree."""
    missing = [
        expected
        for expected in expected_outputs
        if not any(
            path.is_file() for path in archived_output_candidates(directory, expected)
        )
    ]
    missing.extend(
        pattern
        for pattern in expected_output_globs
        if not archived_output_glob_matches(directory, pattern)
    )
    return missing


def remove_declared_outputs(job: Job) -> list[Path]:
    """Remove generated outputs so an invalidated example cannot reuse them."""

    removed: list[Path] = []
    for path in declared_output_files(job, job.directory):
        path.unlink()
        removed.append(path)
    return removed


def move_without_overwrite(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = destination
    counter = 0
    while target.exists():
        target = destination.with_name(
            f'{destination.stem}~{counter:02d}{destination.suffix}'
        )
        counter += 1
    shutil.move(str(source), str(target))
    return target


def copy_output(source: Path, destination: Path) -> Path:
    """Copy an output into its archive, replacing the same-named copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return destination


def harvest_outputs(
    directory: Path,
    started_at_ns: int,
    destination_root: Path | None = None,
    *,
    job: Job | None = None,
    before: dict[str, dict[str, int]] | None = None,
    after: dict[str, dict[str, int]] | None = None,
) -> list[str]:
    """Copy new root or archived results to the directories consumed by examples.

    Results are discovered only in the isolated working directory of this job,
    including its ``saved_results`` and ``saved_html`` subdirectories. They
    are archived in the checked-out example directory for dependent jobs,
    without scanning that shared directory for concurrent outputs. NetCDF
    files are archived only when declared by the job's output contract.
    """
    destination_root = destination_root or directory
    changed: set[str] | None = None
    if before is not None:
        after = snapshot(directory) if after is None else after
        changed = set(changed_artifacts(before, after))
    harvested: list[str] = []
    roots = [directory]
    roots.extend(
        directory / result_directory
        for result_directory in sorted(RESULT_DIRECTORIES)
        if (directory / result_directory).is_dir()
    )
    candidate_paths = [
        path
        for root in roots
        for path in sorted(root.iterdir())
        if path.is_file()
        and (
            path.suffix in HARVEST_SUFFIXES
            or (
                path.suffix == '.txt'
                and job is not None
                and any(
                    Path(expected).suffix.lower() == '.txt'
                    and (
                        Path(expected).name == path.name
                        or fnmatch.fnmatch(path.name, Path(expected).name)
                    )
                    for expected in (
                        *job.expected_outputs,
                        *job.expected_output_globs,
                    )
                )
            )
        )
    ]
    for path in candidate_paths:
        if path.suffix == '.nc' and job is not None:
            declared_netcdf = any(
                Path(expected).suffix.lower() == '.nc'
                and (
                    Path(expected).name == path.name
                    or fnmatch.fnmatch(path.name, Path(expected).name)
                )
                for expected in (*job.expected_outputs, *job.expected_output_globs)
            )
            if not declared_netcdf:
                continue
        relative = path.relative_to(directory).as_posix()
        if changed is not None:
            # Comparing snapshots avoids relying on the timestamp resolution of
            # the host filesystem. In particular, Windows can round a newly
            # written file's mtime below ``started_at_ns``.
            if relative not in changed:
                continue
        elif path.stat().st_mtime_ns < started_at_ns:
            # Keep the timestamp behavior for callers that do not provide the
            # lifecycle snapshots.
            continue
        destination_directory = (
            destination_root / 'saved_html'
            if path.suffix == '.html'
            else destination_root
            if path.suffix == '.txt'
            else destination_root / 'saved_results'
        )
        destination = destination_directory / path.name
        if path.resolve() != destination.resolve():
            destination = copy_output(path, destination)
        harvested.append(destination.relative_to(destination_root).as_posix())
    return harvested


def required_input_paths(job: Job, working_directory: Path | None = None) -> list[Path]:
    root = working_directory or job.directory
    return [root / relative for relative in job.required_inputs]


def job_start(
    job: Job, job_directory: Path, working_directory: Path | None = None
) -> int:
    working_directory = working_directory or job.directory
    job_directory.mkdir(parents=True, exist_ok=True)
    missing = [
        str(path.relative_to(working_directory))
        for path in required_input_paths(job, working_directory)
        if not path.is_file()
    ]
    start = {
        'job': job.script,
        'started_at': now_iso(),
        'started_at_ns': time.time_ns(),
        'missing_inputs': missing,
        'before': snapshot(working_directory),
    }
    json_dump(job_directory / 'start.json', start)
    if missing:
        diagnostic = {
            'category': 'finished with errors',
            'reason': 'missing required input files',
            'missing_inputs': missing,
        }
        json_dump(job_directory / 'diagnostic.json', diagnostic)
        print(
            f'{job.script}: missing required input(s): {", ".join(missing)}',
            file=sys.stderr,
        )
        return MISSING_INPUT_EXIT_CODE
    return 0


def job_finish(
    job: Job,
    job_directory: Path,
    exit_code: int,
    working_directory: Path | None = None,
) -> int:
    working_directory = working_directory or job.directory
    start_path = job_directory / 'start.json'
    start = json_load(start_path) if start_path.is_file() else {}
    after = snapshot(working_directory)
    harvested: list[str] = []
    if exit_code == 0 and start:
        harvested = harvest_outputs(
            working_directory,
            int(start.get('started_at_ns', time.time_ns())),
            destination_root=job.directory,
            job=job,
            before=start.get('before', {}),
            after=after,
        )
    changed = changed_artifacts(start.get('before', {}), after) if start else []
    diagnostics: list[str] = []
    if exit_code != 0:
        diagnostics.append(f'Python/Slurm command exited with code {exit_code}.')
    if start.get('missing_inputs'):
        diagnostics.append(
            'Missing required inputs: ' + ', '.join(start['missing_inputs'])
        )
    if exit_code == 0:
        missing_declared = declared_output_matches(job, working_directory, changed)
        if missing_declared:
            diagnostics.append(
                'Missing or unchanged declared output(s): '
                + ', '.join(missing_declared)
            )
            exit_code = MISSING_OUTPUT_EXIT_CODE
        missing_archived = missing_archived_outputs(
            job.expected_outputs,
            job.expected_output_globs,
            job.directory,
        )
        if missing_archived:
            diagnostics.append(
                'Declared output(s) were not archived in the shared example tree: '
                + ', '.join(missing_archived)
            )
            exit_code = MISSING_OUTPUT_EXIT_CODE
        elif job.requires_artifacts and not changed:
            diagnostics.append(
                'No result/report artifact was created or modified; expected an '
                'estimation or post-processing output.'
            )
            exit_code = MISSING_OUTPUT_EXIT_CODE
    completion = {
        'job': job.script,
        'finished_at': now_iso(),
        'exit_code': exit_code,
        'requires_artifacts': job.requires_artifacts,
        'required_inputs': list(job.required_inputs),
        'harvested_outputs': harvested,
        'changed_artifacts': changed,
        'expected_outputs': list(job.expected_outputs),
        'expected_output_globs': list(job.expected_output_globs),
        'diagnostics': diagnostics,
    }
    json_dump(job_directory / 'completion.json', completion)
    json_dump(
        job_directory / 'diagnostic.json',
        {
            'category': ('Done ' if exit_code == 0 else 'ERROR'),
            **completion,
        },
    )
    if diagnostics:
        print(f'{job.script}: ' + ' '.join(diagnostics), file=sys.stderr)
    return exit_code


def profile_settings(config: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = config.get('profiles', {})
    if name not in profiles:
        raise ValueError(f'Unknown JED resource profile: {name}')
    return profiles[name]


def shell_value(value: str) -> str:
    return shlex.quote(value)


def generated_script(config: dict[str, Any], job: Job, run_directory: Path) -> str:
    cluster = config.get('cluster', {})
    settings = profile_settings(config, job.profile)
    repository = expand(
        os.environ.get(
            'BIOGEME_JED_REPOSITORY',
            cluster.get('repository', str(PROJECT_ROOT)),
        )
    )
    python_default = expand(
        os.environ.get('BIOGEME_JED_PYTHON', cluster.get('python', ''))
    )
    gcc_include = cluster.get('gcc_include', '')
    job_slug = job.script.replace('/', '__').removesuffix('.py')
    job_directory = run_directory / 'jobs' / job_slug
    example_directory = str(job.directory)
    dollar = '$'
    lines = [
        '#!/bin/bash -l',
        f'#SBATCH --job-name=biogeme-{job.script.replace("/", "-").removesuffix(".py")[:100]}',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        f'#SBATCH --cpus-per-task={settings["cpus_per_task"]}',
        f'#SBATCH --time={settings["time"]}',
    ]
    for key in ('mem_per_cpu', 'mem', 'partition', 'qos', 'account'):
        if settings.get(key):
            lines.append(f'#SBATCH --{key.replace("_", "-")}={settings[key]}')
    lines.extend(
        [
            f'#SBATCH --chdir={example_directory}',
            f'#SBATCH --output={job_directory}/slurm.out',
            f'#SBATCH --error={job_directory}/slurm.err',
            '',
            'set -euo pipefail',
            f'REPOSITORY={shell_value(repository)}',
            f'EXAMPLE_DIRECTORY={shell_value(example_directory)}',
            f'JOB_DIRECTORY={shell_value(str(job_directory))}',
            f'PYTHON_DEFAULT={shell_value(python_default)}',
            'if [[ -n "' + dollar + '{BIOGEME_JED_PYTHON:-}" ]]; then',
            '    PYTHON_EXECUTABLE="' + dollar + 'BIOGEME_JED_PYTHON"',
            'elif [[ -x "' + dollar + 'PYTHON_DEFAULT" ]]; then',
            '    PYTHON_EXECUTABLE="' + dollar + 'PYTHON_DEFAULT"',
            'elif [[ -x "' + dollar + 'REPOSITORY/.venv/bin/python" ]]; then',
            '    PYTHON_EXECUTABLE="' + dollar + 'REPOSITORY/.venv/bin/python"',
            'else',
            '    PYTHON_EXECUTABLE="' + dollar + 'HOME/venvs/biogeme/bin/python"',
            'fi',
            'if [[ ! -x "' + dollar + 'PYTHON_EXECUTABLE" ]]; then',
            '    echo "No usable Biogeme Python interpreter: '
            + dollar
            + 'PYTHON_EXECUTABLE" >&2',
            '    exit 127',
            'fi',
            'if type module >/dev/null 2>&1; then',
            f'    module load {shell_value(cluster.get("module", "gcc"))}',
            'fi',
            'unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH GCC_EXEC_PREFIX',
            'if command -v gcc >/dev/null 2>&1; then',
            '    export CC="$(command -v gcc)"',
            'fi',
            'if command -v g++ >/dev/null 2>&1; then',
            '    export CXX="$(command -v g++)"',
            'fi',
            'export PYTHONUNBUFFERED=1',
            'export MPLBACKEND=Agg',
            f'export OPENBLAS_NUM_THREADS={settings["blas_threads"]}',
            f'export OMP_NUM_THREADS={settings["blas_threads"]}',
            'export MKL_NUM_THREADS=1',
            'export NUMEXPR_NUM_THREADS=1',
            'JOB_TMP="'
            + dollar
            + '{SLURM_TMPDIR:-'
            + dollar
            + '{TMPDIR:-/tmp}}/biogeme-'
            + dollar
            + '{SLURM_JOB_ID:-manual}"',
            'WORK_DIRECTORY="' + dollar + 'JOB_TMP/work"',
            'mkdir -p "'
            + dollar
            + 'JOB_TMP/pytensor" "'
            + dollar
            + 'JOB_DIRECTORY" "'
            + dollar
            + 'WORK_DIRECTORY"',
            'if ! command -v rsync >/dev/null 2>&1; then',
            '    echo "rsync is required to create an isolated JED work directory" >&2',
            '    exit 127',
            'fi',
            'rsync -a --delete --delete-excluded \\',
            "    --exclude='/*.F12' --exclude='/*.err' --exclude='/*.html' \\",
            "    --exclude='/*.iter' --exclude='/*.log' --exclude='/*.nc' \\",
            "    --exclude='/*.out' --exclude='/*.pareto' --exclude='/*.pickle' \\",
            "    --exclude='/*.pkl' --exclude='/*.tex' --exclude='/*.yaml' \\",
            "    --exclude='/*.run' --exclude='/slurm-*' --exclude='/*_slurm.out' \\",
            "    --exclude='__pycache__/' \\",
            '    "' + dollar + 'EXAMPLE_DIRECTORY/" "' + dollar + 'WORK_DIRECTORY/"',
            f'GCC_INCLUDE={shell_value(gcc_include)}',
            'if [[ -d "' + dollar + 'GCC_INCLUDE" ]]; then',
            '    export PYTENSOR_FLAGS="cxx='
            + dollar
            + '{CXX:-g++},base_compiledir='
            + dollar
            + 'JOB_TMP/pytensor,gcc__cxxflags=-I'
            + dollar
            + 'GCC_INCLUDE"',
            'else',
            '    export PYTENSOR_FLAGS="cxx='
            + dollar
            + '{CXX:-g++},base_compiledir='
            + dollar
            + 'JOB_TMP/pytensor"',
            'fi',
        ]
    )
    if int(settings.get('xla_devices', 0)) > 0:
        lines.append(
            'export XLA_FLAGS="'
            + dollar
            + '{XLA_FLAGS:-} '
            + '--xla_force_host_platform_device_count='
            + str(settings['xla_devices'])
            + '"'
        )
    lines.extend(
        [
            'cd "' + dollar + 'EXAMPLE_DIRECTORY"',
            'set +e',
            '"'
            + dollar
            + 'PYTHON_EXECUTABLE" "'
            + dollar
            + 'REPOSITORY/jed_runs/jed_examples.py" job-start '
            '--script '
            + shell_value(job.script)
            + ' --job-directory "'
            + dollar
            + 'JOB_DIRECTORY" --work-directory "'
            + dollar
            + 'WORK_DIRECTORY"',
            'start_status=$?',
            'if [[ "' + dollar + 'start_status" -eq 0 ]]; then',
            '    srun --chdir="'
            + dollar
            + 'WORK_DIRECTORY" --ntasks=1 "'
            + dollar
            + 'PYTHON_EXECUTABLE" -u '
            + shell_value(job.script_name),
            '    run_status=$?',
            'else',
            '    run_status=' + dollar + 'start_status',
            'fi',
            '"'
            + dollar
            + 'PYTHON_EXECUTABLE" "'
            + dollar
            + 'REPOSITORY/jed_runs/jed_examples.py" job-finish '
            '--script '
            + shell_value(job.script)
            + ' --job-directory "'
            + dollar
            + 'JOB_DIRECTORY" '
            '--work-directory "' + dollar + 'WORK_DIRECTORY" '
            '--exit-code "' + dollar + 'run_status"',
            'finish_status=$?',
            'set -e',
            'if [[ "' + dollar + 'run_status" -ne 0 ]]; then',
            '    exit "' + dollar + 'run_status"',
            'fi',
            'exit "' + dollar + 'finish_status"',
            '',
        ]
    )
    return chr(10).join(lines)


def write_generated_script(
    config: dict[str, Any], job: Job, run_directory: Path
) -> Path:
    path = run_directory / 'jobs' / f'{job.script.replace("/", "__")}.run'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_script(config, job, run_directory))
    path.chmod(0o755)
    return path


def submit_job(script: Path, dependencies: list[str]) -> tuple[str | None, str | None]:
    command = ['sbatch', '--parsable']
    if dependencies:
        command.append('--dependency=afterok:' + ':'.join(dependencies))
    command.append(str(script))
    try:
        result = subprocess.run(command, text=True, capture_output=True)
    except FileNotFoundError:
        return None, 'sbatch is not available'
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        return None, message or f'sbatch exited with code {result.returncode}'
    job_id = result.stdout.strip().split(';', 1)[0]
    return job_id, None


def command_reset(args: argparse.Namespace) -> int:
    config = load_config()
    state = state_root(config)
    reset_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = state / 'resets' / reset_id
    candidates: list[Path] = []
    for directory in sorted(EXAMPLES_ROOT.rglob('*')):
        if not directory.is_dir() or directory.name not in RESULT_DIRECTORIES:
            continue
        candidates.extend(path for path in directory.rglob('*') if path.is_file())
    for path in sorted(EXAMPLES_ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(EXAMPLES_ROOT).parts
        if any(part in RESULT_DIRECTORIES for part in relative_parts):
            continue
        if path.name.endswith('.run'):
            continue
        if path.suffix in ARTIFACT_SUFFIXES and not (
            path.suffix == '.csv' and path.name in INPUT_CSV_NAMES
        ):
            candidates.append(path)
        elif path.name.startswith('slurm-') or path.name.endswith('_slurm.out'):
            candidates.append(path)
        elif path.name.startswith('revenue_') and path.suffix == '.txt':
            candidates.append(path)
    unique = sorted(set(candidates))
    print(f'Found {len(unique)} generated artifact(s) to reset.')
    if args.dry_run or not args.apply:
        for path in unique:
            print(f'  {path.relative_to(PROJECT_ROOT)}')
        print('Dry run only. Re-run with --apply to move them into', backup)
        return 0
    for path in unique:
        destination = backup / path.relative_to(EXAMPLES_ROOT)
        move_without_overwrite(path, destination)
    # Resetting artifacts starts a new release attempt.  Clear any previous
    # OK/manual decisions so the global loop cannot mistake old attempts for
    # fixtures that still exist in the checkout.
    release_state = load_release_state(state)
    timestamp = now_iso()
    for script in discover_jobs(config):
        release_state['invalidated'][script] = {
            'at': timestamp,
            'reason': 'release reset; artifacts must be regenerated',
        }
        release_state.get('manual_ok', {}).pop(script, None)
    save_release_state(state, release_state)
    print(f'Moved {len(unique)} artifact(s) to {backup}')
    return 0


def command_launch(args: argparse.Namespace) -> int:
    config = load_config()
    all_jobs = discover_jobs(config)
    if args.not_done:
        if args.only or args.slow or args.no_dependencies:
            raise ValueError(
                '--not-done cannot be combined with --only, --slow, or --no-dependencies.'
            )
        statuses = global_statuses(config)
        jobs = select_not_done_jobs(all_jobs, statuses)
        if not jobs:
            unfinished = sum(item['label'] == 'NOT_DONE' for item in statuses.values())
            if unfinished:
                print(
                    'No NOT_DONE jobs are runnable yet; check dependencies '
                    'and the ERROR/RUNNING entries in status.'
                )
            else:
                print('All jobs are already OK or currently running.')
            return 0
        print(f'Preparing {len(jobs)} NOT_DONE job(s).')
    else:
        jobs = None
    if args.no_dependencies and not args.only:
        raise ValueError('--no-dependencies requires --only.')
    if jobs is None:
        jobs = select_jobs(
            all_jobs,
            args.only,
            args.slow,
            include_dependencies=not args.no_dependencies,
        )
    ordered = topological_jobs(jobs)
    run_id = args.run_id or datetime.now().strftime('%Y%m%d-%H%M%S')
    state_directory = state_root(config)
    run_directory = state_directory / run_id
    if not args.run_id:
        base_run_id = run_id
        suffix = 1
        while run_directory.exists():
            run_id = f'{base_run_id}-{suffix}'
            run_directory = state_directory / run_id
            suffix += 1
    if run_directory.exists() and not args.force:
        raise ValueError(f'Run directory already exists: {run_directory}')
    run_directory.mkdir(parents=True, exist_ok=True)
    run_record: dict[str, Any] = {
        'run_id': run_id,
        'created_at': now_iso(),
        'repository': expand(
            os.environ.get(
                'BIOGEME_JED_REPOSITORY',
                config.get('cluster', {}).get('repository', str(PROJECT_ROOT)),
            )
        ),
        'jobs': {},
    }
    job_ids: dict[str, str] = {}
    selected_names = set(jobs)
    for job in ordered:
        selected_dependencies = [
            name for name in job.dependencies if name in selected_names
        ]
        dependencies = [
            job_ids[name] for name in selected_dependencies if name in job_ids
        ]
        script = write_generated_script(config, job, run_directory)
        record: dict[str, Any] = {
            'script': job.script,
            'profile': job.profile,
            'dependencies': list(job.dependencies),
            'scheduled_dependencies': selected_dependencies,
            'required_inputs': list(job.required_inputs),
            'requires_artifacts': job.requires_artifacts,
            'expected_outputs': list(job.expected_outputs),
            'expected_output_globs': list(job.expected_output_globs),
            'run_script': str(script),
        }
        if args.dry_run:
            record.update({'status': 'not scheduled', 'diagnostic': 'dry run'})
            print(f'DRY RUN {job.script} <- {", ".join(selected_dependencies) or "-"}')
        elif len(dependencies) != len(selected_dependencies):
            record.update(
                {
                    'status': 'not scheduled',
                    'diagnostic': 'dependency was not scheduled successfully',
                }
            )
            print(f'SKIP {job.script}: dependency was not scheduled', file=sys.stderr)
        else:
            job_id, error = submit_job(script, dependencies)
            if job_id is None:
                record.update({'status': 'not scheduled', 'diagnostic': error})
                print(f'FAIL {job.script}: {error}', file=sys.stderr)
            else:
                job_ids[job.script] = job_id
                record.update(
                    {
                        'status': 'scheduled',
                        'job_id': job_id,
                        'submitted_at': now_iso(),
                    }
                )
                print(f'{job.script}: submitted {job_id}')
        run_record['jobs'][job.script] = record
        json_dump(run_directory / 'run.json', run_record)
    print(f'Run state: {run_directory}')
    return 0


def run_external(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True)
    except FileNotFoundError:
        return 127, '', f'{command[0]} is not available'
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def slurm_state(job_id: str) -> tuple[str | None, str]:
    code, stdout, stderr = run_external(
        ['squeue', '--noheader', '--jobs', job_id, '--format=%T|%R']
    )
    if code == 0 and stdout:
        state, _, reason = stdout.partition('|')
        return state, reason
    code, stdout, stderr = run_external(
        [
            'sacct',
            '--noheader',
            '--parsable2',
            '--allocations',
            '--jobs',
            job_id,
            '--format=State,ExitCode,Elapsed,MaxRSS,Reason',
        ]
    )
    if code == 0 and stdout:
        row = stdout.splitlines()[0].split('|')
        return (row[0] if row else None), '|'.join(row[1:])
    return None, stderr or 'No Slurm accounting record is available yet.'


def classify_job(record: dict[str, Any], run_directory: Path) -> tuple[str, str]:
    if not record.get('job_id'):
        return 'not scheduled', str(record.get('diagnostic', 'no job id'))
    job_directory = (
        run_directory / 'jobs' / record['script'].replace('/', '__').removesuffix('.py')
    )
    completion_path = job_directory / 'completion.json'
    completion = json_load(completion_path) if completion_path.is_file() else None
    state, detail = slurm_state(str(record['job_id']))
    if state:
        normalized = state.upper().split('+', 1)[0]
        if normalized in {'RUNNING', 'COMPLETING', 'CONFIGURING', 'RESIZING'}:
            return 'running', detail
        if normalized in {'PENDING', 'SUSPENDED'}:
            return 'scheduled and pending', detail
        if normalized in {
            'COMPLETED',
            'FAILED',
            'CANCELLED',
            'TIMEOUT',
            'OUT_OF_MEMORY',
            'NODE_FAIL',
            'PREEMPTED',
            'BOOT_FAIL',
            'DEADLINE',
            'REVOKED',
            'SPECIAL_EXIT',
        }:
            if completion:
                sacct_exit_code = detail.split('|', 1)[0] if '|' in detail else None
                slurm_success = sacct_exit_code in (None, '0', '0:0')
                if (
                    completion.get('exit_code') == 0
                    and normalized == 'COMPLETED'
                    and slurm_success
                ):
                    configured_job = discover_jobs().get(record['script'])
                    expected_outputs = (
                        configured_job.expected_outputs
                        if configured_job is not None
                        else tuple(record.get('expected_outputs', ()))
                    )
                    expected_output_globs = (
                        configured_job.expected_output_globs
                        if configured_job is not None
                        else tuple(record.get('expected_output_globs', ()))
                    )
                    missing_archived = missing_archived_outputs(
                        expected_outputs,
                        expected_output_globs,
                        EXAMPLES_ROOT / Path(record['script']).parent,
                    )
                    if missing_archived:
                        return (
                            'not done',
                            'declared archived output(s) missing: '
                            + ', '.join(missing_archived),
                        )
                    return 'finished without error', 'outputs validated'
                diagnostics = completion.get('diagnostics') or [
                    f'Slurm state={state}, detail={detail}'
                ]
                if normalized == 'COMPLETED' and not slurm_success:
                    diagnostics.append(f'Slurm exit code was {sacct_exit_code}.')
                return 'finished with errors', '; '.join(diagnostics)
            if normalized == 'COMPLETED':
                return 'finished with errors', 'missing job completion marker'
            return 'finished with errors', f'Slurm state={state}; {detail}'
    return 'scheduled and pending', detail


def status_label(status: str) -> str:
    """Return the compact, stable status shown by the status command."""
    return STATUS_LABELS.get(status, status.upper().replace(' ', '_'))


def latest_run(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path for path in root.iterdir() if path.is_dir() and path.name != 'resets'
    ]
    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def run_directories(root: Path) -> list[Path]:
    """Return recorded runs from oldest to newest, excluding reset backups."""
    if not root.is_dir():
        return []

    def sort_key(path: Path) -> tuple[str, float]:
        record = json_load(path / 'run.json')
        return (str(record.get('created_at', '')), path.stat().st_mtime)

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name != 'resets' and (path / 'run.json').is_file()
        ),
        key=sort_key,
    )


def release_state_path(root: Path) -> Path:
    return root / RELEASE_STATE_FILENAME


def load_release_state(root: Path) -> dict[str, Any]:
    path = release_state_path(root)
    if not path.is_file():
        return {'invalidated': {}, 'manual_ok': {}}
    value = json_load(path)
    value.setdefault('invalidated', {})
    value.setdefault('manual_ok', {})
    return value


def save_release_state(root: Path, state: dict[str, Any]) -> None:
    json_dump(release_state_path(root), state)


def event_timestamp(value: str | None, fallback: float = 0.0) -> float:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return fallback


def latest_attempts(
    root: Path,
) -> dict[str, tuple[Path, dict[str, Any], float]]:
    """Return the latest non-dry-run attempt for every script across runs."""
    attempts: dict[str, tuple[Path, dict[str, Any], float]] = {}
    for run_directory in run_directories(root):
        run_record = json_load(run_directory / 'run.json')
        run_time = event_timestamp(
            str(run_record.get('created_at', '')), run_directory.stat().st_mtime
        )
        jobs = run_record.get('jobs', {})
        if not isinstance(jobs, dict):
            continue
        for script, value in jobs.items():
            if not isinstance(value, dict):
                continue
            if (
                value.get('status') == 'not scheduled'
                and value.get('diagnostic') == 'dry run'
            ):
                continue
            previous = attempts.get(str(script))
            if previous is None or run_time >= previous[2]:
                record = dict(value)
                record.setdefault('script', str(script))
                attempts[str(script)] = (run_directory, record, run_time)
    return attempts


def global_statuses(
    config: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute one release status per discovered example across all attempts."""
    config = config or load_config()
    root = root or state_root(config)
    state = load_release_state(root)
    attempts = latest_attempts(root)
    statuses: dict[str, dict[str, Any]] = {}
    try:
        discovered_jobs = discover_jobs(config)
    except TypeError:
        # Keep compatibility with small test doubles and third-party wrappers
        # that expose the historical zero-argument discovery call.
        discovered_jobs = discover_jobs()
    for script in sorted(discovered_jobs):
        attempt = attempts.get(script)
        invalidation = state.get('invalidated', {}).get(script, {})
        manual_ok = state.get('manual_ok', {}).get(script, {})
        invalidated_at = event_timestamp(str(invalidation.get('at', '')))
        manual_ok_at = event_timestamp(str(manual_ok.get('at', '')))
        if attempt is None:
            status, detail = 'not done', 'no completed attempt is recorded'
            run_directory = None
            record: dict[str, Any] = {}
            attempt_time = 0.0
        else:
            run_directory, record, attempt_time = attempt
            status, detail = classify_job(record, run_directory)

            # A submitted job without an ID cannot be retried automatically
            # until it is explicitly invalidated, but it is unfinished from a
            # release perspective rather than a separate terminal state.
            if status == 'not scheduled':
                status = 'not done'
                detail = detail or 'job was not scheduled'

        # Timestamps are stored to second precision.  Treat an invalidation
        # or laptop completion at the same second as the latest attempt as the
        # newer release decision, avoiding a stale ERROR/OK classification.
        if invalidated_at and invalidated_at >= max(attempt_time, manual_ok_at):
            status, detail = (
                'not done',
                str(invalidation.get('reason', 'invalidated after source changes')),
            )
        elif manual_ok_at and manual_ok_at >= max(attempt_time, invalidated_at):
            status, detail = (
                'finished without error',
                (f'manually marked OK ({manual_ok.get("source", "external")})'),
            )

        statuses[script] = {
            'script': script,
            'status': status,
            'label': 'RUNNING'
            if status == 'scheduled and pending'
            else status_label(status),
            'detail': detail,
            'run_directory': run_directory,
            'record': record,
        }
    return statuses


def descendants(jobs: dict[str, Job], roots: set[str]) -> set[str]:
    result = set(roots)
    changed = True
    while changed:
        changed = False
        for script, job in jobs.items():
            if script in result:
                continue
            if any(dependency in result for dependency in job.dependencies):
                result.add(script)
                changed = True
    return result


def command_invalidate(args: argparse.Namespace) -> int:
    config = load_config()
    jobs = discover_jobs(config)
    requested = set(jobs) if args.all else set(args.script or [])
    unknown = requested - jobs.keys()
    if unknown:
        raise ValueError('Unknown job(s): ' + ', '.join(sorted(unknown)))
    if not requested:
        raise ValueError('Specify --script or --all.')
    affected = descendants(jobs, requested) if not args.no_dependents else requested
    root = state_root(config)
    state = load_release_state(root)
    timestamp = now_iso()
    for script in sorted(affected):
        remove_declared_outputs(jobs[script])
        state['invalidated'][script] = {
            'at': timestamp,
            'reason': args.reason or 'invalidated after source changes',
        }
        state.get('manual_ok', {}).pop(script, None)
    save_release_state(root, state)
    print(f'Invalidated {len(affected)} job(s).')
    for script in sorted(affected):
        print(f'NOT_DONE {script}')
    return 0


def command_mark_ok(args: argparse.Namespace) -> int:
    config = load_config()
    jobs = discover_jobs(config)
    requested = set(args.script or [])
    unknown = requested - jobs.keys()
    if unknown:
        raise ValueError('Unknown job(s): ' + ', '.join(sorted(unknown)))
    if not args.script:
        raise ValueError('Specify at least one --script.')
    for script in requested:
        directory = jobs[script].directory
        existing_artifacts = [
            path.relative_to(directory).as_posix()
            for path in example_artifacts(directory)
        ]
        if jobs[script].requires_artifacts and not existing_artifacts:
            raise ValueError(
                f'Cannot mark {script} OK; no result/report artifact was found'
            )
        missing = declared_output_matches(jobs[script], directory, existing_artifacts)
        if missing:
            raise ValueError(
                f'Cannot mark {script} OK; missing declared output(s): '
                + ', '.join(missing)
            )
    affected = descendants(jobs, requested)
    root = state_root(config)
    state = load_release_state(root)
    timestamp = now_iso()
    for script in args.script:
        state['manual_ok'][script] = {
            'at': timestamp,
            'source': args.source,
            'note': args.note or '',
        }
        state.get('invalidated', {}).pop(script, None)
    for script in sorted(affected - requested):
        state['invalidated'][script] = {
            'at': timestamp,
            'reason': 'dependency was repaired outside JED',
        }
        state.get('manual_ok', {}).pop(script, None)
    save_release_state(root, state)
    for script in args.script:
        print(f'OK {script} ({args.source})')
    for script in sorted(affected - requested):
        print(f'NOT_DONE {script} (dependent of repaired job)')
    return 0


def select_not_done_jobs(
    jobs: dict[str, Job], statuses: dict[str, dict[str, Any]]
) -> dict[str, Job]:
    """Select unfinished jobs whose dependencies are OK or also unfinished."""
    selected = {
        script for script, item in statuses.items() if item['label'] == 'NOT_DONE'
    }
    # Keep the dependency closure explicit for filtered status mappings.
    changed = True
    while changed:
        changed = False
        for script in tuple(selected):
            for dependency in jobs[script].dependencies:
                if (
                    statuses[dependency]['label'] == 'NOT_DONE'
                    and dependency not in selected
                ):
                    selected.add(dependency)
                    changed = True
    blocked = {
        script
        for script in selected
        if any(
            statuses[dependency]['label'] not in {'OK', 'NOT_DONE'}
            for dependency in jobs[script].dependencies
        )
    }
    # Jobs with a currently running/pending dependency must wait; they remain
    # NOT_DONE and will be picked up by a later release iteration.
    return {script: jobs[script] for script in selected - blocked}


def command_release_status(args: argparse.Namespace) -> int:
    statuses = global_statuses()
    rows: list[tuple[str, str, str, str]] = []
    counts: dict[str, int] = {}
    for script, item in statuses.items():
        label = item['label']
        record = item['record']
        job_id = str(record.get('job_id', '-'))
        run_directory = item['run_directory']
        run_id = run_directory.name if run_directory else '-'
        detail = f'{item["detail"]} [{run_id}]'
        counts[label] = counts.get(label, 0) + 1
        rows.append((label, job_id, script, detail))
    summary = ' | '.join(
        f'{label}={counts[label]}' for label in STATUS_ORDER if counts.get(label)
    )
    print('Release status (all recorded runs)')
    print(f'Summary: {summary or "NO_JOBS"}')
    errors = [row for row in rows if row[0] == 'ERROR']
    if errors:
        print('\nErrors requiring attention:')
        for _, job_id, script, detail in errors:
            print(f'  ERROR {job_id}: {script}')
            print(f'    {detail}')
    print(f'\n{"STATUS":10} {"JOB ID":12} SCRIPT')
    print('-' * 90)
    for label, job_id, script, detail in rows:
        print(f'{label:10} {job_id:12} {script}')
        if args.verbose:
            print(f'  detail: {detail}')
    if getattr(args, 'require_all_ok', False):
        return 0 if all(item['label'] == 'OK' for item in statuses.values()) else 1
    return 0


def command_status(args: argparse.Namespace) -> int:
    if not args.run_id:
        return command_release_status(args)
    config = load_config()
    root = state_root(config)
    run_directory = root / args.run_id if args.run_id else latest_run(root)
    if run_directory is None or not (run_directory / 'run.json').is_file():
        raise ValueError('No run state was found. Launch a run first.')
    run_record = json_load(run_directory / 'run.json')
    rows: list[tuple[str, str, str, str]] = []
    counts: dict[str, int] = {}
    for script, record in run_record.get('jobs', {}).items():
        status, diagnostic = classify_job(record, run_directory)
        label = status_label(status)
        counts[label] = counts.get(label, 0) + 1
        job_id = str(record.get('job_id', '-'))
        rows.append((label, job_id, script, diagnostic))

    print(f'Run {run_record["run_id"]} ({run_record.get("created_at", "unknown")})')
    summary = ' | '.join(
        f'{label}={counts[label]}' for label in STATUS_ORDER if counts.get(label)
    )
    print(f'Summary: {summary or "NO_JOBS"}')
    errors = [row for row in rows if row[0] == 'ERROR']
    if errors:
        print('\nErrors requiring attention:')
        for _, job_id, script, diagnostic in errors:
            print(f'  ERROR {job_id}: {script}')
            if diagnostic:
                print(f'    {diagnostic}')
    print(f'\n{"STATUS":14} {"JOB ID":12} SCRIPT')
    print('-' * 90)
    for label, job_id, script, diagnostic in rows:
        print(f'{label:14} {job_id:12} {script}')
        if args.verbose and diagnostic:
            print(f'  detail: {diagnostic}')
    return 0


def command_job_start(args: argparse.Namespace) -> int:
    jobs = discover_jobs()
    if args.script not in jobs:
        raise ValueError(f'Unknown example job: {args.script}')
    working_directory = Path(args.work_directory) if args.work_directory else None
    return job_start(jobs[args.script], Path(args.job_directory), working_directory)


def command_job_finish(args: argparse.Namespace) -> int:
    jobs = discover_jobs()
    if args.script not in jobs:
        raise ValueError(f'Unknown example job: {args.script}')
    working_directory = Path(args.work_directory) if args.work_directory else None
    return job_finish(
        jobs[args.script], Path(args.job_directory), args.exit_code, working_directory
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    reset = subparsers.add_parser(
        'reset', help='move generated example artifacts aside'
    )
    reset.add_argument('--apply', action='store_true', help='perform the move')
    reset.add_argument('--dry-run', action='store_true', help='only list targets')
    reset.set_defaults(function=command_reset)

    launch = subparsers.add_parser('launch', help='generate and submit JED jobs')
    launch.add_argument(
        '--dry-run', action='store_true', help='generate but do not call sbatch'
    )
    launch.add_argument('--run-id', help='explicit run-state directory name')
    launch.add_argument('--force', action='store_true', help='reuse an existing run id')
    launch.add_argument(
        '--only',
        nargs='*',
        help='submit selected scripts and their dependencies',
    )
    launch.add_argument(
        '--slow',
        action='store_true',
        help='submit every non-light resource profile and its dependencies',
    )
    launch.add_argument(
        '--no-dependencies',
        action='store_true',
        help=(
            'with --only, retry only the named scripts; use this only when '
            'their required artifacts already exist in saved_results/'
        ),
    )
    launch.add_argument(
        '--not-done',
        action='store_true',
        help='submit every job currently marked NOT_DONE in the release state',
    )
    launch.set_defaults(function=command_launch)

    status = subparsers.add_parser('status', help='summarize a JED run')
    status.add_argument(
        '--run-id',
        help='optional historical run id; omit it for the global release status',
    )
    status.add_argument(
        '--verbose', action='store_true', help='print diagnostics for every job'
    )
    status.add_argument(
        '--require-all-ok',
        action='store_true',
        help='return nonzero unless every discovered job is OK',
    )
    status.set_defaults(function=command_status)

    invalidate = subparsers.add_parser(
        'invalidate', help='mark repaired examples and their consumers NOT_DONE'
    )
    invalidate.add_argument('--script', action='append')
    invalidate.add_argument('--all', action='store_true')
    invalidate.add_argument('--no-dependents', action='store_true')
    invalidate.add_argument('--reason')
    invalidate.set_defaults(function=command_invalidate)

    mark_ok = subparsers.add_parser(
        'mark-ok', help='mark a result produced outside JED as OK'
    )
    mark_ok.add_argument('--script', action='append', required=True)
    mark_ok.add_argument('--source', default='laptop')
    mark_ok.add_argument('--note')
    mark_ok.set_defaults(function=command_mark_ok)

    start = subparsers.add_parser('job-start', help=argparse.SUPPRESS)
    start.add_argument('--script', required=True)
    start.add_argument('--job-directory', required=True)
    start.add_argument('--work-directory')
    start.set_defaults(function=command_job_start)

    finish = subparsers.add_parser('job-finish', help=argparse.SUPPRESS)
    finish.add_argument('--script', required=True)
    finish.add_argument('--job-directory', required=True)
    finish.add_argument('--work-directory')
    finish.add_argument('--exit-code', type=int, required=True)
    finish.set_defaults(function=command_job_finish)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
