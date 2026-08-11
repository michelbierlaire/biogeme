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
}
STATUS_ORDER = ('ERROR', 'RUNNING', 'PENDING', 'NOT_SCHEDULED', 'OK')


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
    jobs: dict[str, Job] = {}
    for path in sorted(EXAMPLES_ROOT.rglob('plot_*.py')):
        relative = path.relative_to(EXAMPLES_ROOT).as_posix()
        source = path.read_text(errors='replace')
        override = configured_jobs.get(relative, {})
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
) -> dict[str, Job]:
    """Select jobs and close the selection over their dependencies.

    ``slow_only`` excludes the ``light`` resource profile.  Explicitly
    requested jobs always include their prerequisites, even when a prerequisite
    uses the light profile.
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
    directory: Path, started_at_ns: int, destination_root: Path | None = None
) -> list[str]:
    """Copy new root or archived results to the directories consumed by examples.

    Results are discovered only in the isolated working directory of this job,
    including its ``saved_results`` and ``saved_html`` subdirectories. They
    are archived in the checked-out example directory for dependent jobs,
    without scanning that shared directory for concurrent outputs.
    """
    destination_root = destination_root or directory
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
        if path.is_file() and path.suffix in HARVEST_SUFFIXES
    ]
    for path in candidate_paths:
        if path.stat().st_mtime_ns < started_at_ns:
            continue
        destination_directory = (
            destination_root / 'saved_html'
            if path.suffix == '.html'
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
    harvested: list[str] = []
    if exit_code == 0 and start:
        harvested = harvest_outputs(
            working_directory,
            int(start.get('started_at_ns', time.time_ns())),
            destination_root=job.directory,
        )
    after = snapshot(working_directory)
    changed = changed_artifacts(start.get('before', {}), after) if start else []
    diagnostics: list[str] = []
    if exit_code != 0:
        diagnostics.append(f'Python/Slurm command exited with code {exit_code}.')
    if start.get('missing_inputs'):
        diagnostics.append(
            'Missing required inputs: ' + ', '.join(start['missing_inputs'])
        )
    if exit_code == 0 and job.requires_artifacts and not changed:
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
    print(f'Moved {len(unique)} artifact(s) to {backup}')
    return 0


def command_launch(args: argparse.Namespace) -> int:
    config = load_config()
    all_jobs = discover_jobs(config)
    jobs = select_jobs(all_jobs, args.only, args.slow)
    ordered = topological_jobs(jobs)
    run_id = args.run_id or datetime.now().strftime('%Y%m%d-%H%M%S')
    run_directory = state_root(config) / run_id
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
    for job in ordered:
        dependencies = [job_ids[name] for name in job.dependencies if name in job_ids]
        script = write_generated_script(config, job, run_directory)
        record: dict[str, Any] = {
            'script': job.script,
            'profile': job.profile,
            'dependencies': list(job.dependencies),
            'required_inputs': list(job.required_inputs),
            'requires_artifacts': job.requires_artifacts,
            'run_script': str(script),
        }
        if args.dry_run:
            record.update({'status': 'not scheduled', 'diagnostic': 'dry run'})
            print(f'DRY RUN {job.script} <- {", ".join(job.dependencies) or "-"}')
        elif len(dependencies) != len(job.dependencies):
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


def command_status(args: argparse.Namespace) -> int:
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
    launch.set_defaults(function=command_launch)

    status = subparsers.add_parser('status', help='summarize a JED run')
    status.add_argument('--run-id', help='run id; defaults to the newest run')
    status.add_argument(
        '--verbose', action='store_true', help='print diagnostics for every job'
    )
    status.set_defaults(function=command_status)

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
