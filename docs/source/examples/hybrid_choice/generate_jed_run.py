#!/usr/bin/env python3
"""

Prepare for server
==================

Generate SLURM .run job scripts for batch execution on the JED cluster.

This script scans the current directory for Python files whose names start
with ``plot_`` and automatically generates corresponding ``.run`` SLURM job
scripts. Each generated job script executes the Python file using ``srun``
with a predefined set of SLURM directives suitable for long Bayesian
computations on the JED cluster.

The goal is to avoid manually writing and maintaining multiple SLURM job
files when running a collection of similar Python scripts.

Michel Bierlaire
Thu Dec 25 2025, 08:13:11
"""

import os
from pathlib import Path

# Directory containing the Python scripts (current directory)
BASE_DIR = Path(".")

JED_DIRECTORY = 'bayesian'

# Template for the .run file
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir {workdir}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 36
#SBATCH --time 70:00:00
#SBATCH --export=ALL,XLA_FLAGS="--xla_force_host_platform_device_count=36"
#SBATCH --job-name={job_name}
#SBATCH --output={log_filename}
#SBATCH --error={err_filename}

source ~/venvs/biogeme/bin/activate
echo STARTING AT `date`
echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_NAME=$SLURM_JOB_NAME HOSTNAME=$(hostname)"
srun python -u {script}
echo FINISHED AT `date`
"""


def is_valid_script(path: Path) -> bool:
    """Return True if the file should have a ``.run`` job script generated.

    A file is considered valid if it:

    - is a regular file,
    - has the ``.py`` extension,
    - starts with the prefix ``plot_``, and
    - is not this script itself.
    """
    return (
        path.is_file()
        and path.suffix == ".py"
        and path.name.startswith("plot_")
        and path.name != Path(__file__).name
    )


def main():
    """Generate one SLURM ``.run`` file per eligible Python script.

    The function:

    - scans the current directory for Python files matching the naming
      convention defined in :func:`is_valid_script`,
    - fills the SLURM template with script-specific parameters (job name,
      log files, working directory),
    - writes the resulting ``.run`` files next to the original scripts, and
    - marks them as executable.

    If no matching Python files are found, a short message is printed and
    the function exits without creating any job scripts.
    """
    py_files = [f for f in BASE_DIR.iterdir() if is_valid_script(f)]

    if not py_files:
        print("No Python scripts found.")
        return

    workdir = f'/home/bierlair/{JED_DIRECTORY}'

    for py in py_files:
        job_name = py.stem
        log_filename = py.with_name(py.stem + "_%j.out")
        err_filename = py.with_name(py.stem + "_%j.err")

        run_filename = py.with_suffix(".run")

        content = SLURM_TEMPLATE.format(
            workdir=workdir,
            script=py.name,
            job_name=job_name,
            log_filename=log_filename,
            err_filename=err_filename,
        )

        with open(run_filename, "w") as f:
            f.write(content)

        # Make run file executable
        os.chmod(run_filename, 0o755)

        print(f"Generated: {run_filename}")


if __name__ == "__main__":
    main()
