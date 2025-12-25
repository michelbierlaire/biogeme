"""Generate SLURM `.run` scripts for Biogeme experiments.

This script scans the current directory for Python files whose names start
with ``plot_`` and automatically generates corresponding SLURM ``.run``
submission scripts for each of them.

Each generated ``.run`` file:
- executes the matching Python script,
- uses a predefined SLURM template (CPU-based, OpenBLAS, JAX/XLA enabled), and
- writes both stdout and stderr to a log file named ``<script>_slurm.out``.

The script is intended to simplify batch submission of multiple Biogeme
experiments on the JED cluster while ensuring consistent resource requests
and runtime settings.

Michel Bierlaire
Thu Dec 25 2025, 10:01:15
"""

#!/usr/bin/env python3
import os
from pathlib import Path

# Directory containing the Python scripts (current directory)
BASE_DIR = Path(".")

JED_DIRECTORY = 'swissmetro'

# Template for the .run file
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir {workdir}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 36
#SBATCH --time 70:00:00
#SBATCH --export=ALL,XLA_FLAGS="--xla_force_host_platform_device_count=36"
#SBATCH --output={log_filename}
#SBATCH --error={log_filename}

# Load required modules (cluster-provided toolchain/libs)
module load gcc python openblas
export OPENBLAS_HOME="$OPENBLAS_ROOT"

# (Optional but often useful) ensure OpenBLAS doesn't oversubscribe threads
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"


source ~/venvs/biogeme/bin/activate
echo STARTING AT `date`
srun python -u {script}
echo FINISHED AT `date`
"""


def is_valid_script(path: Path) -> bool:
    """Return True if the file should have a .run job script generated."""
    return (
        path.is_file()
        and path.suffix == ".py"
        and path.name.startswith("plot_")
        and path.name != Path(__file__).name
    )


def main():
    """Generate one SLURM `.run` file per eligible Python script.

    The function scans the base directory for Python scripts whose names start
    with ``plot_`` (excluding this file), generates a corresponding ``.run``
    file using the SLURM template, makes it executable, and reports the
    generated filenames.
    """
    py_files = [f for f in BASE_DIR.iterdir() if is_valid_script(f)]

    if not py_files:
        print("No Python scripts found.")
        return

    workdir = f'/home/bierlair/{JED_DIRECTORY}'

    for py in py_files:
        run_filename = py.with_suffix(".run")
        log_filename = py.with_name(py.stem + "_slurm.out")

        content = SLURM_TEMPLATE.format(
            workdir=workdir,
            script=py.name,
            log_filename=log_filename,
        )

        with open(run_filename, "w") as f:
            f.write(content)

        # Make run file executable
        os.chmod(run_filename, 0o755)

        print(f"Generated: {run_filename}")


if __name__ == "__main__":
    main()
