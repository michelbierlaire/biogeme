#!/usr/bin/env python3
import os
from pathlib import Path

# Directory containing the Python scripts (current directory)
BASE_DIR = Path(".")

JED_DIRECTORY = 'bayesian_swissmetro'

# Template for the .run file
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir {workdir}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 70:00:00
#SBATCH --export=ALL,XLA_FLAGS="--xla_force_host_platform_device_count=4"
#SBATCH --output={log_filename}


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
