"""Run all the examples on scitas """


import os
import subprocess

exclude_files = (
    'checkhtml.py',
    'test_scitas.py',
    'scitas.py',
    '_tmp.py',
    'generateNotebooks.py',
    'create_run_files.py',
)

exclude_dir = (
    'workingNotToDistribute',
    'sampling',
)


class Example:
    def __init__(self, root, file):
        self.root = root
        self.file = file
        self.name, self.ext = os.path.splitext(self.file)
        self.full_name = f'{root[2:]}_{self.name}'
        self.run_name = f'{self.full_name}.run'
        self.log_name = os.path.join(root, f'{self.full_name}.log')

    def create_run_file(self):
        run_text = f'''#!/bin/bash -l
#SBATCH --chdir /home/bierlair/biogeme/examples/{self.root[2:]}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 192000
#SBATCH --time 70:00:00
#SBATCH -o biogeme%j.log

ln -f biogeme${{SLURM_JOB_ID}}.log {self.full_name}.log
source ~/env_biogeme/bin/activate
echo STARTING AT `date`
echo {self.full_name}
srun python -u {self.file}
echo FINISHED AT `date`
rm biogeme${{SLURM_JOB_ID}}.log
'''

        with open(self.run_name, 'w') as f:
            print(run_text, file=f)

    def does_log_exist(self):
        return os.path.isfile(self.log_name)

    def error(self):
        if not self.does_log_exist():
            return False
        with open(self.log_name, 'r') as f:
            return 'error' in f.read()

    def launch_batch(self):
        subprocess.run(['sbatch', self.run_name], check=True)

    def __str__(self):
        return f'{self.root}/{self.name}'


def main():
    # Walk through the structure
    for root, dirs, files in os.walk(top='.', topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dir]
        if root not in exclude_dir:
            for file in files:
                if file.endswith('.py') and file not in exclude_files:
                    the_example = Example(root, file)
                    if the_example.does_log_exist():
                        if the_example.error():
                            print(f'***ERROR***: {the_example}')
                        else:
                            print(f'***FINISHED***: {the_example}')
                    else:
                        print(f'File {the_example.log_name} does not exist. So we run the example.')
                        the_example.create_run_file()
                        the_example.launch_batch()


if __name__ == "__main__":
    main()
