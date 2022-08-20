import glob
import os

file_list = glob.glob('*.py')
file_list.remove('create_run_files.py')

for test in file_list:
    name, ext = os.path.splitext(test)
    run_text = f'''#!/bin/bash -l
#SBATCH --chdir /home/bierlair/swissmetro
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 192000
#SBATCH --time 70:00:00

source ~/env_biogeme/bin/activate
echo STARTING AT `date`
echo {test}
srun python -u {test}
echo FINISHED AT `date`
'''

    with open(f'{name}.run', 'w') as f:
        print(run_text, file=f)
