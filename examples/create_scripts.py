import sys
import os

def script(model):
    """Defines the script to run the ``model`` on the server

    Args:
      model(str): name of the model

    Returns:
      str: text for the script

    """
    text = '#!/bin/bash -l\n'
    text += '#SBATCH --chdir /home/bierlair/examples/swissmetro\n'
    text += '#SBATCH --nodes 1\n'
    text += '#SBATCH --ntasks 1\n'
    text += '#SBATCH --cpus-per-task 24\n'
    text += '#SBATCH --mem 192000\n'
    text += '#SBATCH --time 72:00:00\n'
    text += '\n'
    text += 'source ~/env_biogeme/bin/activate\n'
    text += 'echo STARTING AT `date`\n'
    text += f'echo {model}\n'
    text += f'srun python -u {model}.py\n'
    text += 'echo FINISHED AT `date`\n'
    return text

def python_script(model):
    text = f'echo "{model}.py"\n'
    text += f'python {model}.py >&  {model}.err\n'
    return text
    
directory = sys.argv[1]
print(f'Generate scripts for directory {directory}')

files = [f for f in os.listdir(directory) if f.endswith('py')]

models = [os.path.splitext(f)[0] for f in sorted(files)]
for themodel in models:
    print(themodel)
    with open(f'{directory}/{themodel}.run', 'w') as runfile:
        print(script(themodel), file=runfile)

with open(f'{directory}/sbatch.csh', 'w') as batchfile:
    for themodel in models:
        print(f'sbatch {themodel}.run', file=batchfile)

with open(f'{directory}/run.csh', 'w') as runfile:
    for themodel in models:
        print(f'{python_script(themodel)}', file=runfile)
        
