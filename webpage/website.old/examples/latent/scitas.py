files = ['00factorAnalysis.py',
         '01oneLatentRegression.py',
         '02oneLatentOrdered.py',
         '02oneLatentOrderedTrue.py',
         '03choiceOnly.py',
         '04latentChoiceSeq.py',
         '05latentChoiceFull.py',
         '06serialCorrelation.py',
         '07problem.py',
         '07problem_simul.py']

for f in files:
    name = f.rsplit('.', 1)[0]
    print(name)
    content = (f'#!/bin/bash -l\n'
               f'#SBATCH --chdir /home/bierlair/examples/latent\n'
               f'#SBATCH --nodes 1\n'
               f'#SBATCH --ntasks 1\n'
               f'#SBATCH --cpus-per-task 24\n'
               f'#SBATCH --mem 192000\n'
               f'#SBATCH --time 20:00:00\n'
               f'\n'
               f'source ~/env_biogeme/bin/activate\n'
               f'echo STARTING AT `date`\n'
               f'srun python -u {name}.py\n'
               f'echo FINISHED AT `date`\n')
    ff = open(f'{name}.run', 'w')
    ff.write(content)
    ff.close()
