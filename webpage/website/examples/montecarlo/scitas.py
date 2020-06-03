files = ['01simpleIntegral.py',
         '02simpleIntegral.py',
         '03antithetic.py',
         '03antitheticExplicit.py',
         '04normalMixtureNumerical.py',
         '05normalMixtureMonteCarlo.py',
         '06estimationIntegral.py',
         '07estimationMonteCarlo.py',
         '07estimationMonteCarlo_500.py',
         '07estimationMonteCarlo_anti.py',
         '07estimationMonteCarlo_anti_500.py',
         '07estimationMonteCarlo_halton.py',
         '07estimationMonteCarlo_halton_500.py',
         '07estimationMonteCarlo_mlhs.py',
         '07estimationMonteCarlo_mlhs_500.py',
         '07estimationMonteCarlo_mlhs_anti.py',
         '07estimationMonteCarlo_mlhs_anti_500.py']

for f in files:
    name = f.rsplit('.', 1)[0]
    print(name)
    content = (f'#!/bin/bash -l\n'
               f'#SBATCH --chdir /home/bierlair/montecarlo\n'
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
