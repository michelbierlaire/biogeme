files = ['01logit.py',
         '01logitBis.py',
         '01logit_allAlgos.py',
         '01logit_simul.py',
         '02weight.py',
         '03scale.py',
         '04modifVariables.py',
         '05normalMixture.py',
         '05normalMixtureIntegral.py',
         '05normalMixture_allAlgos.py',
         '05normalMixture_simul.py',
         '06unifMixture.py',
         '06unifMixtureIntegral.py',
         '06unifMixtureMHLS.py',
         '07discreteMixture.py',
         '08boxcox.py',
         '09nested.py',
         '09nested_allAlgos.py',
         '10nestedBottom.py',
         '11cnl.py',
         '11cnl_simul.py',
         '11cnl_sparse.py',
         '12panel.py',
         '13panel.py',
         '14nestedEndogenousSampling.py',
         '15panelDiscrete.py',
         '15panelDiscreteBis.py',
         '16panelDiscreteSocioEco.py',
         '17lognormalMixture.py',
         '17lognormalMixtureIntegral.py',
         '18ordinalLogit.py',
         '21probit.py',
         '24haltonMixture.py',
         '25triangularMixture.py',
         '26triangularPanelMixture.py']

print(f'There {len(files)} files')

for f in files:
    name = f.rsplit('.', 1)[0]
    print(name)
    content = (f'#!/bin/bash -l\n'
               f'#SBATCH --chdir /home/bierlair/examples/swissmetro\n'
               f'#SBATCH --nodes 1\n'
               f'#SBATCH --ntasks 1\n'
               f'#SBATCH --cpus-per-task 24\n'
               f'#SBATCH --mem 192000\n'
               f'#SBATCH --time 20:00:00\n'
               f'\n'
               f'source ~/env_biogeme/bin/activate\n'
               f'echo STARTING AT `date`\n'
               f'echo {name}\n'
               f'srun python -u {name}.py\n'
               f'echo FINISHED AT `date`\n')
    ff = open(f'{name}.run', 'w')
    ff.write(content)
    ff.close()
