import os

ignoreDirectory = [
    './workingNotToDistribute',
    './assisted',
    './montecarlo',
]

htmlToCheck = [
    './swissmetro/05normalMixture_allAlgos_CFSQP.html',
    './swissmetro/05normalMixture_allAlgos_LS-BFGS.html',
    './swissmetro/05normalMixture_allAlgos_Line search.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds BFGS fCG.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds BFGS iCG.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds Newton fCG.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds Newton iCG.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds hybrid fCG.html',
    './swissmetro/05normalMixture_allAlgos_Simple bounds hybrid iCG.html',
    './swissmetro/05normalMixture_allAlgos_TR-BFGS.html',
    './swissmetro/05normalMixture_allAlgos_Trust region (cg).html',
    './swissmetro/05normalMixture_allAlgos_Trust region (dogleg).html',
    './swissmetro/05normalMixture_allAlgos_scipy.html',
    './swissmetro/01logit_allAlgos_LS-BFGS.html',
    './swissmetro/01logit_allAlgos_Line search.html',
    './swissmetro/01logit_allAlgos_Simple bounds BFGS.html',
    './swissmetro/01logit_allAlgos_Simple bounds Newton.html',
    './swissmetro/01logit_allAlgos_Simple bounds hybrid.html',
    './swissmetro/01logit_allAlgos_TR-BFGS.html',
    './swissmetro/01logit_allAlgos_Trust region (cg).html',
    './swissmetro/01logit_allAlgos_Trust region (dogleg).html',
    './swissmetro/01logit_allAlgos_scipy.html',
    './swissmetro/09nested_allAlgos_Simple bounds BFGS.html',
    './swissmetro/09nested_allAlgos_Simple bounds Newton.html',
    './swissmetro/09nested_allAlgos_Simple bounds hybrid 20%.html',
    './swissmetro/09nested_allAlgos_Simple bounds hybrid 50%.html',
    './swissmetro/09nested_allAlgos_Simple bounds hybrid 80%.html',
    './swissmetro/09nested_allAlgos_scipy.html',
]
htmlNotNeed = [
    './vns/runknapsack.html',
    './vns/knapsack.html',
    './indicators/03nestedElasticities.html',
    './indicators/05nestedElasticitiesConfidenceIntervals.html',
    './indicators/05nestedElasticitiesCI_Bootstrap.html',
    './indicators/02nestedPlot.html',
    './indicators/04nestedElasticities.html',
    './indicators/05nestedElasticities.html',
    './indicators/02nestedSimulation.html',
    './indicators/06nestedWTP.html',
    './swissmetro/01logit_simul.html',
    './swissmetro/05normalMixture_simul.html',
    './swissmetro/01logit_allAlgos.html',
    './swissmetro/11cnl_simul.html',
    './swissmetro/19individualLevelParameters.html',
    './swissmetro/09nested_allAlgos.html',
    './swissmetro/05normalMixture_allAlgos.html',
    './latent/00factorAnalysis.html',
]

version = '3.2.8'

def checkFile(theFile):
    if theFile.is_file() and theFile.name.endswith('py'):
        current_file = os.path.split(theFile.path)
        dir = current_file[0]
        if dir not in ignoreDirectory:
            root = theFile.name.split(".")[0]
            html = f'{dir}/{root}.html'
            if html not in htmlNotNeed:
                if not os.path.isfile(html):
                    print(f'File {html} is missing')
                else:
                    with open(html) as f:
                        content = f.readlines()
                        for line in content:
                            if ('<p>biogeme' in line and
                                not f'biogeme {version}' in line):
                                print(f'Wrong version for {html}: {line}')

with os.scandir('.') as root_dir:
    for path in root_dir:
        if path.is_dir(follow_symlinks=False):
            with os.scandir(path.path) as local:
                if not path.path in ignoreDirectory:
                    print(f'----- {path.path} -----')
                    for file in local:
                        checkFile(file)
