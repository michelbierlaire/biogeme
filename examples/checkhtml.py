import os

ignoreDirectory = [
    './workingNotToDistribute',
    './assisted',
    './montecarlo',
]

htmlToCheck = []
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

version = '3.2.7'

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
