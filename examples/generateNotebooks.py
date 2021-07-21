import sys
import os, shutil
import pathlib
import subprocess

ignoreDirectory = ['workingNotToDistribute']

tmp_file = '_tmp.py'
with os.scandir('.') as root_dir:
    for path in root_dir:
        if path.is_dir(follow_symlinks=False):
            with os.scandir(path.path) as local:
                if not path.path in ignoreDirectory:
                    print(f'----- {path.path} -----')
                    for file in local:
                        if file.is_file() and file.name.endswith('py'):
                            notebook_fname = f'{path.path}/{file.name.split(".")[0]}.ipynb'
                            print(f'Convert {file.name} into {notebook_fname}') 
                            with open(f'{path.path}/{file.name}', 'r') as f:
                                read_file = f.read()
                            with open(tmp_file, 'w') as f:
                                f.write(f'#%%\n{read_file}')
                            subprocess.call([(f'{sys.prefix}/bin/ipynb-py-convert'),
                                             tmp_file,
                                             notebook_fname])
                            
