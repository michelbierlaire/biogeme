"""Script to transform Python files into Jupyter notebooks
Requires ipynb-py-convert to be installed: pip install ipynb-py-convert
"""
import sys
import os
import subprocess

ignoreDirectory = ['workingNotToDistribute', 'notebooks']

TMP_FILE = '_tmp.py'
with os.scandir('.') as root_dir:
    for path in root_dir:
        if path.is_dir(follow_symlinks=False):
            with os.scandir(path.path) as local:
                if path.path not in ignoreDirectory:
                    print(f'----- {path.path} -----')
                    for file in local:
                        if file.is_file() and file.name.endswith('py'):
                            notebook_fname = (
                                f'{path.path}/{file.name.split(".")[0]}.ipynb'
                            )
                            print(f'Convert {file.name} into {notebook_fname}')
                            with open(
                                    f'{path.path}/{file.name}', 'r', encoding='utf-8'
                            ) as f:
                                read_file = f.read()
                            with open(TMP_FILE, 'w', encoding='utf-8') as f:
                                f.write(f'#%%\n{read_file}')
                            subprocess.call(
                                [
                                    (f'{sys.prefix}/bin/ipynb-py-convert'),
                                    TMP_FILE,
                                    notebook_fname,
                                ]
                            )
