"""
Test executing all examples and verifying that they run properly

:author: Michel Bierlaire
:date: Sat Jul  8 17:22:54 2023

"""

import os
import shutil
import subprocess
import tempfile
import unittest

from biogeme.parameters import biogeme_parameters

ROOT_DIR = (
    '/Users/bierlair/ToBackupOnGoogleDrive/FilesOnGoogleDrive/GitHub/biogeme/examples'
)


exclude = ['replace.py']

examples_dirs = (
    'swissmetro',
    'swissmetro_panel',
    'wtp_space',
    'assisted',
    'indicators',
    'latent',
    'latentbis',
    'montecarlo',
    'sampling',
)

extensions_to_clean = ['html', 'pickle', 'iter', 'log', 'pareto']


def clean_directory(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        for ext in extensions_to_clean:
            if file.endswith(ext):
                file_path = os.path.join(directory, file)
                print(f'Remove {file_path}')
                os.remove(file_path)


class ScriptExecutionTests(unittest.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp()
        biogeme_parameters.set_value(
            name='number_of_draws', value=4, section='MonteCarlo'
        )
        biogeme_parameters.set_value(
            name='largest_neighborhood', value=2, section='AssistedSpecification'
        )
        biogeme_parameters.set_value(
            name='maximum_attempts', value=2, section='AssistedSpecification'
        )
        biogeme_parameters.set_value(
            name='number_of_neighbors', value=2, section='AssistedSpecification'
        )
        biogeme_parameters.set_value(
            name='max_iterations', value=2, section='SimpleBounds'
        )
        biogeme_parameters.set_value(
            name='bootstrap_samples', value=2, section='Estimation'
        )

        biogeme_parameters.set_value(
            name='second_derivatives', value=0, section='SimpleBounds'
        )

        biogeme_parameters.dump_file(file_name=self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_script_execution(self):
        # Iterate over all directories
        for directory in examples_dirs:
            # Clean the directory
            script_dir = f'{ROOT_DIR}/{directory}'
            clean_directory(script_dir)
            toml_file = f'{script_dir}/biogeme.toml'
            orig_file_exists = False
            # Replace the file biogeme.toml, if present, with another file
            if os.path.isfile(toml_file):
                orig_file_exists = True
                backup_file = f'{script_dir}/biogeme.toml.backup'
                shutil.move(toml_file, backup_file)

            shutil.copy2(self.filename, toml_file)

            # Iterate over all .py in the directory
            os.chdir(script_dir)
            for file_name in sorted(os.listdir(script_dir)):
                if file_name in exclude:
                    continue
                file_path = os.path.join(script_dir, file_name)
                # Skip directories and non-Python .py
                if not os.path.isfile(file_path) or not file_name.endswith('.py'):
                    continue

                print(f'Run {file_path}...')

                # Execute the script using subprocess
                try:
                    subprocess.check_output(
                        ['python', file_path], stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError as e:
                    self.fail(
                        f'Script {file_name} failed with error: {e.output.decode()}'
                    )
            if orig_file_exists:
                shutil.move(backup_file, toml_file)
            else:
                os.remove(toml_file)


if __name__ == '__main__':
    unittest.main()
