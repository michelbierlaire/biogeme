import os
import subprocess
import unittest

from icecream import ic


class TestScripts(unittest.TestCase):
    def setUp(self):
        # Directory where the scripts are located
        self.script_dir = [
            './assisted',
            './indicators',
            './latent',
            './montecarlo',
            './programmers',
            './sampling',
            './swissmetro',
        ]

    def get_python_scripts(self):
        # Get all .py files in the specified directories
        return [
            (
                f,
                directory,
            )  # Join each directory with each file and include directory
            for directory in self.script_dir  # Iterate over each directory
            for f in os.listdir(directory)  # List files in the current directory
            if f.endswith('.py')  # Only Python files
        ]

    def test_run_scripts(self):
        # Get all Python scripts and their directories
        python_scripts = self.get_python_scripts()
        ic(python_scripts)
        for script_path, script_dir in python_scripts:
            with self.subTest(script=script_path):
                # Run the script as a subprocess in the correct directory (script_dir)
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    cwd=script_dir,  # Set the working directory to the script's directory
                )

                # Check if the script ran successfully
                self.assertEqual(
                    result.returncode,
                    0,
                    f'Script {script_path} failed with output: {result.stdout}\nError: {result.stderr}',
                )


if __name__ == '__main__':
    unittest.main()
