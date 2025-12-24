"""
Test the filenames module

:author: Michel Bierlaire
:date: Fri Feb 24 11:14:33 2023

"""

from pathlib import Path

import biogeme.filenames as fn


def test_get_new_filename(tmp_path):
    # Base name inside the temporary directory
    base = tmp_path / "the_name"

    # First call: file does not exist yet
    the_name = fn.get_new_file_name(str(base), "ext")
    assert the_name == str(base) + ".ext"

    # Create that file
    Path(the_name).touch()

    # Second call: should generate the "~00" version
    the_next_name = fn.get_new_file_name(str(base), "ext")
    assert the_next_name == str(base) + "~00.ext"
