"""
Test the filenames module

:author: Michel Bierlaire
:date: Fri Feb 24 11:14:33 2023

"""
import os
import unittest
import biogeme.filenames as fn


class TestVersion(unittest.TestCase):
    def test_get_new_filename(self):
        the_name = fn.get_new_file_name('the_name', 'ext')
        self.assertEqual(the_name, 'the_name.ext')
        with open(the_name, 'w') as f:
            pass
        the_next_name = fn.get_new_file_name('the_name', 'ext')
        self.assertEqual(the_next_name, 'the_name~00.ext')
        os.remove(the_name)
