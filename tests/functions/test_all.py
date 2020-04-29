"""
Run all tests found in the current directory.
"""
# Too constraining
# pylint: disable=invalid-name,

import unittest

if __name__ == "__main__":
    testsuite = unittest.TestLoader().discover('.', pattern="test*.py")
    unittest.TextTestRunner(verbosity=1).run(testsuite)
