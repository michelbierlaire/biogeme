import unittest

if __name__ == "__main__":
    testsuite = unittest.TestLoader().discover('.',pattern="test_*.py")
    unittest.TextTestRunner(verbosity=1).run(testsuite)
