"""
Test the version module

:author: Michel Bierlaire
:date: Fri Feb 24 07:57:32 2023

"""
import unittest
import biogeme.version as ver


class TestVersion(unittest.TestCase):
    def test_get_html(self):
        html = ver.getHtml()
        self.assertIsInstance(html, str, 'Not a string')
        text = ver.getText()
        self.assertIsInstance(text, str, 'Not a string')
        latex = ver.getLaTeX()
        self.assertIsInstance(latex, str, 'Not a string')
