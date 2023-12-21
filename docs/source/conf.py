# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from biogeme.version import __version__
import sphinx_gallery
from sphinx_gallery.sorting import FileNameSortKey


sys.path.insert(0, os.path.abspath(".hyperlearn"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Biogeme'
copyright = '2023, Michel Bierlaire'
author = 'Michel Bierlaire'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinx_gallery.load_style",
]

suppress_warnings = ['autosectionlabel.*']

sphinx_gallery_conf = {
    'within_subsection_order': FileNameSortKey,
}

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "setup.py", "flycheck*"]

# Set the behavior for type hints. Options are "none", "description", or "signature".
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']


def skip(app, what, name, obj, would_skip, options):
    keep = [
        "__init__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__div__",
        "__rdiv__",
        "__truediv__",
        "__rtruediv__",
        "__neg__",
        "__pow__",
        "__rpow__",
        "__and__",
        "__or__",
        "__eq__",
        "__ne__",
        "__le__",
        "__ge__",
        "__lt__",
        "__gt__",
    ]
    if name in keep:
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
