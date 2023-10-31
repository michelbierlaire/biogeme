# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
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
from sphinx_gallery.sorting import ExampleTitleSortKey

sys.path.insert(0, os.path.abspath('.hyperlearn'))


# -- Project information -----------------------------------------------------

project = 'Biogeme'
copyright = '2023, Michel Bierlaire'
author = 'Michel Bierlaire'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx_gallery.load_style',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
]

sphinx_gallery_conf = {
    'examples_dirs': ['../examples/notebooks'],   # path to your example scripts or notebooks
    'gallery_dirs': ['_auto_examples'],  # path where to save gallery generated examples
    'ignore_pattern': '/run_all\.py',  # Ignore files matching this regex pattern
    'filename_pattern': '.'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'setup.py']

# Set the behavior for type hints. Options are "none", "description", or "signature".
autodoc_typehints = "description"
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'furo'
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_book_theme'
# html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
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
