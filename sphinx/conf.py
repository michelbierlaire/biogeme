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
sys.path.insert(0, os.path.abspath('.hyperlearn'))


# -- Project information -----------------------------------------------------

project = 'Biogeme'
copyright = '2019, Michel Bierlaire'
author = 'Michel Bierlaire'

# The full version, including alpha/beta/rc tags
release = '3.2.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'setup.py']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def skip(app, what, name, obj, would_skip, options):
    keep = ["__init__","__add__","__radd__","__sub__","__rsub__","__mul__","__rmul__","__div__","__rdiv__","__truediv__","__rtruediv__","__neg__","__pow__","__rpow__","__and__","__or__","__eq__","__ne__","__le__","__ge__","__lt__","__gt__"]
    if name in keep :
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
