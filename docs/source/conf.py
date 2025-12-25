# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

from biogeme.version import __version__
from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath('./extensions'))

project = 'Biogeme'
copyright = '2025, Michel Bierlaire'
author = 'Michel Bierlaire'

# The full version, including alpha/Beta/rc tags
release = __version__
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinx_gallery.load_style",
    'deprecated_extension',
]

sphinx_gallery_conf = {
    'examples_dirs': 'examples',  # Path to your example scripts
    'gallery_dirs': 'auto_examples',  # Path to save gallery generated output
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '/plot_',  # Pattern to match example files
    'remove_config_comments': True,  # Remove config comments from examples
}


# Explicitly disable localization
locale_dirs = []
gettext_compact = False

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "setup.py",
    "flycheck*",
    "verify_scripts.py",
    'generate_jed_run.py',
]

# Set the behavior for type hints. Options are "none", "description", or "signature".
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path = ['_static']
