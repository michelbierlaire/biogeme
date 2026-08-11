# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys
import tomllib
from pathlib import Path

from biogeme.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath('./extensions'))

project = 'Biogeme'
copyright = '2026, Michel Bierlaire'
author = 'Michel Bierlaire'

# The full version, including alpha/Beta/rc tags
release = __version__
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
    'sphinx_gallery.gen_gallery',
    'deprecated_extension',
]

def gallery_pattern() -> str:
    """Return the execution pattern selected by the documentation profile."""
    profile = os.environ.get('BIOGEME_DOCS_GALLERY_PROFILE', 'full').lower()
    if profile == 'none':
        return r'(?!)'
    if profile == 'fast':
        manifest = Path(__file__).resolve().parents[2] / 'jed_runs' / 'jed_examples.toml'
        with manifest.open('rb') as manifest_file:
            configuration = tomllib.load(manifest_file)
        fast_examples = [
            name
            for name, metadata in configuration.get('docs', {}).get('examples', {}).items()
            if metadata.get('profile') == 'fast' and metadata.get('gallery', True)
        ]
        if not fast_examples:
            raise RuntimeError('The fast documentation profile has no examples.')
        alternatives = '|'.join(re.escape(name) for name in sorted(fast_examples))
        return rf'/(?:{alternatives})$'
    if profile != 'full':
        raise RuntimeError(f'Unknown BIOGEME_DOCS_GALLERY_PROFILE: {profile}')
    return r'/plot_'


sphinx_gallery_conf = {
    'examples_dirs': 'examples',  # Path to your example scripts
    'gallery_dirs': 'auto_examples',  # Path to save gallery generated output
    'filename_pattern': gallery_pattern(),
    'ignore_pattern': r'/(?:generate_jed_run)\.py$',
    'remove_config_comments': True,  # Remove config comments from examples
    'abort_on_example_error': True,
    'run_stale_examples': True,
}


# Explicitly disable localization
locale_dirs = []
gettext_compact = False

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'setup.py',
    'flycheck*',
    'verify_scripts.py',
    'generate_jed_run.py',
]

# Set the behavior for type hints. Options are "none", "description", or "signature".
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {}
html_title = f'Biogeme {release} documentation'
html_static_path = ['_static']
