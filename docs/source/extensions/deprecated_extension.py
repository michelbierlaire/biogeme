# deprecated_extension.py
from sphinx.application import Sphinx


def process_docstring(app, what, name, obj, options, lines):
    if hasattr(obj, '__deprecated__'):
        lines.insert(
            0,
            f'.. warning:: This function is deprecated. Use :func:`{obj.__newname__}` instead.\n',
        )


def setup(app: Sphinx):
    app.connect('autodoc-process-docstring', process_docstring)
    return {'version': '0.1', 'parallel_read_safe': True}
