"""Information about the version of Biogeme

Michel Bierlaire
Sat Aug 02 2025, 18:16:18

"""

import datetime

from biogeme.deprecated import deprecated

# Release date
versionDate = f'{datetime.date.today()}'
AUTHOR = 'Michel Bierlaire'
URL_AUTHOR = 'http://people.epfl.ch/michel.bierlaire'
DEPARTMENT = 'Transport and Mobility Laboratory'
URL_DEPARTMENT = 'http://transp-or.epfl.ch'
UNIVERSITY = 'Ecole Polytechnique Fédérale de Lausanne (EPFL)'
URL_UNIVERSITY = 'http://www.epfl.ch'
URL_BIOGEME = 'http://biogeme.epfl.ch'
URL_FORUM = 'https://groups.google.com/d/forum/biogeme'

__version__ = '3.3.2'


def get_version() -> str:
    """
     Version of the software

    :return:  version number, and the release.
    :rtype: string
    """
    return __version__


@deprecated(get_version)
def getVersion() -> str:
    pass


def get_html() -> str:
    """Package information in HTML format

    :return: HTML code.
    :rtype: string

    """
    html = f'<p>biogeme {get_version()} [{versionDate}]</p>\n'
    html += (
        '<p><a href="https://www.python.org/" target="_blank">Python</a> package</p>\n'
    )
    html += (
        f'<p>Home page: <a href="{URL_BIOGEME}" target="_blank">{URL_BIOGEME}</a></p>\n'
    )
    html += (
        f'<p>Submit questions to <a href="{URL_FORUM}" '
        f'target="_blank">{URL_FORUM}</a></p>\n'
    )
    html += f'<p><a href="{URL_AUTHOR}">'
    html += AUTHOR
    html += '</a>, <a href="'
    html += URL_DEPARTMENT
    html += '">'
    html += DEPARTMENT
    html += '</a>, <a href="'
    html += URL_UNIVERSITY
    html += '">'
    html += UNIVERSITY.encode('ascii', 'xmlcharrefreplace').decode()
    html += '</a></p>\n'
    return html


@deprecated(get_html)
def getHtml() -> str:
    pass


def get_text() -> str:
    """Package information in text format

    :return: package information
    :rtype: string
    """

    text = f'biogeme {get_version()} [{versionDate}]\n'
    text += f'Home page: {URL_BIOGEME}\n'
    text += f'Submit questions to {URL_FORUM}\n'
    text += f'{AUTHOR}, {DEPARTMENT}, {UNIVERSITY}\n'
    return text


@deprecated(get_text)
def getText() -> str:
    pass


def get_latex() -> str:
    """Package information in LaTeX format

    :return: LaTeX comments
    :rtype: string
    """
    latex = f'%% biogeme {get_version()} [{versionDate}]\n'
    latex += f'%% Home page: {URL_BIOGEME}\n'
    latex += f'%% Submit questions to {URL_FORUM}\n'
    latex += f'%% {AUTHOR}, {DEPARTMENT}, {UNIVERSITY}\n'
    return latex


@deprecated(get_latex)
def getLaTeX() -> str:
    pass
