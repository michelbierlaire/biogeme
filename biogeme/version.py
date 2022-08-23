"""Information about the version of Biogeme

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

"""

# Too constraining
# pylint: disable=invalid-name,


import datetime

# Release date
versionDate = f'{datetime.date.today()}'
# Major version number
versionMajor = 3
# Minor version number
versionMinor = 2
# Release name
versionRelease = '10'
author = 'Michel Bierlaire'
urlAuthor = 'http://people.epfl.ch/michel.bierlaire'
department = 'Transport and Mobility Laboratory'
urlDepartment = 'http://transp-or.epfl.ch'
university = 'Ecole Polytechnique Fédérale de Lausanne (EPFL)'
urlUniversity = 'http://www.epfl.ch'
urlBiogeme = 'http://biogeme.epfl.ch'
urlForum = 'https://groups.google.com/d/forum/biogeme'


def getVersion():
    """
     Version of the software

    :return:  version number, and the release.
    :rtype: string
    """
    v = f'{versionMajor}.{versionMinor}.{versionRelease}'
    return v


__version__ = getVersion()


def getHtml():
    """Package information in HTML format

    :return: HTML code.
    :rtype: string

    """
    h = f'<p>biogeme {getVersion()} [{versionDate}]</p>\n'
    h += (
        '<p><a href="https://www.python.org/" '
        'target="_blank">Python</a> package</p>\n'
    )
    h += (
        f'<p>Home page: <a href="{urlBiogeme}" '
        f'target="_blank">{urlBiogeme}</a></p>\n'
    )
    h += (
        f'<p>Submit questions to <a href="{urlForum}" '
        f'target="_blank">{urlForum}</a></p>\n'
    )
    h += f'<p><a href="{urlAuthor}">'
    h += author
    h += '</a>, <a href="'
    h += urlDepartment
    h += '">'
    h += department
    h += '</a>, <a href="'
    h += urlUniversity
    h += '">'
    h += university.encode('ascii', 'xmlcharrefreplace').decode()
    h += '</a></p>\n'
    return h


def getText():
    """Package information in text format

    :return: package information
    :rtype: string
    """

    h = f'biogeme {getVersion()} [{versionDate}]\n'
    h += 'Version entirely written in Python\n'
    h += f'Home page: {urlBiogeme}\n'
    h += f'Submit questions to {urlForum}\n'
    h += f'{author}, {department}, {university}\n'
    return h


def getLaTeX():
    """Package information in LaTeX format

    :return: LaTeX comments
    :rtype: string
    """
    h = f'%% biogeme {getVersion()} [{versionDate}]\n'
    h += '%% Version entirely written in Python\n'
    h += f'%% Home page: {urlBiogeme}\n'
    h += f'%% Submit questions to {urlForum}\n'
    h += f'%% {author}, {department}, {university}\n'
    return h
