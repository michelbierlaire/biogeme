"""Implements the function providing names for the output files.

:author: Michel Bierlaire

:date: Tue Mar 26 16:48:40 2019
"""

# Too constraining
# pylint: disable=invalid-name

from pathlib import Path


def getNewFileName(name, ext):
    """
      Generate a file name that does not exist.

    :param name: name of the file.
    :type name: string
    :param ext: file extension.
    :type ext: string

    :return: name.ext if the file does not exists.  If it does, returns
       name~xx.ext, where xx is the smallest integer such that the
       corresponding file does not exist. It is designed to avoid erasing
       output files inadvertently.
    :rtype: string
    """
    fileName = name + '.' + ext
    theFile = Path(fileName)
    number = int(0)
    while theFile.is_file():
        fileName = f'{name}~{number:02d}.{ext}'
        theFile = Path(fileName)
        number += 1
    return fileName
