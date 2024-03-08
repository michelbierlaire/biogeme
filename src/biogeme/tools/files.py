import logging
import os
import shutil
import tempfile
import types
import uuid
from os import path
from typing import Type

logger = logging.getLogger(__name__)


class TemporaryFile:
    """Class generating a temporary file, so that the user does not
    bother about its location, or even its name

    Example::

        with TemporaryFile() as filename:
            with open(filename, 'w') as f:
                print('stuff', file=f)
    """

    def __enter__(self, name: str = None) -> str:
        self.dir = tempfile.mkdtemp()
        name = str(uuid.uuid4()) if name is None else name
        return path.join(self.dir, name)

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Destroys the temporary directory"""
        shutil.rmtree(self.dir)


def is_valid_filename(filename: str) -> tuple[bool, str]:
    """Verifies is a string is a valid filename"""
    # Check for zero length or None
    if not filename or len(filename) == 0:
        return False, 'Name is empty'

    # Define invalid characters for file names (common for Windows, Linux, and macOS)
    invalid_chars = '<>:"/\\|?*'
    if any(char in invalid_chars for char in filename):
        return False, f'Name contains one invalid char: {invalid_chars}'

    # Check for system reserved names in Windows
    if os.name == 'nt':  # nt means Windows
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        if filename.upper() in reserved_names:
            return False, f'Name is a reserved name: {reserved_names}'

    # Check filename length (common limit is 255 for many file systems, but this is OS dependent)
    if len(filename) > 255:
        return False, f'The length of the filename exceeds 255: {len(filename)}'

    return True, ''


def create_backup(filename: str, rename: bool = True) -> str:
    """If the file exist, we create a backup copy.

    :param filename: name of the file
    :param rename: if True, the file is renamed. If False, a copy is made.
    """
    original_base_name, original_extension = os.path.splitext(filename)

    # Check if the file already exists
    if os.path.exists(filename):
        # Initialize a counter for potential file name conflicts
        counter = 1
        # Generate a new file name by appending a number
        # Keep incrementing the number if the file already exists
        while True:
            new_name = f'{original_base_name}_{counter}{original_extension}'
            if not os.path.exists(new_name):
                break  # Exit loop if the file name does not exist
            counter += 1  # Increment the counter and try again

        # Rename the old file to the new file name
        if rename:
            os.rename(filename, new_name)
            logger.info(f'File {filename} has been renamed {new_name}')
        else:
            shutil.copy(filename, new_name)
            logger.info(f'File {filename} has been copied as {new_name}')
        return new_name
    logger.info(f'File {filename} does not exist. No backup has been generated')
