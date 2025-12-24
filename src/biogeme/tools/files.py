import glob
import logging
import os
import shutil
import tempfile
import uuid
from os import path

logger = logging.getLogger(__name__)


class TemporaryFile:
    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        self.name = str(uuid.uuid4())
        self.fullpath = path.join(self.dir, self.name)
        return self  # <--- return the object itself

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.dir)

    def __str__(self):
        return self.fullpath

    def remove(self):
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


def files_of_type(extension: str, name: str, all_files: bool = False) -> list[str]:
    """Identify the list of files with a given extension in the
    local directory

    :param extension: extension of the requested files (without
        the dot): 'yaml', or 'html'
    :param name: filename, without extension
    :param all_files: if all_files is False, only files containing
        the name of the model are identified. If all_files is
        True, all files with the requested extension are
        identified.

    :return: list of files with the requested extension.
    """
    if all_files:
        pattern = f"*.{extension}"
        return glob.glob(pattern)
    pattern1 = f"{name}.{extension}"
    pattern2 = f"{name}~*.{extension}"
    files = glob.glob(pattern1) + glob.glob(pattern2)
    return files


def get_file_size(path: str) -> int:
    # Log file size in human-readable form
    try:
        return os.path.getsize(path)
    except Exception as e:
        logger.warning(f"Could not determine file size for {path}: {e}")
        return -1


def print_file_size(path: str) -> str:

    size = get_file_size(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f'{size:.1f} {unit}'
