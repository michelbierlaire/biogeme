"""Scripts deleting Biogeme output files

:author: Michel Bierlaire
:date: Sun Jul 30 15:58:57 2023
"""
import os
import sys

EXTENSIONS_TO_CLEAN = ['html', 'pickle', 'iter', 'log', 'pareto', '.tex', '.F12', '~', '#']


def file_to_erase(the_filename: str) -> bool:
    """Checks if a file must be erased

    :param the_filename: name fo the file
    :type the_filename: str

    :return: True if the file must be erased. False otherwise.
    :rtype: bool
    """
    for ext in EXTENSIONS_TO_CLEAN:
        if the_filename.endswith(ext):
            return True

    return False


def get_file_list():
    """Retrieves the list of files to be erased.

    :return: list of file names
    :rtype: list[str]
    """
    
    the_list = []
    current_directory = os.getcwd()
    for dirpath, _, filenames in os.walk(current_directory):
        for filename in filenames:
            if file_to_erase(filename):
                file_path = os.path.join(dirpath, filename)
                the_list.append(file_path)
    return the_list


def ask_for_confirmation(the_list):
    """Function asking confirmation to the user.

    :param the_list: luist of files to erase
    :type the_list: list[str]

    :return: True if OK to proceed, False otherwise.
    :rtype: bool
    """
    while True:
        print('This script will erase the following files:')
        print('\n'.join(the_list))
        text = 'Do you want to proceed? (yes/no): '
        response = input(text).strip().lower()
        if response in ('yes', 'y'):
            return True
        if response in ('no', 'n'):
            return False
        print("Invalid response. Please enter 'yes' or 'no'.")


file_list = get_file_list()
if not file_list:
    print('No file to be cleaned.')
    sys.exit(0)


if ask_for_confirmation(file_list):
    for file_name in file_list:
        os.remove(file_name)
    if len(file_list) == 1:
        print('One file has been erased.')
    else:
        print(f'{len(file_list)} files have been erased.')

else:
    print("Script execution aborted.")
