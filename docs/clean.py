import os
import fnmatch


def remove_files(directory, patterns, exclude_dir=None):
    """
    Remove files with specified patterns from a directory and its subdirectories.

    :param directory: The root directory to clean.
    :param patterns: List of file extension patterns to remove.
    :param exclude_dir: Directory name to exclude from removal.
    """
    for root, dirs, files in os.walk(directory):
        # Skip the excluded directory
        if exclude_dir and exclude_dir in root:
            continue

        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                file_path = os.path.join(root, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")


def clean_directory(directory):
    """
    Clean the specified directory by removing certain files.

    :param directory: The directory to clean.
    """
    remove_files(directory, ['*.iter'])
    remove_files(
        directory,
        ['*.html', '*.pickle', '*.pareto', '*.tex'],
        exclude_dir='saved_results',
    )


if __name__ == "__main__":
    dir_to_clean = 'source/examples'  # Replace with the path to your directory
    clean_directory(dir_to_clean)
