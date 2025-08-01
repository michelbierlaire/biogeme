def format_memory_size(num_bytes: float) -> str:
    """Format a number of bytes in human readable format

    :param num_bytes: number of bytes
    :return: string to be displayed
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f'{num_bytes:.2f} PB'  # for very large numbers
