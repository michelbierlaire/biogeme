from datetime import timedelta


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta in a "human-readable" way"""

    # Determine the total amount of seconds
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Get the total microseconds remaining
    microseconds = td.microseconds

    # Format based on the most significant unit
    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    if minutes > 0:
        return f'{minutes}m {seconds}s'
    if seconds > 0:
        return f'{seconds}.{microseconds // 100000:01}s'
    if microseconds >= 1000:
        return f'{microseconds // 1000}ms'  # Convert to milliseconds

    return f'{microseconds}μs'  # Microseconds as is
