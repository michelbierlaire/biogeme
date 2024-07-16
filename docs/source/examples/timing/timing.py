import time
from datetime import timedelta
from typing import Callable, Any


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


class Timing:
    """Call for timing the execution of callable functions"""

    def __init__(self, warm_up_runs: int = 10, num_runs: int = 100):
        self._warm_up_runs = warm_up_runs
        self._num_runs = num_runs

    @property
    def warm_up_runs(self) -> int:
        return self._warm_up_runs

    @warm_up_runs.setter
    def warm_up_runs(self, value: int):
        if value < 0:
            raise ValueError("warm_up_runs must be non-negative")
        self._warm_up_runs = value

    @property
    def num_runs(self) -> int:
        return self._num_runs

    @num_runs.setter
    def num_runs(self, value: int):
        if value < 1:
            raise ValueError("num_runs must be at least 1")
        self._num_runs = value

    def time_function(
        self,
        callable_func: Callable,
        kwargs: dict[str, Any],
    ) -> float:
        """
        Times the execution of a callable function, excluding the warm-up period.

        :param callable_func: The function to be timed.
        :param kwargs: Dictionary of keyword arguments to pass to the function. Note that all arguments must be with
            keyword
        :return: Average time per run.
        """

        # Warm-up period
        for i in range(self.warm_up_runs):
            callable_func(**kwargs)

        # Timing the function
        start_time = time.time()
        for _ in range(self.num_runs):
            callable_func(**kwargs)
        end_time = time.time()

        # Calculate the average time per run
        total_time = end_time - start_time
        average_time_per_run = total_time / self.num_runs

        return average_time_per_run
