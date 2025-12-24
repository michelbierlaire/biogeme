import logging
import time
from contextlib import contextmanager
from typing import Any

from biogeme.tools import format_elapsed_time


@contextmanager
def timeit(
    label: str,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    threshold_ms: float = 10.0,
) -> dict[str, Any]:
    """
    Measure the execution time of a block of code and report it only if
    the elapsed duration exceeds a threshold.

    :param label: Name shown in the timing message.
    :param logger: Optional logger used to emit the message. If ``None``,
        the message is printed to stdout.
    :param level: Logging level used when ``logger`` is provided.
    :param threshold_ms: Minimum duration (in milliseconds) for the timing
        message to be emitted. Defaults to ``10.0``.

    :returns: A mutable dictionary allowing the user to store values
        inside the ``with`` block (e.g., to capture return values).

    **Example**
    ::

        # Measure a block without capturing a result
        with timeit("matrix inversion"):
            x = np.linalg.inv(A)

        # Capture a computed value inside the block
        with timeit("simulate choice") as tb:
            tb["value"] = expensive_simulation()

        result = tb["value"]
        print("Simulation result:", result)
    """
    start = time.perf_counter()
    container: dict[str, Any] = {}

    try:
        yield container
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        if elapsed >= threshold_ms:
            msg = f"{label} finished in {format_elapsed_time(elapsed)}"
            if logger is not None:
                logger.log(level, msg)
            else:
                print(msg)
