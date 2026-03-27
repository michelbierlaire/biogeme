from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .timing import timed_call


@dataclass
class BenchmarkCaseResult:
    label: str
    first_call_time: float
    second_call_time: float | None
    repeated_call_times: list[float] = field(default_factory=list)

    @property
    def mean_repeated_time(self) -> float | None:
        if not self.repeated_call_times:
            return None
        return sum(self.repeated_call_times) / len(self.repeated_call_times)


@dataclass
class BenchmarkSeriesResult:
    label: str
    result: BenchmarkCaseResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.label}: first={self.result.first_call_time}, "
            f"second={self.result.second_call_time}, "
            f"mean_repeated={self.result.mean_repeated_time}"
        )


def run_benchmark(
    label: str,
    function: Callable[..., Any],
    *args: Any,
    repeats: int = 5,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BenchmarkSeriesResult:
    """Run a simple benchmark with first-call, second-call, and repeated timings."""
    _, first = timed_call(function, *args, **kwargs)

    second: float | None = None
    repeated_times: list[float] = []

    if repeats >= 1:
        _, second = timed_call(function, *args, **kwargs)

    for _ in range(max(0, repeats - 1)):
        _, elapsed = timed_call(function, *args, **kwargs)
        repeated_times.append(elapsed)

    result = BenchmarkCaseResult(
        label=label,
        first_call_time=first,
        second_call_time=second,
        repeated_call_times=repeated_times,
    )
    return BenchmarkSeriesResult(
        label=label,
        result=result,
        metadata={} if metadata is None else metadata,
    )
