from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .timing import timed_call


def _shape_and_dtype(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    if value is None:
        return None, None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    normalized_shape = tuple(shape) if shape is not None else None
    normalized_dtype = str(dtype) if dtype is not None else None
    return normalized_shape, normalized_dtype


def _make_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    positional = tuple(_shape_and_dtype(arg) for arg in args)
    keyword = tuple(
        sorted((key, _shape_and_dtype(value)) for key, value in kwargs.items())
    )
    return positional + keyword


@dataclass
class FunctionProfile:
    build_count: int = 0
    call_count: int = 0
    total_time: float = 0.0
    first_call_time: float | None = None
    last_call_time: float | None = None
    signatures: set[tuple[Any, ...]] = field(default_factory=set)

    @property
    def mean_time(self) -> float:
        return self.total_time / self.call_count if self.call_count else 0.0


@dataclass
class JaxExecutionProfile:
    """Lightweight profiler for JAX-related build and execution events."""

    enabled: bool = False
    functions: dict[str, FunctionProfile] = field(
        default_factory=lambda: defaultdict(FunctionProfile)
    )
    notes: list[str] = field(default_factory=list)

    def record_build(self, name: str) -> None:
        if not self.enabled:
            return
        self.functions[name].build_count += 1

    def add_note(self, message: str) -> None:
        if not self.enabled:
            return
        self.notes.append(message)

    def timed_call(
        self, name: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not self.enabled:
            return function(*args, **kwargs)

        profile = self.functions[name]
        profile.signatures.add(_make_signature(args, kwargs))

        result, elapsed = timed_call(function, *args, **kwargs)

        profile.call_count += 1
        profile.total_time += elapsed
        profile.last_call_time = elapsed
        if profile.first_call_time is None:
            profile.first_call_time = elapsed

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notes": list(self.notes),
            "functions": {
                name: {
                    "build_count": stats.build_count,
                    "call_count": stats.call_count,
                    "total_time": stats.total_time,
                    "first_call_time": stats.first_call_time,
                    "last_call_time": stats.last_call_time,
                    "mean_time": stats.mean_time,
                    "distinct_signatures": len(stats.signatures),
                    "signatures": [
                        repr(signature)
                        for signature in sorted(stats.signatures, key=repr)
                    ],
                }
                for name, stats in sorted(self.functions.items())
            },
        }

    def summary(self) -> str:
        if not self.enabled:
            return "JaxExecutionProfile(enabled=False)"

        lines = ["JAX execution profile"]
        for name, stats in sorted(self.functions.items()):
            lines.append(
                f"- {name}: builds={stats.build_count}, "
                f"calls={stats.call_count}, "
                f"first={stats.first_call_time}, "
                f"last={stats.last_call_time}, "
                f"mean={stats.mean_time}, "
                f"distinct_signatures={len(stats.signatures)}"
            )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)
