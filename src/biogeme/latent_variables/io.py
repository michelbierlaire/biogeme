from __future__ import annotations

"""Small I/O helpers for generated outputs."""

from pathlib import Path


def save_text(text: str, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
