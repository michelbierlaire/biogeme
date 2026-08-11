#!/usr/bin/env python3
"""Perform cheap structural checks on a generated Sphinx site."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_PAGES = ('index.html', 'examples.html', 'examples_workflow.html')


def check_site(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f'HTML output directory does not exist: {root}']
    for page in REQUIRED_PAGES:
        if not (root / page).is_file():
            errors.append(f'missing required page: {page}')
    html_pages = list(root.rglob('*.html'))
    if not html_pages:
        errors.append('the HTML output directory contains no HTML pages')
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root',
        nargs='?',
        type=Path,
        default=Path(__file__).resolve().parent / 'build' / 'html',
        help='generated Sphinx HTML directory',
    )
    args = parser.parse_args(argv)
    errors = check_site(args.root)
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(f'Checked {len(list(args.root.rglob("*.html")))} HTML pages in {args.root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
