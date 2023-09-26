"""Simple tuple storing validity information

:author: Michel Bierlaire
:date: Fri Aug 18 16:36:22 2023

Stores the validity status and the explanation if invalid.
"""
from typing import NamedTuple


class Validity(NamedTuple):
    status: bool
    reason: str
