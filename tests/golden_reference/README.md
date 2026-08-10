# Golden-reference generation

This directory contains the isolated environment used to regenerate the JAX
golden-reference files under `tests/data/reference/`. Its `pyproject.toml`
pins the released Biogeme version used as the reference implementation; it is
deliberately separate from the repository environment used by the test suite
and by the JED runner.

To regenerate the references, run from this directory:

```bash
uv run python generate_reference_results.py
```

Do not use this environment for JED jobs or for ordinary repository tests.
