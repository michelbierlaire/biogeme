# Hermetic example workflow (developer guide)

This document is for Biogeme developers preparing a release. It is not part
of the public Sphinx documentation.

The full release sequence is documented in
[`RELEASE_WORKFLOW.md`](RELEASE_WORKFLOW.md). This guide records the contract
that makes examples safe to run locally, on JED, and through Sphinx-Gallery.

## Hermetic example contract

An example is hermetic when it succeeds from a clean checkout with the locked
`uv` environment and its declared inputs. It must not depend on:

- a result file left by an earlier run;
- a user's home directory or machine-specific path;
- an undeclared network resource;
- output produced by another example unless that dependency is declared in
  `jed_runs/jed_examples.toml`.

Generated files must be written to the example's working directory and listed
in the manifest as expected outputs. Input data and source files remain
unchanged.

## Local fast validation

From the repository root:

```bash
uv run --locked --group docs python tools/docs_examples.py list
uv run --locked --group docs python tools/docs_examples.py plan --profile fast
uv run --locked --group docs python tools/docs_examples.py run --profile fast
uv run --locked --group docs python tools/docs_examples.py status --verbose
```

Use `--keep-workspace` when diagnosing a failure. Local run state is stored in
the ignored `.docs_runs` directory.

## JED validation for slow examples

The JED workflow is incremental: it submits only examples that are not yet
complete. From the JED checkout, run:

```bash
uv run --locked --group docs python jed_runs/release_examples.py --strict
uv run --locked --group docs python jed_runs/release_phase1.py run --apply
uv run --locked --group docs python jed_runs/release_phase1.py status
```

When an example fails, repair the source, invalidate only that example, and
rerun Phase 1. Successful examples are not resubmitted:

```bash
uv run --locked --group docs python jed_runs/jed_examples.py invalidate \
    --script path/to/plot_example.py
uv run --locked --group docs python jed_runs/release_phase1.py run --apply
```

Wait until every example is `OK`, then finalize the JED phase:

```bash
uv run --locked --group docs python jed_runs/release_phase1.py finalize --apply
```

## Import results and build the documentation

On the laptop, use the phase-2 wrapper. It transfers only manifest-declared
artifacts, imports them into the example tree, and builds the documentation:

```bash
JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'
uv run --locked --group docs python jed_runs/release_phase2.py \
    run --source "$JED_REMOTE" --apply
```

If only the documentation build failed, rerun:

```bash
uv run --locked --group docs python jed_runs/release_phase2.py build --apply
```

The final public documentation build is run only after all examples and their
declared artifacts have been validated.

## Adding a new example

1. Add a `plot_*.py` script under `docs/source/examples/`.
2. Keep paths relative to the example and do not read existing results unless
   that behavior is explicitly part of the example.
3. Declare expected outputs and dependencies in
   `jed_runs/jed_examples.toml`.
4. Run the example from a clean local workspace.
5. Run the strict manifest check and the relevant JED or fast profile.
6. Confirm that the Sphinx-Gallery example succeeds after the generated
   fixtures have been imported.

For release cleanup, diagnostics, artifact contracts, and transfer details,
see [`RELEASE_WORKFLOW.md`](RELEASE_WORKFLOW.md).
