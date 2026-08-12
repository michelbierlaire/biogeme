Hermetic example workflow
=========================

For the personal release-preparation walkthrough, including JED cleanup,
slow-only runs, laptop fixture import, and final Gallery checks, see the
repository-root ``RELEASE_WORKFLOW.md`` file.

Documentation examples are executable programs, but a documentation build
must not depend on files left by an earlier estimation or by another user's
working directory.  The supported workflow gives each example a fresh
temporary workspace and validates the outputs declared in the shared JED
manifest.

The manifest is ``jed_runs/jed_examples.toml``.  It is shared by the local
runner and the JED scheduler.  An example can be marked as ``self_contained``,
``dependent``, or ``server_only``.  Dependent examples are run only after the
declared predecessor has produced its required artifact.

Local validation
----------------

Run these commands from the repository root.  The ``docs`` dependency group
is locked in ``uv.lock``.

.. code-block:: bash

   uv run --locked --group docs python tools/docs_examples.py list
   uv run --locked --group docs python tools/docs_examples.py plan --profile fast
   uv run --locked --group docs python tools/docs_examples.py run --profile fast
   uv run --locked --group docs python tools/docs_examples.py status --verbose

The fast profile currently contains a small pilot set.  Add an example to
that profile only after it succeeds from a clean workspace.  To preserve the
workspace for debugging, use ``--keep-workspace``.  Local run state is stored
under the ignored ``.docs_runs`` directory and can be removed with:

.. code-block:: bash

   uv run --locked --group docs python tools/docs_examples.py clean --apply

The cleaner removes only local documentation-run state.  It does not remove
source code, input data, JED archives, or estimation results.

Artifact-dependent validation
-----------------------------

The indicators examples form the first migrated artifact-dependent family.
The estimator produces ``b02estimation.yaml`` and ``b02estimation.html``;
the seven reporting examples receive a private copy of the YAML artifact in
their own workspaces.  Validate the complete chain with:

.. code-block:: bash

   uv run --locked --group docs python tools/docs_examples.py \
      plan --script indicators/plot_b09wtp.py
   uv run --locked --group docs python tools/docs_examples.py \
      run --script indicators/plot_b09wtp.py --keep-workspace

Selecting the final consumer automatically includes its estimator.  The
runner also validates the revenue text reports produced by
``plot_b05revenues.py``.  Use ``status --verbose`` and the per-job
``stdout.txt``/``stderr.txt`` files under ``.docs_runs`` when diagnosing a
failed chain.

The next migrated family is the Swissmetro logit chain:
``plot_b01a_logit.py`` estimates the model and
``plot_b01d_logit_simul.py`` consumes its YAML result.  Validate it by
selecting the simulation consumer in the same way.

The Swissmetro normal-mixture chain follows the same contract:
``plot_b05a_normal_mixture.py`` estimates the mixture and
``plot_b05c_normal_mixture_simul.py`` consumes its YAML result.  The
estimation uses 10,000 Monte-Carlo draws, so it remains in the full profile
and is validated explicitly rather than on every fast CI run.

The Swissmetro cross-nested-logit chain is also migrated:
``plot_b11a_cnl.py`` estimates the CNL model and
``plot_b11b_cnl_simul.py`` consumes its YAML result.  Validate it by
selecting the simulation consumer.

The Swissmetro panel chain is migrated as a full-profile check:
``plot_b12_panel.py`` estimates the panel mixture and
``plot_b13_panel_simul.py`` consumes its YAML result.  The simulation uses
100,000 Monte-Carlo draws, so run it explicitly rather than as part of the
fast CI profile.

Sphinx builds
-------------

The documentation Makefile provides separate profiles:

.. code-block:: bash

   make -C docs examples-fast
   make -C docs html-fast
   make -C docs html
   make -C docs linkcheck

``examples-fast`` is the hermetic execution check.  ``html-fast`` deliberately
does not execute gallery scripts; it renders the documentation after that
check, so the Sphinx build cannot write results into the source tree.  ``html``
uses the full gallery profile and is intended for nightly or release builds
after the remaining examples have been migrated to the isolated runner.  Both
profiles generate API/configuration RST before invoking Sphinx.

JED and expensive examples
---------------------------

Long-running Bayesian, hybrid-choice, and estimation examples remain JED
jobs.  JED runs them in isolated Slurm workspaces, validates their declared
outputs, and archives only the artifacts needed by dependent jobs.  Those
archives must be promoted to documented fixtures before a release build.

Importing JED results for a release
------------------------------------

The laptop release workflow imports the completed JED artifacts explicitly;
it does not make the Sphinx build scan a server checkout or copy arbitrary
files from ``saved_results``.  Stage the server's example tree with ``rsync``
in the persistent, Git-ignored ``.release_staging`` directory, then inspect
the manifest-limited dry run:

.. code-block:: bash

   cd "$HOME/github/biogeme"
   JED_STAGE="$PWD/.release_staging/examples"
   mkdir -p "$JED_STAGE"
   JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'
   rsync -a --partial --progress --whole-file \
      -e 'ssh -o Compression=no' \
      --include='*/' \
      --include='bayesian_swissmetro/saved_results/b01a_logit.nc' \
      --include='bayesian_swissmetro/saved_results/b05_normal_mixture.nc' \
      --exclude='*.nc' \
      --include='*/saved_results/***' \
      --include='*/saved_html/***' \
      --include='revenue_*.txt' \
      --exclude='*' \
      "$JED_REMOTE/" "$JED_STAGE/"
   uv run --locked --group docs python tools/import_jed_results.py \
      --profile full --strict

The source may instead be a mounted JED checkout.  The importer accepts either
that checkout or its ``docs/source/examples`` directory.  It considers only
the ``expected_outputs`` and ``expected_output_globs`` entries for the selected
profile.  A glob is used for estimators such as the all-algorithm and
multi-model examples whose model names are generated at runtime.  YAML and
Pareto files go to each example's ``saved_results/``, HTML files to
``saved_html/``, and declared text reports remain at the example root.  Only
the two Bayesian examples that consume posterior draws declare NetCDF
fixtures; all other Bayesian examples use YAML summaries.  The single
``rsync`` invocation transfers the selected archives and required NetCDF files
in one SSH session, so the passphrase is requested only once.  For unattended
transfers on macOS, load the key into the keychain first:

.. code-block:: bash

   ssh-add --apple-use-keychain ~/.ssh/id_rsa

For a fresh staging directory, ``--whole-file`` avoids a delta-checksum pass.
If the transfer is interrupted, rerun the command; ``--partial`` keeps the
partial files.  To resume a large partial file block by block, remove
``--whole-file`` on the retry.  Do not add ``--compress`` (``-z``): NetCDF is
already compressed and SSH compression usually makes this transfer slower.
The importer never changes Python source, input data, or undeclared outputs.
The staging directory persists across terminal sessions and can be reused;
rerunning ``rsync`` updates changed files and preserves partial transfers.
Verify that Git ignores it with ``git check-ignore -v "$JED_STAGE"``.

When the dry-run list is complete and every required artifact is present, add
``--apply``:

.. code-block:: bash

   uv run --locked --group docs python tools/import_jed_results.py \
      --profile full --strict --apply

The command backs up overwritten files and records SHA-256 checksums in the
ignored ``.docs_runs/imports/<timestamp>/`` directory.  A strict import exits
with an error and makes no changes if any declared artifact is absent, so a
missing JED result is fixed before building the release documentation.  After
a successful import, run the full gallery build and its structural check:

.. code-block:: bash

   make -C docs html
   make -C docs check-html

The fast local profile remains useful for quick feedback, but it is not a
replacement for importing the full-profile JED results before a release.

The full manifest currently leaves four jobs intentionally outside the import
contract: ``assisted/plot_b09post_processing.py`` and
``swissmetro/plot_b21c_process_pareto.py``/``plot_b22c_process_pareto.py``
produce in-memory post-processing reports, while
``swissmetro/plot_b01e_logit_all_algos.py`` produces a CSV summary that is not
part of the documented YAML/HTML/Pareto fixture set.  They still run on JED
and remain visible in the job status and error reports.  Add an explicit
output contract before treating one of these jobs as a release fixture.

Adding an example
-----------------

When adding a ``plot_*.py`` file:

1. Keep input data read-only and local to the example or declare it in the
   manifest.
2. Write generated files to the runner-provided output location, or document
   the output names so the runner can harvest them.
3. Declare dependencies and expected artifacts in
   ``jed_runs/jed_examples.toml``.
4. Run the fast plan and the example from a clean workspace.
5. Add it to the full or fast profile only after output validation succeeds.

An example is hermetic when it succeeds from a clean checkout with the locked
environment and declared inputs, without reading a previous result, a home
directory cache, or a network resource.
