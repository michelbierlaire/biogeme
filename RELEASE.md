# Biogeme release checklist

This is the maintainer's walkthrough for preparing a Biogeme release. It is
not part of the public Sphinx documentation. The detailed procedures remain in
[RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md) for examples and documentation,
and [RELEASE_WEBPAGE.md](RELEASE_WEBPAGE.md) for the public website.

The normal release path is deliberately linear:

1. prepare and test one release candidate revision;
2. run the examples on JED until every job is successful;
3. import the validated fixtures and build the documentation on the laptop;
4. update and validate the webpage;
5. commit, tag, publish, and deploy.

The normal path does not require maintaining a run identifier, creating an
rsync staging directory, or manually copying individual result files.

## 1. Choose the release candidate

- [ ] Create or switch to the release branch.
- [ ] Select a plain major.minor.patch version.
- [ ] Update the package version and release notes.
- [ ] Do not use an alpha, beta, release-candidate, development,
      post-release, or local version.
- [ ] Update any user-facing release information that must be present before
      the examples are run.

From the repository root, verify the candidate:

~~~bash
git fetch --tags origin
git rev-parse HEAD
git status --short
uv sync --locked --all-groups
uv run --locked python -c \
  "import importlib.metadata as m; from biogeme.version import __version__; \
   print('source:', __version__); print('metadata:', m.version('biogeme'))"
~~~

- [ ] The source and installed metadata report the same version.
- [ ] The working tree contains no unreviewed authored changes.
- [ ] Commit and push this candidate revision before starting JED.

## 2. Run the package test suite

Use the same locked tox environment as GitHub Actions:

~~~bash
uv sync --locked --only-group ci --no-install-project
uv run --locked --no-sync tox -p auto
~~~

- [ ] All local tox environments pass.
- [ ] No package, source, or configuration changes are made after this test
      without repeating it.

GitHub Actions must pass again after the final commit. See
".github/workflows/tests.yml" for the CI matrix.

## 3. Run and repair the examples on JED

Use the same Git revision and locked environment in the JED checkout. Define
the interpreter there once:

~~~bash
cd /home/bierlair/github/biogeme
PY="$PWD/.venv/bin/python"
~~~

Follow the detailed instructions in
[RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md). The normal iteration is:

~~~bash
"$PY" jed_runs/release_examples.py --strict
"$PY" jed_runs/release_phase1.py run --apply
"$PY" jed_runs/release_phase1.py status
~~~

Repeat the status/repair loop until every discovered example is "OK":

- [ ] "OK": leave the successful job untouched.
- [ ] "RUNNING" or "PENDING": wait and check again.
- [ ] "ERROR": inspect the error report, fix the example, invalidate only the
      failed example (and its dependents), then rerun Phase 1.
- [ ] "NOT_DONE" or "NOT_SCHEDULED": rerun Phase 1 to submit only unfinished
      work.
- [ ] No successful example is resubmitted merely because another example
      failed.
- [ ] status --require-all-ok succeeds before leaving JED.

Do not reset the JED tree while jobs are running. For a fresh release or an
exceptional recovery, use the guarded procedures in
[RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md), not manual deletion.

## 4. Import results and build the documentation

On the laptop, let Phase 2 manage the transfer, import, cleanup, and gallery
build. Set the remote checkout, first inspect the dry run, then apply it:

~~~bash
cd /path/to/biogeme
JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'

uv run --locked --group docs python jed_runs/release_phase2.py \
    run --source "$JED_REMOTE"
uv run --locked --group docs python jed_runs/release_phase2.py \
    run --source "$JED_REMOTE" --apply
~~~

The wrapper transfers only manifest-required artifacts, imports them strictly,
cleans stale Sphinx state, runs Sphinx-Gallery, removes disposable example
outputs, and checks the generated HTML. If it reports a missing artifact or a
gallery failure, follow its diagnostic and rerun the same command after the
targeted repair. Do not rerun successful JED jobs.

- [ ] Strict import reports no missing artifacts.
- [ ] The full gallery completes successfully.
- [ ] "docs/warnings.log" contains no unreviewed documentation warnings.
- [ ] Generated HTML checks pass.
- [ ] "docs/source/auto_examples" is not staged; it is disposable build state.
- [ ] Only intentional files below "saved_results/" and "saved_html/" remain.

If the documentation source or fixtures are changed after Phase 2, repeat the
documentation build before continuing.

## 5. Update and validate the webpage

Follow [RELEASE_WEBPAGE.md](RELEASE_WEBPAGE.md) for the complete website
procedure. In order:

- [ ] Prepend a `# Biogeme major.minor.patch` section to `RELEASE_NOTES.md`
      with the user-visible changes for this version; retain all prior
      sections below it as historical notes.
- [ ] Update the short release announcement in `webpage/sections.py`.
- [ ] Do not duplicate the full notes in `sections.py` or `faq.py`; the webpage
      generator inserts `RELEASE_NOTES.md` into both locations.
- [ ] Review installation, documentation, GitHub, PyPI, and historical links.
- [ ] Check dataset descriptions and assets.
- [ ] Confirm the source and package versions agree.
- [ ] Build/check the documentation input before generating the site:

  ~~~bash
  uv run --locked --group docs make -C docs check-html
  ~~~

- [ ] Generate the website from inside "webpage/":

  ~~~bash
  cd webpage
  uv run --locked --group dev python generate.py
  cd ..
  ~~~

- [ ] Review the generated site in a browser, including documentation,
      examples, assets, FAQ, and mobile layout.
- [ ] Do not edit "webpage/website/" by hand; it is generated output.

The JavaScript version displayed by the live site comes from PyPI. Before
publication, validate the static version; after publication, perform a fresh
browser check of the PyPI-derived version.

## 6. Final review before committing

~~~bash
git status --short
git diff --check
git diff --stat
~~~

- [ ] Only intentional source, version, release-note, webpage, documentation,
      and reviewed result-fixture changes are present.
- [ ] No ".jed_runs", ".docs_runs", ".release_staging", Slurm logs, caches,
      "auto_examples", or temporary transfer files are staged.
- [ ] Root-level generated example files have been removed by the workflow.
- [ ] Required tracked fixtures such as "revenue_1.00.txt" remain present.
- [ ] The final tox and documentation checks have been run after the last
      authored change.

Commit and push the reviewed release candidate. Wait for all GitHub checks to
pass before tagging it.

## 7. Tag, publish, and deploy

- [ ] Create the Git tag for the exact validated commit.
- [ ] Create the GitHub release from that tag.
- [ ] Confirm that the release workflow builds the sdist and wheel and
      publishes them to PyPI.
- [ ] Verify the package version and files on PyPI.
- [ ] Deploy the generated webpage only after the package is public, unless a
      temporary version mismatch is explicitly accepted.
- [ ] Open the live website and verify the homepage, documentation, examples,
      assets, FAQ, and displayed version.

### 7.1 Create and verify the tag

Tag only the commit for which the final tox, documentation, JED, webpage, and
GitHub checks have passed. Do not tag a moving branch name such as `master`:
record the full commit identifier and verify the tag against that identifier.

From the repository root, replace `3.3.4` with the release version and replace
the value of `VALIDATED_COMMIT` with the full 40-character hash reported by the
final successful GitHub Actions run:

~~~bash
VERSION=3.3.4
TAG="v${VERSION}"
VALIDATED_COMMIT=0123456789abcdef0123456789abcdef01234567
VALIDATED_REMOTE_REF=origin/master

git fetch origin --tags
test -z "$(git status --porcelain)"
test "$(git rev-parse "${VALIDATED_COMMIT}^{commit}")" = \
     "$(git rev-parse "${VALIDATED_REMOTE_REF}^{commit}")"
test "$(git rev-parse --verify "${TAG}^{commit}" 2>/dev/null || true)" = ""

git tag -a "$TAG" "$VALIDATED_COMMIT" -m "Biogeme $VERSION"
test "$(git rev-parse "${TAG}^{commit}")" = \
     "$(git rev-parse "${VALIDATED_COMMIT}^{commit}")"
git show --no-patch --decorate "$TAG"
git push origin "$TAG"
~~~

The first equality check prevents tagging a local commit that is not the
validated remote branch commit. Set `VALIDATED_REMOTE_REF` to the remote
release branch if it is not `origin/master`. The second check prevents
accidentally reusing an existing release tag. The final check verifies that
the annotated tag points to the intended commit; use `TAG^{commit}` because an
annotated tag itself is a tag object, not the commit object.

After pushing, verify the remote tag before creating the release:

~~~bash
REMOTE_TAG_COMMIT=$(git ls-remote --exit-code origin "refs/tags/${TAG}^{}" \
    | cut -f1)
test "$REMOTE_TAG_COMMIT" = \
     "$(git rev-parse "${VALIDATED_COMMIT}^{commit}")"
~~~

Do not force-move a tag after it has been published. If the wrong commit was
tagged, stop and resolve the release attempt explicitly; never silently make a
published tag point at different source code.

### 7.2 Create the GitHub release from the tag

The release must be created from the already pushed tag. Creating a release
from a branch or from a different commit breaks the correspondence between the
validated source, the tag, the PyPI package, and the webpage.

There are two equivalent ways to create it.

Using the GitHub website:

1. Open the Biogeme repository on GitHub and select **Releases**.
2. Select **Draft a new release**.
3. In **Choose a tag**, select the existing tag `v3.3.4`; do not create a new
   tag from the release dialog.
4. Confirm that the displayed target commit is the full
   `VALIDATED_COMMIT` recorded above.
5. Set the release title to `Biogeme 3.3.4`.
6. Paste the corresponding section of `RELEASE_NOTES.md` into the release
   description and review the formatting.
7. Select **Publish release**. Do not leave it as a draft: the publishing
   workflow is triggered by a release whose state changes to `created`.

If the GitHub CLI is installed and authenticated, the same operation can be
performed from the repository root:

~~~bash
gh auth status
gh release create "$TAG" \
    --repo michelbierlaire/biogeme \
    --verify-tag \
    --title "Biogeme $VERSION" \
    --notes-file "/tmp/biogeme-${VERSION}-release-notes.md"
~~~

The notes file should contain only the release section intended for GitHub;
the command does not generate or edit `RELEASE_NOTES.md`. If `gh` reports that
the tag or release already exists, stop rather than creating a second release
or moving the tag.

### 7.3 Confirm the automatic publication

Creating the published release starts
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). Monitor the
**Build & publish (pure Python)** workflow under the repository's **Actions**
tab:

- [ ] the build job checks out the tagged commit;
- [ ] the sdist and wheel are built successfully;
- [ ] the PyPI publishing job succeeds in the configured `release`
      environment;
- [ ] the expected `3.3.4` files appear on PyPI.

If a job fails, inspect and correct the workflow or repository configuration,
then rerun the failed workflow from GitHub. Do not create another tag or GitHub
release merely to retry publication.

See ".github/workflows/deploy.yml" and
[RELEASE_WEBPAGE.md](RELEASE_WEBPAGE.md) for deployment-specific details and
rollback instructions.

## 8. Record the release

Keep a short release record containing:

- version;
- Git commit and tag;
- date and time (UTC);
- tox and GitHub Actions results;
- JED status result;
- documentation import/build result;
- webpage preview and deployment result;
- PyPI URL and live website URL;
- accepted warnings or deviations.

For adding or reorganizing examples, use [EXAMPLES_WORKFLOW.md](EXAMPLES_WORKFLOW.md)
and the example-specific documentation; that process is intentionally separate
from this release checklist.
