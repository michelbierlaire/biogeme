# Biogeme webpage release workflow

This document describes how to update, generate, review, deploy, and, if
necessary, roll back the public Biogeme webpage.

It is deliberately separate from:

- `RELEASE_WORKFLOW.md`, which prepares executable examples and documentation
  fixtures;
- the package release procedure, which verifies the Python distribution,
  creates the GitHub release, and publishes to PyPI.

The webpage depends on the output of the documentation workflow, but it has
its own content, templates, generated files, deployment target, and rollback
procedure.

## 1. How the webpage is organized

The `webpage` directory is a Python-driven static-site generator. It is not a
second Python package and it does not contain an independent build system.

| Path | Role | Edit directly? |
| --- | --- | --- |
| `webpage/generate.py` | Orchestrates generation and copies all inputs into the output tree | Only when changing the generator itself |
| `webpage/sections.py` | Main page sections: installation, documentation, resources, archives, release information, and conditions of use | Yes |
| `webpage/faq.py` | Non-release FAQ questions and answers | Yes |
| `RELEASE_NOTES.md` | Canonical current and historical release notes | Yes |
| `webpage/index.html.orig` | Main HTML template | Yes, when changing layout or fixed text |
| `webpage/card.html` | Template for one content card | Yes, when changing card layout |
| `webpage/faq.html` | Template for one FAQ item | Yes, when changing FAQ layout |
| `webpage/special.html` | Template for a release/special announcement | Yes, when changing its layout |
| `webpage/portfolio_grid.html.orig` | Data portfolio template | Yes, when changing its layout |
| `webpage/portfolio_grid_item.html` | One data-portfolio item | Yes, when changing its layout |
| `webpage/portfolio_modal.html` | Modal view for one data-portfolio item | Yes, when changing its layout |
| `webpage/data.toml` | Dataset descriptions and links used by the portfolio | Yes |
| `webpage/assets/` | Images, icons, and other static assets | Yes |
| `webpage/css/` | Static CSS | Yes |
| `webpage/js/` | Static JavaScript; `get_version.js` fetches the latest PyPI version | Yes |
| `webpage/website/` | Newly generated website | No; generated and ignored by Git |
| `webpage/website.old/` | Previous generated website retained by the generator as a local backup | No; generated and ignored by Git |
| `old_webpage/` | Legacy website generator | No; it is not part of the current release workflow |

The generator expects to be run from inside `webpage/`. It reads the Sphinx
output from `../docs/build/html`, expands the homepage templates, copies the
CSS, JavaScript, production assets, and Sphinx site to a temporary staging
directory, validates that staging directory, and only then replaces
`website/`. A successful replacement keeps the previous site in
`website.old/`; a failed generation leaves the existing site untouched.

The generated directories are ignored by Git. Commit the source files and
release content, not `webpage/website/` or `webpage/website.old/`.

## 2. Important version behavior

There are two different version mechanisms on the website:

1. `generate.py` reads `biogeme.version.__version__` and substitutes that
   value into the HTML title and generated cards.
2. `js/get_version.js` queries the live PyPI JSON endpoint and fills the
   visible version in the homepage header at runtime. It caches that value in
   the browser for one day.

Before every release, verify that the generated static version and the
published package version agree. The generator rejects every version that is
not a plain `major.minor.patch` value. Alpha, beta, release-candidate,
development, post-release, and local versions therefore stop generation with
a clear error instead of producing a publishable webpage.

Run this check from the repository root:

```bash
uv run --locked python -c \
  "import importlib.metadata as m; from biogeme.version import __version__; \
   print('source:', __version__); print('metadata:', m.version('biogeme'))"
```

The release candidate version must be the same in the source module, package
metadata, generated page title, release notes, FAQ, Git tag, GitHub release,
and PyPI.

The JavaScript version is intentionally obtained from PyPI. Therefore, before
the new package has been published, a local preview may show the previous
public version in the large homepage heading. Verify the static HTML before
publication, and perform the final browser check after PyPI publication. A
hard refresh or private browser window may be necessary because the JavaScript
stores the version in `localStorage` for 24 hours.

## 3. Preconditions

Do not start a webpage release from an arbitrary working tree. Use the same
release candidate commit that was used for the package and the example
workflow.

From the repository root:

```bash
cd /path/to/biogeme
git fetch --tags origin
git rev-parse HEAD
git status --short
uv --version
uv sync --locked --all-groups
uv run --locked python -c \
  "import sys, biogeme; print(sys.executable); print(biogeme.__file__)"
```

Check that:

- the checkout is on the intended branch or release-candidate commit;
- the working tree contains no accidental source changes;
- the locked environment is synchronized;
- `uv` is using the repository `.venv`, not an unrelated active virtual
  environment;
- the package imported by the environment comes from this checkout;
- the release version has already been selected and recorded.

The release commands do not provide a dirty-tree bypass. Generated JED
artifacts may be present as described in `RELEASE_WORKFLOW.md`, but changes to
Python, TOML, HTML templates, CSS, JavaScript, or webpage content must be
reviewed, committed, or stashed before continuing.

## 4. Update webpage content

### 4.1 Release announcement

Update the release announcement in `webpage/sections.py`:

- replace the generic or obsolete `special['New release']` text;
- mention the correct release version;
- summarize the main user-visible improvements;
- remove or revise the announcement when it is no longer the current release.

The announcement is rendered near the top of the homepage. It should be
short, accurate, and suitable for users who have not read the technical
release notes.

### 4.2 “What’s new” content

The current release description has one source:
`RELEASE_NOTES.md` in the repository root. Its first heading must be exactly
`# Biogeme major.minor.patch`, using the candidate version. The webpage
generator converts the Markdown to HTML and inserts the same content into both
the homepage card and the FAQ. This prevents the two public versions from
drifting apart.

Keep the notes focused on user-visible changes:

- the title must use the exact release version;
- claims must describe behavior that is actually present in the tagged code;
- compatibility and migration information must be included when relevant;
- important bug fixes, performance changes, deprecations, and documentation
  changes should be mentioned when useful.

Historical sections in `RELEASE_NOTES.md` are rendered as historical FAQ
entries automatically. Keep the file append-only: put the new release section
first, retain the previous sections below it, and do not copy generated HTML
from a previous release. `webpage/faq.py` contains only non-release FAQ
content.

### 4.3 Installation and documentation links

Review the installation and documentation cards in `sections.py`:

- PyPI URL;
- GitHub repository URL;
- users’ group URL;
- links to the current Sphinx site;
- supported Python versions;
- installation examples and package names.

Check that no old release-specific assistant link, local filesystem URL, or
obsolete documentation path remains in the public page.

### 4.4 Archives and historical links

The archive section contains links to historical versions. Add the new
release only when the corresponding webpage is actually available, and make
sure that old links remain valid. Do not remove historical releases merely
because the current release is newer.

### 4.5 Dataset portfolio

When adding or correcting a dataset, update `webpage/data.toml` and verify all
of the following for each entry:

- `title`;
- image path;
- short and long descriptions;
- technical-report link;
- data-file link.

Images referenced by `data.toml` must be available in the generated website’s
asset tree. Check spelling and case because a link may work on macOS but fail
on a case-sensitive server.

### 4.6 Templates, CSS, JavaScript, and assets

Use the templates for structural changes:

- `index.html.orig` for the page shell and fixed text;
- `card.html`, `faq.html`, and `special.html` for reusable components;
- `portfolio_*.html` for the data section.

Use `css/` for visual changes and `js/` for behavior changes. Review JavaScript
changes carefully: `get_version.js` makes a network request to PyPI and uses
browser-local caching, so a browser preview is not necessarily an immediate
reflection of the source tree.

Do not edit `website/index.html` or files below `website/sphinx/` by hand.

## 5. Build the documentation input

The webpage generator copies the existing Sphinx build. It does not build
Sphinx itself. Therefore, build and validate the documentation first.

From the repository root:

```bash
uv run --locked --group docs make -C docs html PROFILE=full
uv run --locked --group docs make -C docs check-html
```

For a release, also run the additional checks when their output is available:

```bash
uv run --locked --group docs make -C docs linkcheck
uv run --locked --group docs make -C docs doctest
```

Review `docs/warnings.log`, `docs/linkcheck.log`, and `docs/doctest.log`.
Warnings that are accepted must be recorded explicitly; broken links,
undefined references, failed doctests, and missing generated examples are
release blockers.

Confirm that the expected input exists before generating the webpage:

```bash
test -f docs/build/html/index.html
test -f docs/build/html/examples.html
```

If the full documentation build depends on the JED fixtures, complete
`RELEASE_WORKFLOW.md` first. Do not generate the webpage from a partial or
stale documentation tree.

## 6. Generate the website

Run the generator from the `webpage` directory, because it uses relative paths
for templates, assets, and the Sphinx build:

```bash
cd webpage
uv run --locked --group dev python generate.py
cd ..
```

The generator performs these operations:

1. parses `data.toml`;
2. loads the content dictionaries from `sections.py` and `faq.py`;
3. reads and renders all sections from `RELEASE_NOTES.md` (the first section is
   current; later sections become historical FAQ entries);
4. expands the HTML templates;
5. expands the homepage and FAQ templates;
6. creates a temporary staging directory;
7. copies CSS, JavaScript, production assets, and `docs/build/html` to the
   staging directory;
8. validates local links, fragments, placeholders, and release-local paths;
9. atomically replaces `website/` and keeps the prior site as `website.old/`.

Source-only asset templates and notes are not copied to the public asset tree.
Obsolete FAQ links are removed from the FAQ source rather than retained as
broken historical links.

If the generated site is important before the next generation, copy or archive
`webpage/website.old/` separately. The next invocation removes and recreates
`website.old/`.

## 7. Validate the generated website before deployment

Run a structural smoke test from the repository root:

```bash
test -f webpage/website/index.html
test -f webpage/website/sphinx/index.html
test -f webpage/website/sphinx/examples.html
test -f webpage/website/js/get_version.js
test -f webpage/website/assets/favicon.ico
```

The generator performs these checks automatically. They can also be inspected
manually:

```bash
rg -n '__[A-Z][A-Z0-9_]*__' webpage/website --glob '!sphinx/**'
rg -n 'file://|/Users/bierlair/|/home/bierlair/|CloudStorage/' webpage/website \
  --glob '!sphinx/**'
```

Both commands should return no matches. The generated Sphinx pages may contain
filesystem paths in source-code examples; the generator validates those pages
for links while applying release-content path checks to the webpage-owned
HTML.

Check the generated static version and release text:

```bash
rg -n 'Biogeme 3\.3\.4|What.s new in Biogeme 3\.3\.4|New release' \
  webpage/website/index.html
```

Replace `3.3.4` with the candidate version. The exact release description
should appear once in the main page and once in the FAQ section, with no stale
candidate version in the new-release material.

### 7.1 Local browser preview

Serve the generated directory, not the source directory:

```bash
cd webpage
uv run --locked --group dev python -m http.server 8000 --directory website
```

Open `http://127.0.0.1:8000/` and inspect at desktop and mobile widths.
Check:

- homepage layout and navigation;
- release announcement;
- visible version and page title;
- installation instructions;
- FAQ expansion and links;
- documentation and examples links;
- data portfolio images, modals, reports, and data downloads;
- CSS and JavaScript loading;
- no browser console errors;
- no links that unexpectedly point to the local filesystem.

Stop the server with `Ctrl-C` after the review.

Because `get_version.js` reads PyPI, the large visible version heading may
show the currently published version rather than the candidate during local
preview. Verify the static title and release content locally, then repeat the
browser check after PyPI publication.

### 7.2 Optional link and asset audit

For a release with substantial webpage changes, inspect every local link and
asset reference. Relative links must resolve below `webpage/website/`; links
to Sphinx pages must resolve below `webpage/website/sphinx/`; data links may be
external and should be checked separately.

At minimum, verify that all files copied by the generator are present:

```bash
find webpage/website/assets -type f | sort
find webpage/website/css -type f | sort
find webpage/website/js -type f | sort
find webpage/website/sphinx -maxdepth 1 -type f | sort | head
```

## 8. Review the source changes

The generated website is ignored, so review the source changes instead:

```bash
git status --short -- webpage
git diff -- webpage/generate.py webpage/sections.py webpage/faq.py \
  webpage/index.html.orig webpage/data.toml webpage/css webpage/js webpage/assets
```

Confirm that:

- only intended source files changed;
- no generated `website/` files are staged;
- no `.DS_Store`, `__pycache__`, or temporary files are included;
- external URLs are intentional and use the correct protocol;
- the release text is consistent with the package release notes;
- the Sphinx tree was generated from the same release candidate commit.

The webpage source changes should be committed before deployment. The
generated output is a deployment artifact and should not be used as the source
of truth.

## 9. Deployment

The repository currently contains the generator but does not contain a
canonical command for uploading `webpage/website/` to the EPFL web server.
Do not invent a destination or run an unreviewed `rsync --delete` command.

Before the first automated deployment is introduced, record the following
values in the release environment or an operations document, not in Git:

- remote host;
- remote directory;
- SSH identity or deployment mechanism;
- whether the target is versioned or a single live directory;
- backup directory and retention period;
- rollback command;
- required permissions and web-server cache behavior.

A safe deployment procedure is:

1. confirm the source commit and candidate version;
2. archive the current remote website;
3. run a dry-run transfer and review additions, deletions, and replacements;
4. transfer the generated `webpage/website/` tree;
5. preserve the previous remote tree until the live site has been checked;
6. verify the homepage, documentation, assets, and release links;
7. record the deployment time, commit, tag, and backup location.

If the server supports versioned directories and an atomic switch, deploy to a
new versioned directory and switch the live pointer only after validation. If
it does not, use a remote backup and a carefully reviewed transfer. Keep the
deployment command out of the public document until the host and destination
have been verified.

## 10. Deployment verification

After deployment, check both the homepage and representative deep links:

```bash
curl --fail --location --head https://biogeme.epfl.ch/
curl --fail --location --head https://biogeme.epfl.ch/sphinx/index.html
curl --fail --location --head https://biogeme.epfl.ch/sphinx/examples.html
```

Use a browser to verify the rendered result. Check the version twice:

1. the static page title and release text;
2. the JavaScript-populated visible version, after the new PyPI release is
   available.

Use a private window or clear the site’s `localStorage` if the browser still
shows a cached PyPI version.

Also verify:

- the favicon and CSS load;
- the Sphinx search page works;
- an example page opens;
- at least one dataset report and data link work;
- FAQ items expand correctly;
- there are no 404 responses in the browser network panel;
- the website is served over the intended HTTPS URL.

## 11. Rollback

### Local rollback

Before regenerating again, preserve the existing `website.old/` directory if
it is needed. The generator itself maintains only one previous local copy.

### Remote rollback

Restore the remote backup created immediately before deployment. If the server
uses versioned directories, switch the live pointer back to the previous
version. Re-run the deployment verification checks after the rollback.

Do not roll back the website by editing generated HTML manually. Correct the
source files, regenerate, and redeploy.

### Package/version mismatch

If the package was published with an incorrect version or the website points
to the wrong release, do not silently overwrite a PyPI release. PyPI releases
are immutable in normal operation. Follow the package release procedure for a
withdrawal/yank and publish a new patch version when necessary; restore the
previous webpage until the corrected package is available.

## 12. Recommended order relative to the package release

The safest order is:

1. select the release version and update package/release content;
2. prepare and validate examples using `RELEASE_WORKFLOW.md`;
3. build and validate the full Sphinx documentation;
4. update the webpage source and generate a local preview;
5. run the complete tests and package checks;
6. commit the final release candidate;
7. create and push the Git tag;
8. create the GitHub release and publish to PyPI;
9. regenerate or revalidate the webpage against the tagged commit;
10. deploy the webpage;
11. verify the live website and record the deployment.

The website should normally be deployed after PyPI publication because its
JavaScript version indicator reads the latest version from PyPI. If the site
must be deployed earlier, document and accept the temporary mismatch.

## 13. Release record

For every webpage deployment, record:

- Biogeme version;
- Git commit and tag;
- date and time in UTC;
- documentation build result;
- local preview result;
- deployment target;
- backup location;
- live URL checks;
- PyPI verification result;
- any accepted warnings or deviations.

This record makes a later rollback or investigation possible without relying
on a maintainer’s shell history.

## 14. Final checklist

- [ ] The package version and webpage candidate version agree.
- [ ] `sections.py` release announcement is current.
- [ ] `RELEASE_NOTES.md` contains the new section first and retains all prior
      historical sections; `faq.py` contains no duplicated release notes.
- [ ] Installation, PyPI, GitHub, documentation, and users’ group links work.
- [ ] Historical archive links remain valid.
- [ ] Dataset descriptions, images, reports, and data links are correct.
- [ ] The full Sphinx documentation has been built and checked.
- [ ] `webpage/website/` was generated from the validated Sphinx output.
- [ ] No template markers or local filesystem paths remain in generated HTML.
- [ ] The site was reviewed in a browser at desktop and mobile widths.
- [ ] The deployment target and backup were confirmed.
- [ ] A deployment dry run was reviewed before applying it.
- [ ] The live homepage, documentation, examples, assets, and FAQ were checked.
- [ ] The visible PyPI-derived version was checked after publication.
- [ ] The commit, tag, backup, and verification results were recorded.
