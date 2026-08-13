# Makefile for the Biogeme documentation.

SHELL := /bin/bash

UVRUN ?= uv run --locked --group docs
SPHINXBUILD ?= $(UVRUN) sphinx-build
DOCSEXAMPLES ?= $(UVRUN) python ../tools/docs_examples.py
SOURCEDIR = source
BUILDDIR = build
GALLERYDIR = $(SOURCEDIR)/auto_examples
PROFILE ?= full
DOCS_CACHE ?= $(CURDIR)/.cache
DOCS_ENV = MPLCONFIGDIR=$(DOCS_CACHE)/matplotlib XDG_CACHE_HOME=$(DOCS_CACHE)

.PHONY: help generate examples-fast examples-full html html-fast check-html linkcheck doctest clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

generate:
	@mkdir -p "$(DOCS_CACHE)/matplotlib"
	@$(DOCS_ENV) $(UVRUN) python create_code_rst.py --force

examples-fast:
	@$(DOCSEXAMPLES) run --profile fast

examples-full:
	@$(DOCSEXAMPLES) run --profile full

html: generate
	@set -o pipefail; $(DOCS_ENV) BIOGEME_DOCS_GALLERY_PROFILE=$(PROFILE) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee warnings.log

html-fast: generate
	@set -o pipefail; $(DOCS_ENV) BIOGEME_DOCS_GALLERY_PROFILE=none $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee warnings.log

check-html:
	@$(UVRUN) python check_generated_html.py "$(BUILDDIR)/html"

linkcheck: generate
	@set -o pipefail; $(DOCS_ENV) BIOGEME_DOCS_GALLERY_PROFILE=none $(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee linkcheck.log

doctest: generate
	@set -o pipefail; $(DOCS_ENV) BIOGEME_DOCS_GALLERY_PROFILE=none $(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee doctest.log

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(DOCSEXAMPLES) clean --apply
	@echo "Removing generated Sphinx-Gallery output: $(GALLERYDIR)"
	@rm -rf "$(GALLERYDIR)"
	@rm -rf "$(DOCS_CACHE)"
