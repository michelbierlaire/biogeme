from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_docs_clean_removes_generated_sphinx_gallery_output():
    makefile = (PROJECT_ROOT / 'docs' / 'Makefile').read_text()

    assert 'GALLERYDIR = $(SOURCEDIR)/auto_examples' in makefile
    assert 'Removing generated Sphinx-Gallery output: $(GALLERYDIR)' in makefile
    assert '\t@rm -rf "$(GALLERYDIR)"' in makefile
    assert 'CLEANEXAMPLES ?=' in makefile
    assert '$(CLEANEXAMPLES) --apply' in makefile


def test_docs_build_cleans_disposable_example_outputs_even_after_failure():
    makefile = (PROJECT_ROOT / 'docs' / 'Makefile').read_text()

    assert 'cleanup_status=0' in makefile
    assert 'status=$$?' in makefile
    assert 'rm -rf "$(GALLERYDIR)"' in makefile
    assert 'exit $$status' in makefile


def test_windows_docs_build_also_cleans_disposable_example_outputs():
    make_bat = (PROJECT_ROOT / 'docs' / 'make.bat').read_text()

    assert 'clean_example_artifacts.py --apply' in make_bat
    assert 'auto_examples' in make_bat
    assert 'set CLEANUP_STATUS=%ERRORLEVEL%' in make_bat
    assert 'set BUILD_STATUS=%ERRORLEVEL%' in make_bat


def test_docs_makefile_exposes_generated_api_validation():
    makefile = (PROJECT_ROOT / 'docs' / 'Makefile').read_text()
    make_bat = (PROJECT_ROOT / 'docs' / 'make.bat').read_text()

    assert 'check-code' in makefile
    assert '\t@$(UVRUN) python create_code_rst.py --check' in makefile
    assert 'if "%1" == "check-code"' in make_bat
    assert 'create_code_rst.py --check' in make_bat


def test_biogeme_overview_is_part_of_public_documentation():
    index = (PROJECT_ROOT / 'docs' / 'source' / 'index.rst').read_text()
    overview = PROJECT_ROOT / 'docs' / 'source' / 'biogeme_overview.rst'

    assert overview.is_file()
    assert not (PROJECT_ROOT / 'docs' / 'source' / 'code_overview.rst').exists()
    assert not (PROJECT_ROOT / 'docs' / 'source' / 'biogeme.rst').exists()
    assert 'biogeme_overview.rst' in index
    assert 'code/biogeme_api.rst' in index
