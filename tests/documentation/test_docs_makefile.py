from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_docs_clean_removes_generated_sphinx_gallery_output():
    makefile = (PROJECT_ROOT / 'docs' / 'Makefile').read_text()

    assert 'GALLERYDIR = $(SOURCEDIR)/auto_examples' in makefile
    assert 'Removing generated Sphinx-Gallery output: $(GALLERYDIR)' in makefile
    assert '\t@rm -rf "$(GALLERYDIR)"' in makefile
