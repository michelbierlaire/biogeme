from pathlib import Path

import pytest

from docs import create_code_rst


def _write(path: Path, content: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def test_discover_modules_excludes_helpers_and_hidden_directories(tmp_path: Path):
    source = tmp_path / 'src'
    _write(source / 'demo' / '__init__.py')
    _write(source / 'demo' / 'public.py')
    _write(source / 'demo' / 'logging_tmp.py')
    _write(source / 'demo' / 'generate_jed_run.py')
    _write(source / 'demo' / 'nested' / 'child.py')
    _write(source / 'demo' / '.ipynb_checkpoints' / 'checkpoint.py')
    _write(source / 'demo' / '__pycache__' / 'cached.py')

    pages = create_code_rst.discover_modules(source)

    assert [(page.module, page.relative_rst.as_posix()) for page in pages] == [
        ('demo.nested.child', 'demo/nested/child.rst'),
        ('demo.public', 'demo/public.rst'),
    ]


def test_generation_is_deterministic_and_omits_empty_indexes(tmp_path: Path):
    source = tmp_path / 'src'
    destination = tmp_path / 'code'
    _write(source / 'demo' / '__init__.py')
    _write(source / 'demo' / 'public.py')
    _write(source / 'demo' / 'nested' / 'child.py')
    _write(source / 'demo' / 'nested' / 'data.txt')

    create_code_rst.create_rst_structure(source, destination)

    assert (destination / 'biogeme_api.rst').is_file()
    assert (destination / 'demo' / 'index.rst').is_file()
    assert (destination / 'demo' / 'public.rst').is_file()
    assert (destination / 'demo' / 'nested' / 'index.rst').is_file()
    assert not (destination / 'demo' / 'nested' / 'data.rst').exists()
    assert ':undoc-members:' not in (destination / 'demo' / 'public.rst').read_text(
        encoding='utf-8'
    )
    assert 'generated on' not in (destination / 'README.rst').read_text(
        encoding='utf-8'
    )

    pages = create_code_rst.discover_modules(source)
    create_code_rst.validate_generated_tree(destination, pages)


def test_force_generation_preserves_previous_tree_if_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / 'src'
    destination = tmp_path / 'code'
    _write(source / 'demo' / 'public.py')
    destination.mkdir()
    sentinel = destination / 'sentinel.txt'
    sentinel.write_text('keep me', encoding='utf-8')

    def fail_generation(*_args, **_kwargs):
        raise RuntimeError('synthetic generation failure')

    monkeypatch.setattr(create_code_rst, '_generate_tree', fail_generation)

    with pytest.raises(RuntimeError, match='synthetic generation failure'):
        create_code_rst.create_rst_structure(source, destination, force=True)

    assert sentinel.read_text(encoding='utf-8') == 'keep me'


def test_validation_rejects_unexpected_generated_files(tmp_path: Path):
    source = tmp_path / 'src'
    destination = tmp_path / 'code'
    _write(source / 'demo' / 'public.py')
    create_code_rst.create_rst_structure(source, destination)
    _write(destination / 'stale.rst', 'stale')

    pages = create_code_rst.discover_modules(source)
    with pytest.raises(RuntimeError, match='unexpected'):
        create_code_rst.validate_generated_tree(destination, pages)
