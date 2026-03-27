from __future__ import annotations

from pathlib import Path

from biogeme.latent_variables.io import save_text


def test_save_text_writes_utf8_text_to_string_path(tmp_path) -> None:
    filepath = tmp_path / 'subdir' / 'output.txt'

    save_text('Hello, world!', str(filepath))

    assert filepath.exists()
    assert filepath.read_text(encoding='utf-8') == 'Hello, world!'


def test_save_text_writes_utf8_text_to_path_object(tmp_path) -> None:
    filepath = tmp_path / 'nested' / 'deeper' / 'report.txt'

    save_text('Café été naïve', filepath)

    assert filepath.exists()
    assert filepath.read_text(encoding='utf-8') == 'Café été naïve'


def test_save_text_creates_missing_parent_directories(tmp_path) -> None:
    filepath = tmp_path / 'a' / 'b' / 'c' / 'file.txt'

    assert not filepath.parent.exists()

    save_text('content', filepath)

    assert filepath.parent.exists()
    assert filepath.parent.is_dir()
    assert filepath.read_text(encoding='utf-8') == 'content'


def test_save_text_overwrites_existing_file(tmp_path) -> None:
    filepath = tmp_path / 'existing.txt'
    filepath.write_text('old content', encoding='utf-8')

    save_text('new content', filepath)

    assert filepath.read_text(encoding='utf-8') == 'new content'


def test_save_text_accepts_relative_string_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    relative_path = 'relative_dir/relative_file.txt'

    save_text('relative text', relative_path)

    filepath = Path(relative_path)
    assert filepath.exists()
    assert filepath.read_text(encoding='utf-8') == 'relative text'


def test_save_text_with_current_directory_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    save_text('root file', 'root.txt')

    filepath = tmp_path / 'root.txt'
    assert filepath.exists()
    assert filepath.read_text(encoding='utf-8') == 'root file'
