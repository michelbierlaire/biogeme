import pytest

from webpage import generate
from webpage.generate import validate_release_version


def test_plain_release_version_is_accepted() -> None:
    assert validate_release_version('3.3.4') == '3.3.4'


@pytest.mark.parametrize(
    'version',
    ['3.3.4a1', '3.3.4b0', '3.3.4rc1', '3.3.4.dev1', '3.3.4.post1', '3.3.4+local'],
)
def test_provisional_release_versions_are_rejected(version: str) -> None:
    with pytest.raises(RuntimeError, match='not publishable'):
        validate_release_version(version)


def test_current_release_notes_are_rendered_in_homepage_and_faq() -> None:
    notes = generate.load_release_notes()

    assert generate.release_notes_title().startswith("What's new in Biogeme")
    assert '<code>sparse_cnl</code>' in notes
    homepage = generate.build_homepage()
    assert homepage.count('expression-based computational backend') == 2


def test_historical_release_notes_are_loaded_from_the_canonical_file() -> None:
    sections = generate.load_release_note_sections()

    assert [version for version, _ in sections] == [
        '3.3.4',
        '3.3.3',
        '3.3.2',
        '3.3.1',
        '3.2.14',
        '3.2.13',
        '3.2.12',
        '3.2.11',
        '3.2.10',
        '3.2.8',
        '3.2.6',
    ]
    assert not any('Biogeme 3.' in question for question in generate.faq)

    homepage = generate.build_homepage()
    assert homepage.count(
        'The main new feature introduced in Biogeme 3.3.3'
    ) == 1


def test_release_notes_must_match_the_package_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    notes_file = tmp_path / 'RELEASE_NOTES.md'
    notes_file.write_text('# Biogeme 0.0.0\n\nOld notes.\n', encoding='utf-8')
    monkeypatch.setattr(generate, 'RELEASE_NOTES_FILE', notes_file)

    with pytest.raises(RuntimeError, match='must begin'):
        generate.load_release_notes()
