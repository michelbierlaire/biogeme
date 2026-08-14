import pytest

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
