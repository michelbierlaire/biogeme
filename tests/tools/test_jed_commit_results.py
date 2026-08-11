import subprocess
from pathlib import Path

import pytest

from jed_runs.jed_commit_results import commit_results


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ['git', *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def initialized_repository(tmp_path: Path) -> Path:
    git(tmp_path, 'init')
    git(tmp_path, 'config', 'user.name', 'Biogeme tests')
    git(tmp_path, 'config', 'user.email', 'biogeme-tests@example.com')
    (tmp_path / 'README.md').write_text('checkout')
    (tmp_path / '.gitignore').write_text('*.nc\n')
    git(tmp_path, 'add', 'README.md', '.gitignore')
    git(tmp_path, 'commit', '-m', 'Initial commit')
    (tmp_path / 'docs' / 'source' / 'examples').mkdir(parents=True)
    return tmp_path


def test_commits_only_archived_results(tmp_path: Path):
    repository = initialized_repository(tmp_path)
    results = repository / 'docs/source/examples/swissmetro/saved_results'
    html = repository / 'docs/source/examples/swissmetro/saved_html'
    results.mkdir(parents=True)
    html.mkdir()
    (results / 'model.yaml').write_text('result')
    (results / 'model.nc').write_bytes(b'netcdf result')
    (html / 'model.html').write_text('<html></html>')
    (repository / 'unrelated.txt').write_text('leave uncommitted')

    assert commit_results(repository, message='Archive JED results') == 0

    assert git(repository, 'log', '-1', '--format=%s').strip() == (
        'Archive JED results'
    )
    committed = set(git(repository, 'show', '--format=', '--name-only').splitlines())
    assert committed == {
        'docs/source/examples/swissmetro/saved_html/model.html',
        'docs/source/examples/swissmetro/saved_results/model.nc',
        'docs/source/examples/swissmetro/saved_results/model.yaml',
    }
    assert (repository / 'unrelated.txt').is_file()


def test_refuses_unrelated_staged_changes(tmp_path: Path):
    repository = initialized_repository(tmp_path)
    (repository / 'unrelated.txt').write_text('do not commit')
    git(repository, 'add', 'unrelated.txt')
    results = repository / 'docs/source/examples/swissmetro/saved_results'
    results.mkdir(parents=True)
    (results / 'model.yaml').write_text('result')

    with pytest.raises(RuntimeError, match='unrelated files are already staged'):
        commit_results(repository)

    assert not git(repository, 'log', '-1', '--format=%s').startswith(
        'Update JED example results'
    )


def test_dry_run_does_not_stage_results(tmp_path: Path):
    repository = initialized_repository(tmp_path)
    results = repository / 'docs/source/examples/swissmetro/saved_results'
    results.mkdir(parents=True)
    (results / 'model.yaml').write_text('result')

    assert commit_results(repository, dry_run=True) == 0
    assert git(repository, 'diff', '--cached', '--name-only').strip() == ''
