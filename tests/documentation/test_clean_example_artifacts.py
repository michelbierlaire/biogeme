from pathlib import Path

from tools import clean_example_artifacts


def test_collect_preserves_sources_inputs_and_archived_results(tmp_path: Path):
    examples = tmp_path / 'examples'
    (examples / 'family' / 'saved_results').mkdir(parents=True)
    (examples / 'family' / 'saved_html').mkdir()
    (examples / 'family' / 'plot_model.py').write_text('print(1)\n')
    (examples / 'family' / 'data.csv').write_text('input\n')
    (examples / 'family' / 'optima.csv').write_text('input\n')
    (examples / 'family' / 'test~00.dat').write_text('input\n')
    (examples / 'family' / 'saved_results' / 'model.yaml').write_text('result\n')
    (examples / 'family' / 'saved_html' / 'model.html').write_text('result\n')
    (examples / 'family' / 'model.yaml').write_text('generated\n')
    (examples / 'family' / 'test~02.dat').write_text('generated\n')
    (examples / 'family' / 'revenue_1.00.txt').write_text('generated\n')
    (examples / 'family' / 'trace.png').write_bytes(b'generated')
    (examples / 'family' / '.DS_Store').write_bytes(b'generated')

    files, directories = clean_example_artifacts.collect_targets(
        examples, tracked_paths={'family/plot_model.py', 'family/test~00.dat'}
    )

    assert directories == []
    assert {path.name for path in files} == {
        '.DS_Store',
        'model.yaml',
        'revenue_1.00.txt',
        'test~02.dat',
        'trace.png',
    }


def test_collect_removes_cache_directories_but_not_symlinks(tmp_path: Path):
    examples = tmp_path / 'examples'
    cache = examples / 'family' / '__pycache__'
    cache.mkdir(parents=True)
    (cache / 'module.pyc').write_bytes(b'cache')
    (examples / 'family' / 'plot_model.py').write_text('print(1)\n')

    files, directories = clean_example_artifacts.collect_targets(
        examples, tracked_paths={'family/plot_model.py'}
    )

    assert files == []
    assert directories == [cache]
