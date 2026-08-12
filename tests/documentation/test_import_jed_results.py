from pathlib import Path
from types import SimpleNamespace

from tools import import_jed_results


def test_resolve_examples_root_accepts_checkout_and_staged_tree(tmp_path: Path):
    checkout = tmp_path / 'checkout'
    examples = checkout / 'docs' / 'source' / 'examples'
    examples.mkdir(parents=True)

    assert import_jed_results.resolve_examples_root(checkout) == examples
    assert import_jed_results.resolve_examples_root(examples) == examples


def test_build_plan_prefers_archives_and_maps_outputs(tmp_path: Path):
    source = tmp_path / 'server' / 'examples'
    target = tmp_path / 'laptop' / 'examples'
    (source / 'swissmetro' / 'saved_results').mkdir(parents=True)
    (source / 'swissmetro' / 'saved_html').mkdir(parents=True)
    (source / 'swissmetro' / 'saved_results' / 'model.yaml').write_text('yaml\n')
    (source / 'swissmetro' / 'saved_html' / 'model.html').write_text('html\n')
    (source / 'swissmetro' / 'revenue_1.00.txt').write_text('report\n')
    spec = SimpleNamespace(
        script='swissmetro/plot_model.py',
        expected_outputs=('model.yaml', 'model.html', 'revenue_1.00.txt'),
    )

    items = import_jed_results.build_plan([spec], source, target)

    assert [item.source for item in items] == [
        (source / 'swissmetro' / 'saved_results' / 'model.yaml').resolve(),
        (source / 'swissmetro' / 'saved_html' / 'model.html').resolve(),
        (source / 'swissmetro' / 'revenue_1.00.txt').resolve(),
    ]
    assert (
        items[0].target
        == (target / 'swissmetro' / 'saved_results' / 'model.yaml').resolve()
    )
    assert (
        items[1].target
        == (target / 'swissmetro' / 'saved_html' / 'model.html').resolve()
    )
    assert items[2].target == (target / 'swissmetro' / 'revenue_1.00.txt').resolve()


def test_apply_plan_backups_existing_files_and_records_missing(tmp_path: Path):
    source = tmp_path / 'server' / 'examples'
    target = tmp_path / 'laptop' / 'examples'
    source_result = source / 'indicators' / 'saved_results' / 'model.yaml'
    target_result = target / 'indicators' / 'saved_results' / 'model.yaml'
    source_result.parent.mkdir(parents=True)
    target_result.parent.mkdir(parents=True)
    source_result.write_text('new\n')
    target_result.write_text('old\n')
    spec = SimpleNamespace(
        script='indicators/plot_model.py',
        expected_outputs=('model.yaml', 'missing.html'),
    )
    items = import_jed_results.build_plan([spec], source, target)
    import_directory = tmp_path / 'state' / 'imports' / 'run'
    import_directory.mkdir(parents=True)

    records = import_jed_results.apply_plan(items, target, import_directory)

    assert target_result.read_text() == 'new\n'
    backup = import_directory / 'backup' / 'indicators' / 'saved_results' / 'model.yaml'
    assert backup.read_text() == 'old\n'
    assert [record['status'] for record in records] == ['copied', 'missing']
    assert records[0]['source_sha256'] == records[0]['target_sha256']


def test_build_plan_imports_declared_dynamic_outputs(tmp_path: Path):
    source = tmp_path / 'server' / 'examples'
    target = tmp_path / 'laptop' / 'examples'
    result = source / 'swissmetro' / 'saved_results' / 'model_000001.yaml'
    result.parent.mkdir(parents=True)
    result.write_text('yaml\n')
    spec = SimpleNamespace(
        script='swissmetro/plot_models.py',
        expected_outputs=(),
        expected_output_globs=('model_*.yaml',),
    )

    items = import_jed_results.build_plan([spec], source, target)

    assert len(items) == 1
    assert items[0].pattern is True
    assert items[0].expected == 'model_000001.yaml'
    assert items[0].source == result.resolve()
    assert (
        items[0].target
        == (target / 'swissmetro' / 'saved_results' / 'model_000001.yaml').resolve()
    )


def test_build_plan_deduplicates_archived_and_root_glob_matches(tmp_path: Path):
    source = tmp_path / 'server' / 'examples'
    target = tmp_path / 'laptop' / 'examples'
    archived = source / 'swissmetro' / 'saved_results' / 'model_000001.yaml'
    root = source / 'swissmetro' / 'model_000001.yaml'
    archived.parent.mkdir(parents=True)
    archived.write_text('archived\n')
    root.write_text('root\n')
    spec = SimpleNamespace(
        script='swissmetro/plot_models.py',
        expected_outputs=(),
        expected_output_globs=('model_*.yaml',),
    )

    items = import_jed_results.build_plan([spec], source, target)

    assert len(items) == 1
    assert items[0].source == archived.resolve()


def test_build_plan_reports_missing_dynamic_output_pattern(tmp_path: Path):
    source = tmp_path / 'server' / 'examples'
    target = tmp_path / 'laptop' / 'examples'
    (source / 'swissmetro').mkdir(parents=True)
    spec = SimpleNamespace(
        script='swissmetro/plot_models.py',
        expected_outputs=(),
        expected_output_globs=('model_*.yaml',),
    )

    items = import_jed_results.build_plan([spec], source, target)

    assert len(items) == 1
    assert items[0].pattern is True
    assert items[0].source is None
    assert items[0].expected == 'model_*.yaml'


def test_replace_result_archives_moves_stale_files_to_backup(tmp_path: Path):
    target = tmp_path / 'examples'
    kept = target / 'family' / 'saved_results' / 'new.yaml'
    stale = target / 'family' / 'saved_results' / 'old.yaml'
    kept.parent.mkdir(parents=True)
    kept.write_text('new')
    stale.write_text('old')
    item = import_jed_results.ImportItem(
        script='family/plot_model.py',
        expected='new.yaml',
        source=None,
        target=kept,
        candidates=(),
    )
    import_directory = tmp_path / 'state' / 'imports' / 'run'
    import_directory.mkdir(parents=True)

    removed = import_jed_results.replace_result_archives(
        [item], target, import_directory
    )

    assert removed == [stale]
    assert not stale.exists()
    assert (
        import_directory / 'backup' / 'family' / 'saved_results' / 'old.yaml'
    ).read_text() == 'old'
