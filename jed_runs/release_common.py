"""Shared helpers for the incremental release workflow.

The release command-line tools deliberately keep their state below the
ignored ``.jed_runs`` directory.  Users therefore work with the phase
commands and do not need to manage release identifiers themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / '.jed_runs' / 'releases'
CURRENT_RELEASE = RELEASE_ROOT / 'current.json'


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a release-start command finds uncommitted files."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash() -> str:
    return sha256(PROJECT_ROOT / 'jed_runs' / 'jed_examples.toml')


def git_revision() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'git rev-parse failed')
    return result.stdout.strip()


def git_status() -> list[str]:
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'git status failed')
    return [line for line in result.stdout.splitlines() if line]


def command_text(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    apply: bool,
    environment: dict[str, str] | None = None,
) -> int:
    """Print and optionally execute one command.

    The command is always printed.  A dry run never starts a subprocess.
    """

    prefix = 'RUN' if apply else 'PLAN'
    print(f'[{prefix}] {command_text(command)}')
    if not apply:
        return 0
    env = None
    if environment is not None:
        env = os.environ.copy()
        env.update(environment)
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    return result.returncode


def next_steps(lines: Iterable[str]) -> None:
    print('\nNEXT STEPS')
    print('----------')
    for index, line in enumerate(lines, start=1):
        print(f'{index}. {line}')


def load_current_release() -> dict[str, Any] | None:
    if not CURRENT_RELEASE.is_file():
        return None
    pointer = json_load(CURRENT_RELEASE)
    release_id = pointer.get('release_id')
    if not isinstance(release_id, str) or not release_id:
        raise ValueError(f'Invalid current release pointer: {CURRENT_RELEASE}')
    state_path = RELEASE_ROOT / release_id / 'release.json'
    if not state_path.is_file():
        raise ValueError(f'Current release state is missing: {state_path}')
    return json_load(state_path)


def release_directory(release: dict[str, Any]) -> Path:
    release_id = release.get('release_id')
    if not isinstance(release_id, str) or not release_id:
        raise ValueError('Release state has no release_id')
    return RELEASE_ROOT / release_id


def ensure_release(*, apply: bool, phase: str) -> dict[str, Any]:
    """Return the current release, creating it only in apply mode.

    A changed revision or manifest never silently reuses an active release.
    This is the main guard against mixing artifacts from different source
    trees.  The release identifier is kept internal to the state directory.
    """

    revision = git_revision()
    current_manifest = manifest_hash()
    existing = load_current_release()
    if existing is not None:
        complete = bool(existing.get('phase2', {}).get('built'))
        revision_mismatch = existing.get('revision') != revision
        manifest_mismatch = existing.get('manifest_sha256') != current_manifest
        if revision_mismatch:
            # Phase 2 is a local, resumable operation.  Its state can be left
            # behind when the release tooling itself is committed between an
            # interrupted transfer/import and the next attempt.  When there
            # is no Phase 1 state in the record, no JED jobs are being
            # resumed, so it is safe to start a new local Phase 2 attempt and
            # leave the old record as history.  An active Phase 1 record is
            # deliberately still protected: silently mixing its artifacts
            # with a different checkout would invalidate the release.
            phase2_only = (
                phase == 'phase2'
                and existing.get('phase') == 'phase2'
                and not existing.get('phase1')
            )
            if complete or (phase2_only and not manifest_mismatch):
                if phase2_only and not complete:
                    print(
                        'The previous incomplete Phase 2 state belongs to an '
                        'older Git revision; starting a new local Phase 2 '
                        'attempt.'
                    )
                existing = None
            else:
                raise RuntimeError(
                    'The current release uses a different Git revision. Run the '
                    'release reset command or finish that release before starting '
                    'another one.'
                )
        if existing is not None and manifest_mismatch:
            # A manifest change can alter the declared workload or artifact
            # contract.  Unlike a tooling-only revision change, it must never
            # be silently adopted by an incomplete release.
            if complete:
                existing = None
            else:
                raise RuntimeError(
                    'The release manifest changed during the current release. '
                    'Reset or explicitly finish the current release before '
                    'continuing.'
                )
        if existing is not None:
            return existing

    release_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    release = {
        'release_id': release_id,
        'created_at': now_iso(),
        'revision': revision,
        'manifest_sha256': current_manifest,
        'phase': phase,
        'phase1': {},
        'phase2': {},
    }
    print(f'Using a new release attempt for revision {revision[:12]}.')
    if not apply:
        print('The release state will be created only when --apply is supplied.')
        return release
    directory = release_directory(release)
    directory.mkdir(parents=True, exist_ok=False)
    json_dump(directory / 'release.json', release)
    json_dump(CURRENT_RELEASE, {'release_id': release_id})
    return release


def save_release(release: dict[str, Any]) -> None:
    json_dump(release_directory(release) / 'release.json', release)


def python_command(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script), *arguments]


def _porcelain_path(status_line: str) -> Path:
    """Extract the path from one ``git status --porcelain`` line."""
    value = status_line[3:] if len(status_line) >= 3 else status_line
    if ' -> ' in value:
        value = value.rsplit(' -> ', 1)[-1]
    return Path(value)


def _is_generated_release_path(status_line: str) -> bool:
    """Return whether a dirty path is an expected generated release artifact."""
    path = _porcelain_path(status_line)
    parts = path.parts
    # Slurm smoke-test diagnostics are written at the repository root by the
    # cluster wrapper.  They are disposable runtime artifacts, not source
    # changes.  Keep the match deliberately narrow so an arbitrary root-level
    # ``.err`` file still blocks a release.
    if path.parent == Path('.') and path.name.startswith('biogeme-smoke-'):
        return path.suffix in {'.err', '.out'}
    if 'docs' in parts and 'source' in parts and 'examples' in parts:
        if 'saved_results' in parts or 'saved_html' in parts:
            return True
        name = path.name
        return (
            name.startswith(('slurm-', 'revenue_', 'test~'))
            or name.endswith(('.run', '_slurm.out'))
        )
    return False


def ensure_clean_tree(*, allow_generated: bool = False) -> None:
    dirty = git_status()
    generated = [item for item in dirty if _is_generated_release_path(item)]
    relevant = [item for item in dirty if item not in generated or not allow_generated]
    if relevant:
        preview = '\n'.join(relevant[:20])
        suffix = '\n...' if len(relevant) > 20 else ''
        preservation_hint = ''
        if generated and not allow_generated:
            preservation_hint = (
                '\n\nExisting archived results are not removed by this check. To preserve '
                'them while making the checkout clean, first inspect them with:\n'
                '  $PY jed_runs/jed_commit_results.py --dry-run\n'
                'Then either commit the reviewed archives, or stash the generated '
                'files with `git stash push --include-untracked`. The root-level '
                'copies can be removed separately with:\n'
                '  $PY jed_runs/jed_cleanup.py\n'
                '  $PY jed_runs/jed_cleanup.py --apply\n'
                'Do not use release_reset.py or jed_fresh_start.py if the archived '
                'results must be retained.'
            )
        elif generated:
            preservation_hint = (
                '\n\nThe generated paths may remain; they do not block the release. '
                'The paths above are authored or unrecognized changes and must '
                'be committed or stashed. If the generated archives are a '
                'historical snapshot, preserve them before starting a fresh '
                'release with:\n'
                '  $PY jed_runs/jed_commit_results.py --dry-run\n'
                'or:\n'
                '  git stash push --include-untracked -m "Biogeme generated release artifacts"'
            )
        else:
            preservation_hint = (
                '\n\nIf the listed files are generated artifacts you want to retain, '
                'inspect them first and either commit or stash them; do not reset '
                'the release until they are safely preserved.'
            )
        raise DirtyWorkingTreeError(
            'The Git working tree contains authored or unrecognized changes. '
            'Release commands require a clean checkout; do not bypass this '
            'check.\n'
            + preview
            + suffix
            + preservation_hint
        )
    if allow_generated and generated and not relevant:
        print(
            f'INFO: ignoring {len(generated)} generated release artifact(s); '
            'they are handled by the release workflow.'
        )
