"""Relocated workflow commands must work even when called outside the project."""
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent


def invoke(monkeypatch, caller, arguments):
    workflow = runpy.run_path(str(ROOT / 'scripts' / 'run_local.py'))
    calls = []
    monkeypatch.chdir(caller)
    monkeypatch.setattr(sys, 'argv', ['run_local.py', *arguments])
    monkeypatch.setattr(workflow['subprocess'], 'run',
                        lambda command, **kwargs: calls.append((command, Path.cwd(), kwargs)))
    workflow['main']()
    return calls


@pytest.mark.parametrize('dataset,script,prepared', [
    ('dff', 'prepare_dff.py', 'dff_prepared'),
    ('ffpp', 'prepare_ffpp.py', 'prepared'),
])
def test_prepare_routes_to_relocated_script_and_project_data(monkeypatch, tmp_path, dataset, script, prepared):
    calls = invoke(monkeypatch, tmp_path, ['prepare', '--dataset', dataset])
    command, cwd, options = calls[0]
    assert Path(command[1]) == ROOT / 'scripts' / script
    assert Path(command[1]).is_file()
    assert Path(command[command.index('--output') + 1]) == ROOT / 'data' / prepared
    assert cwd == ROOT and options['check'] is True


def test_relative_approved_downloader_is_resolved_before_changing_directory(monkeypatch, tmp_path):
    downloader = tmp_path / 'approved download.py'
    downloader.write_text('# Test fixture only; never executed.\n')
    calls = invoke(monkeypatch, tmp_path,
                   ['download', '--dataset', 'ffpp', '--downloader', downloader.name, '--videos', 'videos'])
    assert len(calls) == 5
    for command, cwd, options in calls:
        assert Path(command[1]) == downloader
        assert Path(command[2]) == tmp_path / 'videos'
        assert cwd == ROOT and options['check'] is True


def test_historical_app_routes_to_source_and_results_run(monkeypatch, tmp_path):
    command, cwd, options = invoke(monkeypatch, tmp_path, ['app', '--dataset', 'ffpp'])[0]
    assert Path(command[1]) == ROOT / 'src' / 'app.py'
    assert Path(command[1]).is_file()
    assert Path(command[command.index('--checkpoint') + 1]) == ROOT / 'results' / 'runs' / 'local_frequency' / 'calibrated.pt'
    assert cwd == ROOT and options['check'] is True
