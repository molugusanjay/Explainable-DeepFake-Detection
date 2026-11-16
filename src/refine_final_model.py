"""One bounded mixed-data refinement, selected using validation only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

def main():
    root = Path(__file__).resolve().parent.parent  # project root (parent of src/)
    os.chdir(root)
    run = root / 'results' / 'runs' / 'mixed_transfer'
    run.mkdir(parents=True, exist_ok=True)
    manifest = root / 'data' / 'mixed_manifest.csv'
    parent = root / 'results' / 'runs' / 'local_frequency' / 'calibrated.pt'
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), epochs=4, patience=3, lr=1e-5,
        seed=42, batch_size=8, accumulation=2,
        initialization='Earlier DFF to FFPP transfer model, weights only',
        parent_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        selection='Compare mean dataset validation balanced accuracy with mixed_frequency; retain the higher score',
        disclosure='Refinement planned after inspecting legacy mixed test results. Additional DFF holdout remains unexamined.',
        final_test_policy='Freeze selection before evaluating the additional DFF holdout; no further tuning')
    if not (run / 'experiment_protocol.json').exists():
        (run / 'experiment_protocol.json').write_text(json.dumps(protocol, indent=2))
    commands = [
        ('training', ['src/train.py', '--manifest', str(manifest), '--output', str(run),
         '--epochs', '4', '--patience', '3', '--lr', '0.00001', '--workers', '0',
         '--batch-size', '8', '--accumulation', '2']),
        ('calibration', ['src/evaluate.py', '--calibrate', '--manifest', str(manifest),
         '--checkpoint', str(run / 'best.pt'), '--output', str(run), '--workers', '0'])]
    for stage, command in commands:
        if (run / (stage + '.complete')).exists():
            continue
        if stage == 'training':
            command += (['--resume', str(run / 'last.pt')] if (run / 'last.pt').exists()
                        else ['--init-checkpoint', str(parent)])
        print('Refinement ' + stage, flush=True)
        with (run / (stage + '.log')).open('a') as f:
            subprocess.run([sys.executable, '-u', *command], stdout=f, stderr=subprocess.STDOUT, check=True)
        (run / (stage + '.complete')).write_text(datetime.now(timezone.utc).isoformat())
    scores = {}
    for name in ['mixed_frequency', 'mixed_transfer']:
        history = json.loads((root / 'results' / 'runs' / name / 'history.json').read_text())
        scores[name] = max(h['validation']['selection_score'] for h in history)
    selected = max(scores, key=scores.get)
    final = root / 'models' / 'checkpoints' / 'final_frequency'
    final.mkdir(parents=True, exist_ok=True)
    if not (final / 'selection.json').exists():
        for filename in ['calibrated.pt', 'history.json', 'calibration.json']:
            shutil.copy2(root / 'results' / 'runs' / selected / filename, final / filename)
        (final / 'selection.json').write_text(json.dumps(dict(selected_run=selected,
            validation_scores=scores, rule='Highest macro validation balanced accuracy',
            holdout_used_for_selection=False), indent=2))
    if not (final / 'evaluation' / 'test_metrics.json').exists():
        subprocess.run([sys.executable, 'src/evaluate.py', '--manifest', str(manifest),
            '--checkpoint', str(final / 'calibrated.pt'),
            '--output', str(final / 'evaluation'), '--workers', '0'], check=True)
    print('Final selected candidate: ' + selected, flush=True)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
