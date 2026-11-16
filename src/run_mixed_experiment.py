"""Reproduce the mixed-data comparison without changing previous experiments."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from deepfake.data import load_manifest

ROOT = Path(__file__).resolve().parent.parent  # project root (parent of src/)
os.chdir(ROOT)
OUT = ROOT / 'results' / 'runs' / 'mixed_comparison'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'models' / 'assets' / 'matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / 'models' / 'assets' / 'torch')


def status(stage, **extra):
    state = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = OUT/'status.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(OUT/'status.json')
    print(stage, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status('verifying_manifests')
    manifest = ROOT / 'data' / 'mixed_manifest.csv'
    if not manifest.exists():
        rows = []
        for domain, folder in [('DFF', 'dff_prepared'), ('FFPP', 'prepared')]:
            for row in load_manifest(ROOT / 'data' / folder / 'manifest.csv'):
                rows.append(dict(path=f"{folder}/{row['path']}", label=row['label'],
                    split=row['split'], group=domain+':'+row['group'],
                    source_ids='|'.join(domain+':'+s for s in row['source_ids'].split('|')),
                    sha256=row['sha256'], method=row['method'], dataset=domain))
        with manifest.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    load_manifest(manifest)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        initialization='ImageNet ConvNeXt Tiny for both models', epochs=6, patience=3,
        lr=4e-5, batch_size=8, accumulation=2, seed=42,
        sampling='Equal dataset and class mass; equal fake-method mass within each dataset',
        selection='Mean of dataset validation balanced accuracies at threshold 0.5',
        calibration='Validation-only temperature scaling',
        test_disclosure='Legacy test splits were previously inspected. These are repeated benchmark evaluations, not new untouched tests.',
        deployment='Frequency model selected on validation only; no model selection using test accuracy',
        target='95% measured accuracy is a goal, not a guaranteed result')
    if not (OUT / 'experiment_protocol.json').exists():
        (OUT / 'experiment_protocol.json').write_text(json.dumps(protocol, indent=2))
    for name, mode in [('mixed_frequency', 'both'), ('mixed_cnn', 'cnn')]:
        run = ROOT / 'results' / 'runs' / name
        run.mkdir(parents=True, exist_ok=True)
        commands = [
            ('training', ['src/train.py', '--manifest', str(manifest), '--output', str(run),
             '--epochs', '6', '--patience', '3', '--mode', mode, '--lr', '0.00004',
             '--batch-size', '8', '--accumulation', '2', '--workers', '0']),
            ('calibration', ['src/evaluate.py', '--calibrate', '--manifest', str(manifest),
             '--checkpoint', str(run / 'best.pt'), '--output', str(run), '--workers', '0']),
            ('evaluation', ['src/evaluate.py', '--manifest', str(manifest), '--checkpoint',
             str(run / 'calibrated.pt'), '--output', str(run / 'evaluation'), '--workers', '0'])]
        for stage, command in commands:
            marker = run / (stage + '.complete')
            if marker.exists():
                continue
            if stage == 'training' and (run / 'last.pt').exists():
                command += ['--resume', str(run / 'last.pt')]
            status(stage, run=name)
            with (run / (stage + '.log')).open('a', encoding='utf-8') as f:
                subprocess.run([sys.executable, '-u', *command], stdout=f,
                               stderr=subprocess.STDOUT, check=True)
            marker.write_text(datetime.now(timezone.utc).isoformat())
    status('complete')



if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        main()
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
