"""Evaluate frozen checkpoints on the additional source-disjoint DFF holdout."""
import csv
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepfake.checkpoints import load_detector
from deepfake.data import Faces, load_manifest
from deepfake.metrics import metrics, cluster_accuracy_interval
from train import collect


def main():
    root = Path(__file__).resolve().parent.parent  # project root (parent of src/)
    if not (root / 'models' / 'checkpoints' / 'final_frequency' / 'selection.json').is_file():
        raise ValueError('Freeze final model selection before opening the additional holdout')
    manifest = root / 'data' / 'fresh_dff' / 'manifest.csv'
    rows = load_manifest(manifest, required_splits=('test',))
    if any(r['split'] != 'test' for r in rows):
        raise ValueError('External holdout must contain only test examples')
    original = list(csv.DictReader((root / 'data' / 'dff_prepared' / 'manifest.csv').open()))
    sources = {s for r in original for s in r['source_ids'].split('|')}
    hashes = {r['sha256'] for folder in ['dff_prepared', 'prepared']
              for r in csv.DictReader((root / 'data' / folder / 'manifest.csv').open())}
    assert not sources & {s for r in rows for s in r['source_ids'].split('|')}
    assert not hashes & {r['sha256'] for r in rows}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Faces(rows, 'test')
    for name, run_dir in [
        ('mixed_frequency', root / 'results' / 'runs' / 'mixed_frequency'),
        ('mixed_cnn',       root / 'results' / 'runs' / 'mixed_cnn'),
        ('final_frequency', root / 'models' / 'checkpoints' / 'final_frequency'),
    ]:
        out = run_dir / 'fresh_evaluation'
        if (out / 'test_metrics.json').exists():
            continue
        model, metadata = load_detector(run_dir / 'calibrated.pt', device)
        logits, labels = collect(model, DataLoader(dataset, batch_size=8, num_workers=0), device)
        probs = (logits / metadata['temperature']).softmax(1)[:, 1].numpy()
        labels = labels.numpy()
        report = metrics(labels, probs)
        report.update(dataset='DeepFakeFace additional holdout', split='test',
            target_met_on_this_test=report['accuracy'] >= .95,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((run_dir / 'calibrated.pt').read_bytes()).hexdigest(),
            accuracy_95pct_cluster_bootstrap=cluster_accuracy_interval(labels, probs, [r['group'] for r in rows]),
            selection_policy='Unused source pairs frozen before training completed; no model selection on this holdout')
        out.mkdir(exist_ok=True)
        (out / 'test_metrics.json').write_text(json.dumps(report, indent=2))
        with (out / 'test_predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'group', 'label', 'probability_fake'])
            writer.writerows((r['path'], r['group'], int(y), float(p)) for r, y, p in zip(rows, labels, probs))
        print(name, json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()

