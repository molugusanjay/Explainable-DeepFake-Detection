"""Calibrate on validation; evaluate frozen settings on the held-out test split."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'deepfake-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from deepfake.checkpoints import load_detector, save_atomic
from deepfake.data import load_manifest, Faces
from deepfake.metrics import metrics, cluster_accuracy_interval
from train import collect


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = load_manifest(args.manifest)
    model, checkpoint = load_detector(args.checkpoint, device)
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if digest != checkpoint['manifest_sha256']:
        raise ValueError('Use the training manifest to evaluate its held-out splits')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split = 'val' if args.calibrate else 'test'
    dataset = Faces(rows, split)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    logits, labels = collect(model, loader, device)
    if args.calibrate:
        # A bounded, single-parameter temperature fit using validation labels only.
        temperatures = torch.logspace(-.7, .7, 201)
        losses = torch.stack([torch.nn.functional.cross_entropy(logits/t, labels) for t in temperatures])
        temperature = temperatures[losses.argmin()].item()
        checkpoint.update(temperature=temperature, calibrated=True,
                          calibration_split='val', calibration_count=len(labels))
        save_atomic(checkpoint, out/'calibrated.pt')
        report = dict(split='val', temperature=temperature,
                      before=metrics(labels.numpy(), logits.softmax(1)[:, 1].numpy()),
                      after=metrics(labels.numpy(), (logits/temperature).softmax(1)[:, 1].numpy()))
        (out/'calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Calibration saved to {out / "calibrated.pt"}. Test data was not used.')
        return
    result_path = out/'test_metrics.json'
    if result_path.exists():
        raise ValueError('Test report already exists. Keep it; use a new folder for a separately documented evaluation.')
    temperature = checkpoint.get('temperature', 1.)
    probs = (logits/temperature).softmax(1)[:, 1].numpy()
    labels = labels.numpy()
    result = metrics(labels, probs)
    result.update(split='test', unit='face image', calibrated=checkpoint.get('calibrated', False),
                  temperature=temperature, model_config=checkpoint['config'],
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  manifest_sha256=digest, accuracy_target=.95,
                  target_met_on_this_test=result['accuracy'] >= .95,
                  accuracy_95pct_cluster_bootstrap=cluster_accuracy_interval(
                      labels, probs, [r['group'] for r in dataset.rows]))
    methods = np.array([r['method'] for r in dataset.rows])
    result['per_method_recall'] = {method: float(((probs[methods == method] >= .5)
                                    == labels[methods == method]).mean()) for method in sorted(set(methods))}
    if all(r.get('dataset') for r in dataset.rows):
        domains = np.array([r['dataset'] for r in dataset.rows])
        result['per_dataset'] = {domain: metrics(labels[domains == domain], probs[domains == domain])
                                 for domain in sorted(set(domains))}
        result['test_disclosure'] = 'Repeated legacy test evaluation; these test results informed earlier project discussion.'
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'source_ids', 'group', 'method', 'label', 'probability_fake'])
        for row, label, prob in zip(dataset.rows, labels, probs):
            writer.writerow([row['path'], row['source_ids'], row['group'], row['method'], int(label), float(prob)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(labels, probs >= .5, labels=[0, 1],
                                           display_labels=['REAL', 'FAKE'], ax=axes[0], colorbar=False)
    RocCurveDisplay.from_predictions(labels, probs, ax=axes[1])
    fig.suptitle('Held-out test results — face-image level')
    fig.tight_layout()
    fig.savefig(out/'test_charts.png', dpi=180)
    plt.close(fig)
    lines = [f'# Measured test results', '', f"Accuracy: {result['accuracy']:.2%}",
             f"Balanced accuracy: {result['balanced_accuracy']:.2%}", f"ROC AUC: {result['auc']:.4f}",
             f"Fake precision: {result['precision_fake']:.4f}; recall: {result['recall_fake']:.4f}; F1: {result['f1_fake']:.4f}",
             f"Face images: {result['count']} ({result['real_count']} real, {result['fake_count']} fake)",
             f"95% target met on this test: {result['target_met_on_this_test']}", '',
             'This score describes this dataset and split. It is not a guarantee for internet images.',
             'Face-detection failures were excluded; report preparation.json rejection counts alongside these metrics.',
             'Confidence intervals resample source groups to account for correlated images.']
    (out/'RESULTS.md').write_text('\n\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/prepared/manifest.csv')
    p.add_argument('--checkpoint', default='results/runs/frequency/best.pt')
    p.add_argument('--output', default='results/runs/frequency/evaluation')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    run(p.parse_args())
