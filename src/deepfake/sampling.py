"""Balance mixed training without allowing one dataset or fake method to dominate."""
from collections import Counter
import numpy as np
from .metrics import metrics


def sample_weights(rows):
    if not all(r.get('dataset') for r in rows):
        counts = Counter(r['label'] for r in rows)
        return [1 / counts[r['label']] for r in rows]
    counts = Counter((r['dataset'], r['label'], r['method']) for r in rows)
    methods = Counter((domain, label) for domain, label, method in counts)
    for domain in {r['dataset'] for r in rows}:
        if {r['label'] for r in rows if r['dataset'] == domain} != {'0', '1'}:
            raise ValueError('Each training dataset must contain both classes')
    return [1 / (counts[r['dataset'], r['label'], r['method']] *
                 methods[r['dataset'], r['label']]) for r in rows]


def validation_score(rows, probabilities):
    labels = np.array([int(r['label']) for r in rows])
    if not all(r.get('dataset') for r in rows):
        return metrics(labels, probabilities)['balanced_accuracy'], {}
    domains = np.array([r['dataset'] for r in rows])
    results = {str(d): metrics(labels[domains == d], probabilities[domains == d])
               for d in sorted(set(domains))}
    return float(np.mean([r['balanced_accuracy'] for r in results.values()])), results
