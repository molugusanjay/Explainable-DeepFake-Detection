from collections import defaultdict
import pytest
from deepfake.sampling import sample_weights
from deepfake.sampling import validation_score
import numpy as np


def test_balances_domains_classes_and_methods_despite_unequal_sizes():
    rows = []
    for domain, counts in [('A', [2, 20, 8]), ('B', [15, 3, 30])]:
        for method, count in enumerate(counts):
            rows += [dict(dataset=domain, label=str(int(method > 0)), method=str(method))] * count
    totals = defaultdict(float)
    for row, weight in zip(rows, sample_weights(rows)):
        totals[row['dataset'], row['label']] += weight
    assert list(totals.values()) == pytest.approx([1, 1, 1, 1])


def test_missing_class_rejected():
    with pytest.raises(ValueError):
        sample_weights([dict(dataset='A', label='1', method='fake')])


def test_selection_equalizes_domains_not_image_counts():
    rows = [dict(dataset='A', label=str(i % 2)) for i in range(20)]
    rows += [dict(dataset='B', label='0'), dict(dataset='B', label='1')]
    probabilities = np.array([.1 if i % 2 == 0 else .9 for i in range(20)] + [.9, .1])
    score, domains = validation_score(rows, probabilities)
    assert score == .5
    assert all(type(key) is str for key in domains)
    assert domains['A']['balanced_accuracy'] == 1
    assert domains['B']['balanced_accuracy'] == 0
