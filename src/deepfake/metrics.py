import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix,
                             log_loss, brier_score_loss)


def metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation requires both classes')
    prediction = (probabilities >= .5).astype(int)
    ece = 0.
    confidence = np.maximum(probabilities, 1-probabilities)
    correct = prediction == labels
    for lo in np.arange(0, 1, .1):
        selected = (confidence >= lo) & (confidence < lo+.1 if lo < .9 else confidence <= 1)
        if selected.any():
            ece += selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return dict(accuracy=float(accuracy_score(labels, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
                precision_fake=float(precision_score(labels, prediction, zero_division=0)),
                recall_fake=float(recall_score(labels, prediction)),
                f1_fake=float(f1_score(labels, prediction, zero_division=0)),
                auc=float(roc_auc_score(labels, probabilities)),
                log_loss=float(log_loss(labels, np.column_stack([1-probabilities, probabilities]))),
                brier_score=float(brier_score_loss(labels, probabilities)),
                expected_calibration_error=float(ece),
                confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
                count=int(len(labels)), real_count=int((labels == 0).sum()),
                fake_count=int((labels == 1).sum()), threshold=.5)


def cluster_accuracy_interval(labels, probabilities, groups, repetitions=1000):
    """Resample source-pair clusters, not correlated individual frames."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return None
    correct = (np.asarray(probabilities) >= .5) == np.asarray(labels)
    sizes = np.array([(groups == g).sum() for g in unique])
    successes = np.array([correct[groups == g].sum() for g in unique])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    values = successes[draws].sum(1)/sizes[draws].sum(1)
    return np.quantile(values, [.025, .975]).tolist()
