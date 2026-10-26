"""Shared inference metrics and subject-level split selection."""

import csv
import re


def canonical_subject_id(value):
    match = re.search(r'(PD|NC)\d+', str(value))
    if not match:
        raise ValueError(f'Cannot identify a PD/NC subject from {value!r}.')
    return match.group(0)


def select_partition(samples, split_csv, partition, ppmi=False):
    assignments = {}
    with open(split_csv, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'subject_id', 'set'}.issubset(reader.fieldnames or []):
            raise ValueError('Split CSV must contain subject_id and set columns.')
        for row in reader:
            subject = canonical_subject_id(row['subject_id'])
            group = row['set'].strip()
            if group not in {'train', 'val', 'test'}:
                raise ValueError(f'Invalid split label: {group!r}')
            if subject in assignments and assignments[subject] != group:
                raise ValueError(f'Subject {subject} appears in multiple partitions.')
            assignments[subject] = group
    requested = {subject for subject, group in assignments.items() if group == partition}
    if not requested:
        raise ValueError(f'The split CSV has no {partition} subjects.')
    selected, found = [], set()
    for sample in samples:
        subject = canonical_subject_id(sample[3] if ppmi else sample[0])
        if subject in requested:
            selected.append(sample)
            found.add(subject)
    if requested - found:
        raise ValueError(f'{len(requested - found)} requested subjects have no matching paired inputs.')
    return selected


def classification_metrics(labels, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), 2) or not len(labels):
        raise ValueError('Expected nonempty labels and an N x 2 probability array.')
    if not np.isfinite(probabilities).all():
        raise ValueError('Predicted probabilities contain NaN or infinity.')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be 0 (NC) or 1 (PD).')
    predictions = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'f1': float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'auc': float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else None,
        'samples': int(len(labels)),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
