"""Metrics with the numerator and denominator kept visible.

Accuracy is reported as a fraction of actual counts, never as a bare percentage, because the
validation split holds 32 clips and a percentage alone hides how few that is.
"""
from __future__ import annotations

from collections import Counter

from .data import CLASSES


def confusion(y_true, y_pred, n_classes: int = len(CLASSES)):
    m = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
        m[t][p] += 1
    return m


def report(y_true, y_pred, signers=None) -> dict:
    n = len(y_true)
    undecided = sum(1 for p in y_pred if p is None)
    correct = sum(1 for t, p in zip(y_true, y_pred) if p is not None and t == p)
    m = confusion(y_true, y_pred)

    per_class, f1s = {}, []
    for i, c in enumerate(CLASSES):
        support = sum(m[i])
        tp = m[i][i]
        pred_i = sum(m[r][i] for r in range(len(CLASSES)))
        recall = tp / support if support else None
        precision = tp / pred_i if pred_i else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)
        per_class[c] = {"support": support, "correct": tp,
                        "recall": None if recall is None else round(recall, 4),
                        "precision": None if precision is None else round(precision, 4),
                        "f1": round(f1, 4)}

    out = {
        "n": n,
        "accuracy_numerator": correct,
        "accuracy_denominator": n,
        "accuracy": round(correct / n, 4) if n else None,
        "undecided": undecided,
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "confusion_rows_true_cols_pred": m,
        "class_order": list(CLASSES),
    }
    if signers is not None:
        by = {}
        for s, t, p in zip(signers, y_true, y_pred):
            d = by.setdefault(str(s), {"n": 0, "correct": 0})
            d["n"] += 1
            d["correct"] += int(p is not None and t == p)
        out["per_signer"] = dict(sorted(by.items()))
        out["signer_counts"] = dict(sorted(Counter(map(str, signers)).items()))
    return out
