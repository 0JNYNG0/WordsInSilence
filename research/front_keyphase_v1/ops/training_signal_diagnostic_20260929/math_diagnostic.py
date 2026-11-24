import math
import numpy as np


def _arr(x, name, ndim=None, shape=None, dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {a.ndim}")
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {a.shape}")
    if dtype is np.float64 and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    s = Z - m
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def analyze(W, b, W0, b0, groups, teacher_X, teacher_Z, teacher_weights,
            anchor_coefficient, output_coefficient, displacement=None):
    W = _arr(W, "W", 2)
    C, D = W.shape
    if C < 1 or D < 1:
        raise ValueError("W must be non-empty")
    b = _arr(b, "b", 1, (C,))
    W0 = _arr(W0, "W0", 2, (C, D))
    b0 = _arr(b0, "b0", 1, (C,))
    ac = float(anchor_coefficient)
    oc = float(output_coefficient)
    if not (math.isfinite(ac) and math.isfinite(oc)):
        raise ValueError("coefficients must be finite")
    P = C * D + C
    if displacement is not None:
        displacement = _arr(displacement, "displacement", 1, (P,))
    if not isinstance(groups, dict):
        raise ValueError("groups must be a dict")

    flat = lambda gW, gb: np.concatenate([gW.ravel(), gb.ravel()])
    grads, result_groups, losses = {}, {}, {}

    for name, g in groups.items():
        key = str(name)
        X = _arr(g["X"], f"{key}.X", 2)
        N = X.shape[0]
        if X.shape[1] != D:
            raise ValueError(f"{key}.X must have {D} columns")
        y_raw = np.asarray(g["y"])
        if y_raw.shape != (N,):
            raise ValueError(f"{key}.y must have shape ({N},)")
        if N and not (np.issubdtype(y_raw.dtype, np.integer) or
                      (np.all(np.isfinite(y_raw.astype(np.float64))) and
                       np.all(y_raw == np.round(y_raw)))):
            raise ValueError(f"{key}.y must be integer labels")
        y = y_raw.astype(np.int64)
        if N and (y.min() < 0 or y.max() >= C):
            raise ValueError(f"{key}.y labels out of range [0,{C})")
        w = _arr(g["weights"], f"{key}.weights", 1, (N,))
        if np.any(w < 0):
            raise ValueError(f"{key}.weights must be nonnegative")

        Z = X @ W.T + b
        lp = _log_softmax(Z)
        idx = np.arange(N)
        nll = -lp[idx, y]
        loss = float(np.sum(w * nll))
        dZ = np.exp(lp)
        dZ[idx, y] -= 1.0
        dZ *= w[:, None]
        grads["group:" + key] = flat(dZ.T @ X, dZ.sum(axis=0))

        wsum = float(w.sum())
        correct_mask = Z.argmax(axis=1) == y if N else np.zeros(0, bool)
        counts = np.bincount(y, minlength=C)
        corr = np.bincount(y[correct_mask], minlength=C)
        present = counts > 0
        result_groups[key] = {
            "loss": _f(loss),
            "weight_sum": _f(wsum),
            "weighted_mean_ce": _f(loss / wsum) if wsum > 0 else None,
            "accuracy": _f(correct_mask.sum() / N) if N else None,
            "count": int(N),
            "by_class": {str(c): {"count": int(counts[c]), "correct": int(corr[c]),
                                  "recall": _f(corr[c] / counts[c]) if counts[c] else None}
                         for c in range(C)},
            "macro_recall": _f(np.mean(corr[present] / counts[present])) if present.any() else None,
        }
        losses["group:" + key] = loss

    dW_a, db_a = W - W0, b - b0
    a_loss = 0.5 * ac * (float(np.sum(dW_a ** 2)) + float(np.sum(db_a ** 2)))
    grads["anchor"] = ac * flat(dW_a, db_a)
    losses["anchor"] = a_loss

    TX = _arr(teacher_X, "teacher_X", 2)
    M = TX.shape[0]
    if TX.shape[1] != D:
        raise ValueError(f"teacher_X must have {D} columns")
    TZ = _arr(teacher_Z, "teacher_Z", 2, (M, C))
    tw = _arr(teacher_weights, "teacher_weights", 1, (M,))
    if np.any(tw < 0):
        raise ValueError("teacher_weights must be nonnegative")
    if M and not np.isclose(tw.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("teacher_weights must sum to 1")
    R = TX @ W.T + b - TZ
    Rc = R - R.mean(axis=1, keepdims=True)
    o_loss = 0.5 * oc * float(np.sum(tw * np.mean(Rc ** 2, axis=1))) if M else 0.0
    dZo = oc * tw[:, None] * Rc / C
    grads["output"] = flat(dZo.T @ TX, dZo.sum(axis=0))
    losses["output"] = o_loss

    names = list(grads)
    norms = {k: float(np.linalg.norm(grads[k])) for k in names}
    pairwise = {}
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            d = float(grads[a] @ grads[c])
            den = norms[a] * norms[c]
            pairwise[f"{a}|{c}"] = {"dot": _f(d), "cosine": _f(d / den) if den > 0 else None}
    total = np.sum([grads[k] for k in names], axis=0)
    grads["total"] = total

    result = {
        "num_classes": int(C), "num_features": int(D), "num_parameters": int(P),
        "groups": result_groups,
        "anchor": {"loss": _f(a_loss), "coefficient": _f(ac)},
        "output": {"loss": _f(o_loss), "coefficient": _f(oc), "count": int(M)},
        "losses": {k: _f(v) for k, v in losses.items()},
        "total_loss": _f(sum(losses.values())),
        "gradient_norms": {k: _f(v) for k, v in norms.items()},
        "pairwise": pairwise,
        "total_gradient_norm": _f(np.linalg.norm(total)),
    }
    if displacement is not None:
        result["displacement_norm"] = _f(np.linalg.norm(displacement))
        result["displacement_dot"] = {k: _f(grads[k] @ displacement) for k in names + ["total"]}
    return result, grads
