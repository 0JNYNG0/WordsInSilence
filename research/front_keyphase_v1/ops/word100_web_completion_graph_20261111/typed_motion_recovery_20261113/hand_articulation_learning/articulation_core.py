"""Pure projected hand articulation, independent of word and image-plane rotation.

This is a deterministic feature transform, not new anatomical ground truth.
Input hand63 is the already canonicalized R coordinate stream, but these local
angles/ratios also work on untranslated/rotated/scaled planar hand coordinates.
The four non-thumb fingers contribute PIP cosine, DIP cosine and tip-to-MCP
distance divided by their three projected bone lengths. Unknown geometry has
value zero AND mask zero; a zero cosine or ratio with mask one is valid data.
"""
import numpy as np

EPSILON = 1e-6  # Numerical degeneracy only; never a hand-shape threshold.
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FEATURE_NAMES = [f'{finger}_{metric}' for finger in ('index', 'middle', 'ring', 'little')
                 for metric in ('PIP_cosine', 'DIP_cosine', 'tip_MCP_chain_ratio')]


def articulation24(x):
    """Return [T,24]: 12 projected scalars followed by 12 availability masks.

    Cosine uses the two vectors pointing AWAY from the joint, so a straight
    projected finger has cosine -1. No angle cutoff is used for labels/verdicts.
    A point mask <=.5 is unavailable, exactly as in the frozen source contract.
    """
    a = np.asarray(x)
    if a.ndim != 2 or a.shape[1] != 63 or not np.isfinite(a).all():
        raise ValueError('Expected finite [frames,63] original hand coordinates and masks')
    if ((a[:, 42:] < 0) | (a[:, 42:] > 1)).any():
        raise ValueError('Availability must lie in [0,1]')
    p = a[:, :42].reshape(-1, 21, 2).astype(np.float64)
    observed = a[:, 42:] > .5
    values = np.zeros((len(a), 12), np.float64)
    masks = np.zeros((len(a), 12), bool)
    for f, joints in enumerate(FINGERS):
        for k, triple in enumerate((joints[:3], joints[1:])):
            u, v = p[:, triple[0]] - p[:, triple[1]], p[:, triple[2]] - p[:, triple[1]]
            nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
            ok = observed[:, triple].all(axis=1) & (nu > EPSILON) & (nv > EPSILON)
            index = 3 * f + k
            values[ok, index] = np.clip((u[ok] * v[ok]).sum(axis=1) / (nu[ok] * nv[ok]), -1, 1)
            masks[:, index] = ok
        bones = p[:, joints[1:]] - p[:, joints[:-1]]
        lengths = np.linalg.norm(bones, axis=2)
        ok = observed[:, joints].all(axis=1) & (lengths > EPSILON).all(axis=1)
        index = 3 * f + 2
        values[ok, index] = np.linalg.norm(p[ok, joints[3]] - p[ok, joints[0]], axis=1) / lengths[ok].sum(axis=1)
        # Triangle inequality bounds this ratio; clipping removes floating error.
        values[ok, index] = np.clip(values[ok, index], 0, 1)
        masks[:, index] = ok
    return np.c_[values, masks.astype(np.float64)].astype(np.float32)


def fit_articulation_stats(arrays):
    """Caller supplies ONLY training-person sequences, one original copy each.

    Availability masks stay unnormalized. Missing values never enter statistics.
    The .05 floor matches the original hand-coordinate standardization recipe;
    it is numerical scaling, not a semantic decision threshold.
    """
    a = np.concatenate(arrays, axis=0)
    if a.ndim != 2 or a.shape[1] != 24:
        raise ValueError('Expected articulation24 arrays')
    mean, std, counts = np.zeros(24, np.float32), np.ones(24, np.float32), []
    for j in range(12):
        v = a[a[:, 12 + j] > .5, j]
        if not len(v):
            raise ValueError(f'No training observation for feature {FEATURE_NAMES[j]}')
        mean[j], std[j] = v.mean(), max(float(v.std()), .05)
        counts.append(int(len(v)))
    return mean, std, counts


def normalize_articulation_numpy(a, mean, std):
    """Test/runtime counterpart of the model's explicitly masked normalization."""
    a = np.asarray(a, np.float32)
    out = (a - mean) / std
    out[..., :12] *= a[..., 12:] > .5
    return out
