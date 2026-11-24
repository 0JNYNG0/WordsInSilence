"""Pure9-sample statistics, not a learned or semantic direction classifier."""
import numpy as np

T = np.linspace(0., 1., 9)
PAIR_I, PAIR_J = np.triu_indices(9, 1)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape[-1] == 9 and np.isfinite(values).all()
    differences = np.diff(values, axis=-1)
    slopes = (values[..., PAIR_J] - values[..., PAIR_I]) / (T[PAIR_J] - T[PAIR_I])
    signs = np.sign(differences)
    return dict(start=values[..., 0], end=values[..., -1], net=values[..., -1]-values[..., 0],
        trend=np.median(slopes, axis=-1),
        positive_path=np.maximum(differences, 0).sum(-1),
        negative_path=np.maximum(-differences, 0).sum(-1),
        path_variation=np.abs(differences).sum(-1), range=np.ptp(values, axis=-1),
        adjacent_sign_switches=((signs[..., 1:] * signs[..., :-1]) < 0).sum(-1),
        positive_steps=(differences > 0).sum(-1), negative_steps=(differences < 0).sum(-1))


def reverse_samples(values):
    values = np.asarray(values)
    assert values.shape[-1] == 9
    return values[..., ::-1].copy()
