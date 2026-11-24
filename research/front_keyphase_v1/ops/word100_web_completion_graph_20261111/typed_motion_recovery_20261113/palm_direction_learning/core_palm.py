"""Isolated axis2 palm direction head; no model is loaded on import.

Direction scores obey the observed 2D endpoint sign. The learned hold score
sees the trajectory, not word identity, expected direction, or hand context.
This is not a contact, completion, or full-path semantic classifier.
"""
from pathlib import Path
import copy
import sys
import time
import traceback

UNIT = Path(__file__).resolve().parent
sys.path.insert(0, str(UNIT.parent))
from common import HERE, GRAPH, OPS, PRIOR, old, np, torch, read, write, sha, stamp

nn = torch.nn
D = HERE / 'partial_direction_contract'
N = HERE / 'null_candidate_learning'
MEASUREMENT_SHA256 = 'ddcaee0652846059b6d2f9c83371431dc304ee282aea4ba158ae0b1de76b835a'
SCHEMA = 'palm58_sign_constrained_symmetric_trajectory_hold_v1'


def stop_if_requested():
    for folder in (UNIT, HERE, GRAPH):
        for name in ('STOP_REQUESTED', 'STOP_REQUESTED.json'):
            assert not (folder / name).exists(), f'Pause marker: {folder / name}'


def preserve_failure(stage):
    UNIT.mkdir(parents=True, exist_ok=True)
    message = stage + '\n' + traceback.format_exc()
    with (UNIT / f'{stage}_FAILURE_{time.time_ns()}.txt').open('x', encoding='utf-8') as f:
        f.write(message)
    first = UNIT / 'FIRST_FAILURE.txt'
    if not first.exists():
        with first.open('x', encoding='utf-8') as f:
            f.write(message)


def reverse_geometry(g):
    """Reverse the raw 9-point trajectory before normalization, tensor version.

    Relative samples, absolute start, endpoint delta change. Absolute step
    means, ranges, axis one-hot and coverage remain invariant. No neural call.
    """
    assert g.ndim == 2 and g.shape[1] == 58
    out = g.clone()
    trajectory = g[:, :36].reshape(-1, 9, 4)
    out[:, :36] = (trajectory.flip(1) - trajectory[:, -1:]).reshape(-1, 36)
    out[:, 36:40] = g[:, 36:40] + g[:, 40:44]
    out[:, 40:44] = -g[:, 40:44]
    return out


def reverse_geometry_numpy(g):
    assert g.shape == (58,)
    out = g.copy()
    trajectory = g[:36].reshape(9, 4)
    out[:36] = (trajectory[::-1] - trajectory[-1]).reshape(36)
    out[36:40] = g[36:40] + g[40:44]
    out[40:44] = -g[40:44]
    return out


def static_geometry(g, anchor):
    """Constant measured geometry at sample 0,4,8; not synthetic RGB/pose truth.

    An unavailable source must remain unavailable via the separate mask.
    The centre sample may have been interpolated by the measurement helper.
    """
    assert g.shape == (58,) and anchor in (0, 4, 8)
    out = g.copy()
    out[36:40] = g[36:40] + g[:36].reshape(9, 4)[anchor]
    out[:36] = 0
    out[40:52] = 0
    return out


class PalmDirection(nn.Module):
    def __init__(self, observer, mean, std, delta_scale):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        assert mean.shape == std.shape == (58,) and bool((std > 0).all())
        assert torch.isfinite(mean).all() and torch.isfinite(std).all()
        assert np.isfinite(delta_scale) and delta_scale > 0
        self.register_buffer('gm', mean)
        self.register_buffer('gs', std)
        self.register_buffer('delta_scale', torch.tensor(float(delta_scale), dtype=torch.float32))
        self.motion_conv = copy.deepcopy(observer.motion_conv)
        self.motion_gru = copy.deepcopy(observer.motion_gru)
        self.motion_head = copy.deepcopy(observer.motion_head)
        # softplus(raw_scale)==1 initially, a fixed initialization, not selection.
        self.raw_scale = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.))), dtype=torch.float32))
        self.requires_grad_(True)

    def _hold_score(self, raw_g):
        z = (raw_g - self.gm) / self.gs
        seq = z[:, :36].reshape(-1, 9, 4)
        seq = self.motion_conv(seq.transpose(1, 2)).transpose(1, 2)
        _, h = self.motion_gru(seq)
        # Exactly 192 zero context values: no common100 image/hand features.
        zero_context = z.new_zeros((len(z), 192))
        previous = self.motion_head(torch.cat([h[0], h[1], z, zero_context], -1))
        return previous[:, 1] - (previous[:, 0] + previous[:, 2]) / 2

    def forward(self, g):
        assert g.ndim == 2 and g.shape[1] == 58
        # Each time direction is normalized separately AFTER raw reversal.
        hold = (self._hold_score(g) + self._hold_score(reverse_geometry(g))) / 2
        signed = torch.nn.functional.softplus(self.raw_scale) * (g[:, 40] / self.delta_scale)
        return torch.stack([-signed, hold, signed], -1)


def partial_progression_loss(logits, reduction='mean'):
    """No partial direction target; teach only progression vs hold.

    Detach directional mass so partial supervision cannot simply inflate the
    positive scale or silently assign a direction label. This updates the
    trajectory hold scorer; full directional CE trains the directional scale.
    """
    direction_mass = torch.logsumexp(logits[:, [0, 2]], -1).detach()
    loss = torch.nn.functional.softplus(logits[:, 1] - direction_mass)
    if reduction == 'none':
        return loss
    assert reduction == 'mean'
    return loss.mean()


def load_inputs():
    meta = read(UNIT / 'INPUTS_COMPLETE.json')
    assert sha(meta['path']) == meta['sha256']
    for path, digest in meta['pins'].items():
        assert sha(path) == digest, path
    with np.load(meta['path'], allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    return meta, arrays
