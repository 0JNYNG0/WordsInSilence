"""Isolated quality copies; no word, desired direction or source-index input."""
from pathlib import Path
import json
import torch
from torch import nn


class LocalMotionQuality(nn.Module):
    """B=existing summary MLP; C=the same MLP plus raw-sample Conv/GRU.

    sequence[..., :4]: body-relative values, 4:8: adjacent observed deltas,
    8: observed, 9: observed adjacent pair, 10: elapsed seconds (first zero).
    Returns (extent_logits[B,4], edit_aux_logits[B]). Neither is sign correctness.
    """
    def __init__(self, condition='B', seed=42, ops_root=None):
        super().__init__()
        if condition not in ('B', 'C'):
            raise ValueError('condition must be B or C')
        self.condition = condition
        root = Path(ops_root) if ops_root else Path(__file__).resolve().parents[2]
        source = root/'word100_candidate_role_learning_20261111/runs'/str(seed)/'model.pt'
        local = root/'word100_autonomous_localization_20261111'
        ck = torch.load(source, map_location='cpu', weights_only=True)
        norm = json.loads((local/'NORMALIZATION.json').read_text(encoding='utf-8-sig'))
        self.register_buffer('relation_mean', torch.tensor(norm['mean'][:4], dtype=torch.float32))
        self.register_buffer('relation_std', torch.tensor(norm['std'][:4], dtype=torch.float32))
        if not bool((self.relation_std > 0).all()):
            raise ValueError('invalid source normalization')
        width = 304 if condition == 'B' else 400
        self.summary = nn.Linear(width, 64)
        self.extent = nn.Linear(64, 4)
        with torch.no_grad():
            self.summary.weight.zero_()
            self.summary.weight[:, :304].copy_(ck['quality.0.weight'])
            self.summary.bias.copy_(ck['quality.0.bias'])
            self.extent.weight.copy_(ck['quality.2.weight'])
            self.extent.bias.copy_(ck['quality.2.bias'])
        # Identical auxiliary initialization across B/C, irrespective of the
        # different number of tensors constructed for their temporal branches.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7919)
            self.edit_aux = nn.Linear(64, 1)
        if condition == 'C':
            local_ck = torch.load(local/'runs'/str(seed)/'model.pt', map_location='cpu', weights_only=True)
            self.conv = nn.Sequential(nn.Conv1d(4, 16, 3, padding=1), nn.GELU())
            self.gru = nn.GRU(16, 16, batch_first=True, bidirectional=True)
            self.conv.load_state_dict({k[5:]:v for k,v in local_ck.items() if k.startswith('conv.')})
            self.gru.load_state_dict({k[4:]:v for k,v in local_ck.items() if k.startswith('gru.')})
            self.observation_projection = nn.Linear(7, 16, bias=False)
            nn.init.zeros_(self.observation_projection.weight)
        self.source_paths = dict(quality=str(source), localizer=str(local/'runs'/str(seed)/'model.pt'))

    def forward(self, summary, sequence, lengths):
        if summary.ndim != 2 or summary.shape[1] != 304:
            raise ValueError('summary must be [B,304]')
        if sequence.ndim != 3 or sequence.shape[0] != len(summary) or sequence.shape[2] != 11:
            raise ValueError('sequence must be [B,T,11]')
        if lengths.shape != (len(summary),) or bool((lengths < 1).any()) or bool((lengths > sequence.shape[1]).any()):
            raise ValueError('invalid sequence lengths')
        if not bool(torch.isfinite(summary).all()) or not bool(torch.isfinite(sequence).all()):
            raise ValueError('nonfinite model input')
        z = summary
        if self.condition == 'C':
            device_lengths = lengths.to(sequence.device)
            valid_time = torch.arange(sequence.shape[1], device=sequence.device)[None] < device_lengths[:, None]
            observed = sequence[..., 8] * valid_time
            pair_observed = sequence[..., 9] * valid_time
            pos = ((sequence[..., :4] - self.relation_mean) / self.relation_std).clamp(-12, 12)
            pos = pos * observed[..., None]
            delta = sequence[..., 4:8] / self.relation_std
            delta = delta * pair_observed[..., None]
            aux = torch.cat([delta, observed[..., None], pair_observed[..., None], torch.log1p(sequence[..., 10:11].clamp_min(0))], -1)
            local = self.conv(pos.transpose(1,2)).transpose(1,2)
            local = (local + self.observation_projection(aux)) * valid_time[..., None]
            packed = nn.utils.rnn.pack_padded_sequence(local, lengths.cpu(), batch_first=True, enforce_sorted=False)
            hidden, _ = self.gru(packed)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=sequence.shape[1])
            mean = (hidden * observed[..., None]).sum(1) / observed.sum(1).clamp_min(1)[:, None]
            last = hidden[torch.arange(len(hidden), device=hidden.device), device_lengths - 1]
            z = torch.cat([summary, mean, hidden[:, 0], last], -1)
        hidden = nn.functional.gelu(self.summary(z))
        return self.extent(hidden), self.edit_aux(hidden).squeeze(-1)


def make_sequence(values, observed, timestamps_us):
    """Pure array packing. Missing samples never become observed zero poses."""
    import numpy as np
    v = np.asarray(values, dtype=np.float32)
    mask = np.asarray(observed)
    if mask.dtype.kind != 'b':
        raise ValueError('observed must be an explicit boolean mask')
    good = mask.copy()
    ts = np.asarray(timestamps_us)
    if v.ndim != 2 or v.shape[1] != 4 or good.shape != (len(v),) or ts.shape != (len(v),) or len(v) == 0:
        raise ValueError('expected values[T,4], observed[T], timestamps[T]')
    if not np.isfinite(ts).all() or np.any(ts[1:] <= ts[:-1]):
        raise ValueError('timestamps must increase')
    good &= np.isfinite(v).all(1)
    values_clean = np.where(good[:, None], v, 0)
    pair = np.zeros(len(v), bool)
    pair[1:] = good[1:] & good[:-1]
    delta = np.zeros_like(v)
    delta[1:] = np.where(pair[1:, None], values_clean[1:] - values_clean[:-1], 0)
    dt = np.zeros(len(v), np.float32)
    dt[1:] = np.diff(ts).astype(np.float64) / 1e6
    return np.c_[values_clean, delta, good.astype(np.float32), pair.astype(np.float32), dt].astype(np.float32)


def raw_sequence(values, observed, timestamps_us, a=0, b=None):
    """Inclusive crop; first delta/pair-valid/dt are zero, no prefix borrowed."""
    import numpy as np
    v,good,ts=np.asarray(values),np.asarray(observed),np.asarray(timestamps_us)
    b=len(v)-1 if b is None else b
    if not isinstance(a,(int,np.integer)) or not isinstance(b,(int,np.integer)) or not 0<=a<=b<len(v):
        raise ValueError('invalid inclusive raw sequence crop')
    return make_sequence(v[a:b+1],good[a:b+1],ts[a:b+1])
