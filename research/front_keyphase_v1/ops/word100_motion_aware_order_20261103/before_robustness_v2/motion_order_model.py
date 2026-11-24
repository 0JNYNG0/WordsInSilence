"""Matched frozen-Y readouts with optional frozen AH occurrence features.

Importing this module loads no checkpoint and runs no previous experiment.
Only the shared query and zero-initialized auxiliary projection are trainable.
Metadata, source identity, edit kind and labels are not model inputs.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
Y = OPS / 'word100_direct_occurrence_order_20261102'
INITIAL_Y = Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102/runs/tied_readout/model.pt')
CONDITIONS = ('matched_readout_only', 'matched_readout_plus_AH')
TRAINABLE_NAMES = {'readout.query.weight', 'auxiliary.weight'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def legacy():
    """Definitions only; none of the old training/evaluation mains are called."""
    if str(Y) not in sys.path:
        sys.path.insert(0, str(Y))
    module = importlib.import_module('readout_model')
    if Path(module.__file__).resolve() != (Y / 'readout_model.py').resolve():
        raise RuntimeError('Conflicting readout_model import')
    return module


def initial_y_sha():
    runs = read(Y / 'TRAINING_RESULT.json')['runs']
    row = [r for r in runs if r['condition'] == 'tied_readout']
    assert len(row) == 1
    return row[0]['checkpoint_sha256']


class MotionOrderNet(nn.Module):
    def __init__(self, condition, *, tied_model=None, aliases=None):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError('Unknown matched condition')
        self.condition = condition
        if tied_model is None:
            old = legacy()
            assert sha(INITIAL_Y) == initial_y_sha(), 'Y checkpoint changed'
            tied_model = old.ReadoutNet('tied_readout')
            tied_model.load_state_dict(torch.load(INITIAL_Y, map_location='cpu', weights_only=True), strict=True)
            aliases = old.episode.ALIASES
        if aliases is None or getattr(tied_model, 'condition', None) != 'tied_readout':
            raise ValueError('A tied readout and its family aliases are required')
        self.readout = deepcopy(tied_model)
        self.aliases = tuple(tuple(int(v) for v in ids) for ids in aliases)
        assert len(self.aliases) == 8
        assert tuple(self.readout.query.weight.shape) == (32, 48)
        assert self.readout.query.bias is None
        assert tuple(self.readout.base.family_guide_keys.shape) == (8, 32)
        for p in self.readout.parameters():
            p.requires_grad_(False)
        self.readout.query.weight.requires_grad_(True)
        self.auxiliary = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.auxiliary.weight)
        self.readout.eval()
        assert {n for n, p in self.named_parameters() if p.requires_grad} == TRAINABLE_NAMES
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == 1792

    def train(self, mode=True):
        super().train(mode)
        # All context encoders and all base buffers keep inference behavior.
        self.readout.eval()
        return self

    def auxiliary_input(self, aux, lengths):
        if aux.ndim != 3 or aux.shape[-1] != 8 or len(lengths) != len(aux):
            raise ValueError('Auxiliary input must be batch x native rows x 8')
        if not torch.isfinite(aux).all():
            raise ValueError('Non-finite auxiliary input')
        valid = torch.arange(aux.shape[1], device=aux.device)[None, :] < lengths.to(aux.device)[:, None]
        a = aux[valid]
        if not torch.all((a[:, 5:] == 0) | (a[:, 5:] == 1)):
            raise ValueError('Availability channels must be explicit binary masks')
        if not torch.all((a[:, :5] >= -1) & (a[:, :5] <= 1)):
            raise ValueError('Expected centered AH probabilities')
        if not torch.all(a[a[:, 5] == 0, :5] == 0):
            raise ValueError('Unavailable AH scores must be masked, not negative evidence')
        out = aux.clone()
        out[~valid] = 0
        if self.condition == CONDITIONS[0]:
            out[:, :, :5] = 0
        return out

    def forward(self, x, rgb, lengths, flow, aux):
        if x.shape[:2] != rgb.shape[:2] or x.shape[:2] != flow.shape[:2] or x.shape[:2] != aux.shape[:2]:
            raise ValueError('Native input axes differ')
        if lengths.ndim != 1 or torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError('Invalid native lengths')
        auxiliary = self.auxiliary_input(aux, lengths)
        r = self.readout
        # This is the existing tied forward, with all feature/context/physical
        # operations fixed. Only the two projections below carry gradients.
        with torch.no_grad():
            feature, h = r.base.encode(x, rgb, lengths)
            logits = r.base.actor.query(feature) @ (r.base.actor.guide(r.base.actor.guide_vectors) + r.base.actor.embedding).T / (32 ** .5)
            families = torch.stack([torch.logsumexp(logits[:, :, ids], -1) - np.log(len(ids)) for ids in self.aliases], -1)
            action = torch.cat([families, r.base.actor.rest(feature)], -1)
            visual = torch.cat([feature, h, x[:, :, 64:74]], -1)
            fixed = r.base.role_embed(visual + r.base.flow_bridge(flow))
            fixed, _ = r.base.role_context(nn.utils.rnn.pack_padded_sequence(fixed, lengths.cpu(), batch_first=True, enforce_sorted=False))
            fixed, _ = nn.utils.rnn.pad_packed_sequence(fixed, batch_first=True, total_length=x.shape[1])
            role = r.base.role_output(fixed)
            z = r.visual_embed(visual)
            z, _ = r.visual_context(nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False))
            z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=x.shape[1])
        occurrence = (r.query(z) + self.auxiliary(auxiliary)) @ r.base.family_guide_keys.T / (32 ** .5)
        occurrence = occurrence + (role[:, :, 1] - role[:, :, 0])[:, :, None]
        return action, role, occurrence

    def frozen_state(self):
        return {n: v.detach().cpu().clone() for n, v in self.state_dict().items() if n not in TRAINABLE_NAMES}

    def assert_frozen(self, snapshot):
        current = self.state_dict()
        assert set(snapshot) == set(current) - TRAINABLE_NAMES
        for n, v in snapshot.items():
            assert torch.equal(current[n].detach().cpu(), v), 'Frozen state changed: ' + n


def auxiliary_batch(rows, device='cpu'):
    n = max(len(r['x']) for r in rows)
    result = np.zeros((len(rows), n, 8), dtype=np.float32)
    for i, r in enumerate(rows):
        a = np.asarray(r['aux'])
        if a.dtype != np.float32 or a.shape != (len(r['x']), 8) or not np.isfinite(a).all():
            raise ValueError('Unprepared auxiliary stream')
        result[i, :len(a)] = a
    return torch.from_numpy(result).to(device)


def grouped_mean(row_values, source_ids):
    if row_values.ndim != 1 or len(row_values) != len(source_ids):
        raise ValueError('One loss per source variant required')
    groups = {}
    for i, key in enumerate(source_ids):
        groups.setdefault(key, []).append(i)
    return torch.stack([row_values[indices].mean() for indices in groups.values()]).mean()


def occurrence_group_loss(logits, targets, positive_weights, source_ids):
    known = targets >= 0
    values = nn.functional.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), pos_weight=positive_weights, reduction='none')
    row_loss = (values * known).sum((1, 2)) / known.sum((1, 2)).clamp_min(1)
    return grouped_mean(row_loss, source_ids)


def pair_group_loss(physical, occurrence, rows):
    old = legacy()
    values = []
    used = 0
    for i, row in enumerate(rows):
        value, count = old.pair_loss(physical[i:i+1], occurrence[i:i+1], [row], 'tied_readout')
        values.append(value)
        used += count
    return grouped_mean(torch.stack(values), [r['source_group_id'] for r in rows]), used
