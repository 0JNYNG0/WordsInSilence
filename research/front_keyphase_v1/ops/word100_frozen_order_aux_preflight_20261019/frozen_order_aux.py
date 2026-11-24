"""Research-only frozen T3-F + GRU16. No trainer or web integration.

Real missing frames stay in the sequence. Only trailing batch padding is
excluded. The auxiliary branch consumes contextual TCN features; the whole
model is not a causal/streaming model because the frozen TCN is centered.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
sys.path.insert(0, str(ROOT / "research/sign_model_pilot_v1/src"))
sys.path.insert(0, str(UNIT.parent / "word100_frozen_c_temporal_20261019"))
from frozen_temporal import FrozenCTemporal
from temporal import phase_weights, summarize


class FrozenOrderAux(nn.Module):
    model_type = "wis_frozen_t3f_order_aux"
    model_version = "gru16_zero_residual_preflight_v1"

    def __init__(self, model_config, base_state):
        super().__init__()
        if model_config["width"] != 32 or model_config["n_classes"] != 100:
            raise ValueError("This frozen candidate requires width32/classes100")
        self.base = FrozenCTemporal(**model_config)
        self.base.load_state_dict(base_state, strict=True)
        self.base.requires_grad_(False)  # includes the previously trained phase head
        self.memory = nn.GRU(32, 16, num_layers=1, batch_first=True)
        self.residual = nn.Linear(16, 100, bias=False)
        nn.init.zeros_(self.residual.weight)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # includes BN buffers and dropout, even while the new part trains
        return self

    def assert_frozen(self):
        if any(p.requires_grad for p in self.base.parameters()):
            raise RuntimeError("Frozen T3-F has a trainable parameter")
        if any(m.training for m in self.base.modules()):
            raise RuntimeError("Frozen T3-F must stay in eval mode")

    @staticmethod
    def validate(x, pad_mask):
        if x.ndim != 3 or x.shape[-1] != 280 or pad_mask.shape != x.shape[:2]:
            raise ValueError("Expected x[B,T,280] and pad_mask[B,T]")
        if x.shape[0] == 0 or x.shape[1] == 0 or not x.is_floating_point():
            raise ValueError("Nonempty floating inputs required")
        if x.device != pad_mask.device or not torch.isfinite(x).all():
            raise ValueError("Finite inputs and same-device padding mask required")
        if not ((pad_mask == 0) | (pad_mask == 1)).all():
            raise ValueError("Binary padding mask required")
        lengths = pad_mask.long().sum(1)
        expected = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        if (lengths == 0).any() or not torch.equal(pad_mask.bool(), expected):
            raise ValueError("Real frames must be a nonempty prefix; missing hands are NOT padding")
        if (x[:, 0, 279] != 0).any() or ((x[:, 1:, 279] <= 0) & expected[:, 1:]).any():
            raise ValueError("Keep real chronology: first dt=0 and later real dt>0")
        return lengths

    def remember(self, features, lengths):
        """No persistent hidden state between clips; h_n excludes trailing padding."""
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.memory(packed)  # h_0 defaults to zero on every call
        return last[-1]

    def components(self, x, pad_mask):
        lengths = self.validate(x, pad_mask)
        self.assert_frozen()
        pad_mask = pad_mask.to(x.dtype)
        # Exact frozen_temporal.py arithmetic, shared once to expose pre-mean features.
        # Do not call the old assert_frozen(): its old contract requires phase_weight
        # trainable, whereas this candidate deliberately freezes it too.
        with torch.no_grad():
            mask = pad_mask.unsqueeze(1)
            h = self.base.proj(x.transpose(1, 2) * mask) * mask
            for block in self.base.blocks:
                h = block(h, mask)
            g, contrast = summarize(h, pad_mask, phase_weights(x, pad_mask))
            logits = self.base.head(g) + torch.nn.functional.linear(contrast, self.base.phase_weight)
        memory = self.remember(h.transpose(1, 2), lengths)
        delta = self.residual(memory)
        return logits, delta, memory, h.transpose(1, 2), lengths

    def forward(self, x, pad_mask):
        base, delta, _, _, _ = self.components(x, pad_mask)
        return base + delta
