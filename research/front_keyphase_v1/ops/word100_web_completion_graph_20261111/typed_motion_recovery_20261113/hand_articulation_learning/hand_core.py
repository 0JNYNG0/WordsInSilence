"""Add a zero-initialized projected-articulation residual to a frozen-parent copy.

The R branch's 63 input values and every old weight/buffer are copied exactly at
initialization. Fine-tuning is permitted only after root review, on the same
person-excluding FIT splits and existing labels. No deployed module is changed.
"""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
PARENT = U.parent / 'hand_rotation_invariant_learning'
ORIGINAL = U.parent / 'hand_transition_learning'
spec = importlib.util.spec_from_file_location('_articulation_original_core', ORIGINAL / 'hand_core.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
for name in ['np', 'torch', 'nn', 'read', 'write', 'sha', 'stamp']:
    globals()[name] = getattr(original, name)
ATTRIBUTES = list(original.ATTRIBUTES)


class HandObserver(original.HandObserver):
    def __init__(self, mean, std, articulation_mean, articulation_std):
        super().__init__(mean, std)
        self.register_buffer('articulation_mean', torch.as_tensor(articulation_mean, dtype=torch.float32))
        self.register_buffer('articulation_std', torch.as_tensor(articulation_std, dtype=torch.float32))
        # No bias: no all-missing articulation-specific offset can be learned.
        self.articulation_residual = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.articulation_residual.weight)

    def forward(self, x, lengths):
        if x.shape[-1] != 87:
            raise ValueError('Expected unchanged hand63 followed by articulation24')
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        base, extra = x[..., :63], x[..., 63:]
        norm = (extra - self.articulation_mean) / self.articulation_std
        norm = torch.cat([norm[..., :12] * (extra[..., 12:] > .5), norm[..., 12:]], -1)
        h = self.spatial[1](self.spatial[0]((base - self.mean) / self.std) + self.articulation_residual(norm)) * valid[:, :, None]
        h = h.transpose(1, 2)
        for layer in self.temporal:
            h = layer(h) * valid[:, None]
        pack = nn.utils.rnn.pack_padded_sequence(h.transpose(1, 2), lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.context(pack)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        first = h[:, 0]
        last = h[torch.arange(len(h)), lengths - 1]
        avg = (h * valid[:, :, None]).sum(1) / lengths[:, None]
        return dict(attributes=self.attributes(h), transition=self.transition(torch.cat([first, avg, last, last-first], -1)), latent=h, valid=valid)


def load_parent_weights(model, parent_state):
    """Copy exactly the matching R fold/seed; verify every inherited tensor."""
    status = model.load_state_dict(parent_state, strict=False)
    expected = {'articulation_mean', 'articulation_std', 'articulation_residual.weight'}
    assert set(status.missing_keys) == expected and not status.unexpected_keys
    state = model.state_dict()
    assert all(torch.equal(state[k], value) for k, value in parent_state.items())
    assert torch.count_nonzero(model.articulation_residual.weight).item() == 0


def padded(sequences):
    lens = np.asarray([len(x) for x in sequences], np.int64)
    out = np.zeros((len(lens), int(lens.max()), 87), np.float32)
    for i, x in enumerate(sequences):
        out[i, :len(x)] = x
    return torch.tensor(out), torch.tensor(lens)


def preserve_failure(stage):
    import traceback, time
    U.mkdir(exist_ok=True, parents=True)
    (U / f'{stage}_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(), encoding='utf-8')
