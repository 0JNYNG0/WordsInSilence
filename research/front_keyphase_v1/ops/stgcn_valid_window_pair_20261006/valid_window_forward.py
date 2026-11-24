# Masked window-sequence forward: window encoder -> ReLU -> GRU -> final_norm -> classifier.
import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ['forward_valid_windows', 'self_test']


def _check_module_device(module, name, device):
    for t in list(module.parameters()) + list(module.buffers()):
        if t.device != device:
            raise ValueError(f'{name} tensor on {t.device}, expected {device}')


def _check_gru(recurrent):
    if not isinstance(recurrent, nn.GRU):
        raise TypeError('recurrent must be torch.nn.GRU')
    if not recurrent.batch_first or recurrent.bidirectional or recurrent.num_layers != 1:
        raise ValueError('recurrent must be batch_first, unidirectional, num_layers=1')
    if recurrent.input_size != recurrent.hidden_size:
        raise ValueError('recurrent must have input_size == hidden_size')


def forward_valid_windows(encoder, recurrent, final_norm, classifier, blocks, window_valid):
    if not isinstance(blocks, Tensor) or not blocks.is_floating_point():
        raise TypeError('blocks must be a floating-point Tensor')
    if blocks.dim() != 5 or any(s <= 0 for s in blocks.shape):
        raise ValueError(f'blocks must be [B,N,C,T,V] with positive dims, got {tuple(blocks.shape)}')
    if not isinstance(window_valid, Tensor) or window_valid.dtype != torch.bool:
        raise TypeError('window_valid must be a bool Tensor')
    if tuple(window_valid.shape) != tuple(blocks.shape[:2]):
        raise ValueError(f'window_valid shape {tuple(window_valid.shape)} != {tuple(blocks.shape[:2])}')
    if window_valid.device != blocks.device:
        raise ValueError(f'window_valid on {window_valid.device}, blocks on {blocks.device}')
    _check_gru(recurrent)
    for m, name in ((encoder, 'encoder'), (recurrent, 'recurrent'),
                    (final_norm, 'final_norm'), (classifier, 'classifier')):
        if not isinstance(m, nn.Module):
            raise TypeError(f'{name} must be an nn.Module')
        _check_module_device(m, name, blocks.device)
    # Reject all non-finite input (including invalid windows); padding must be finite.
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError('blocks contains non-finite values')
    lengths = window_valid.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError('every sample needs at least one valid window')

    B, N, C, T, V = blocks.shape
    H = recurrent.hidden_size
    dev = blocks.device
    # Row-major nonzero: samples in input order, chronological within each sample.
    flat_idx = torch.nonzero(window_valid.reshape(B * N)).reshape(-1)
    M = flat_idx.numel()
    # Invalid windows are removed before the encoder (never seen by BatchNorm etc.).
    feats = encoder(blocks.flatten(0, 1).index_select(0, flat_idx))
    if not isinstance(feats, Tensor) or tuple(feats.shape) != (M, H):
        got = tuple(feats.shape) if isinstance(feats, Tensor) else type(feats)
        raise ValueError(f'encoder must return [{M},{H}], got {got}')
    feats = F.relu(feats)

    starts = torch.cumsum(lengths, 0) - lengths
    order_parts, ctx_parts = [], []
    for L in torch.unique(lengths).tolist():
        sids = torch.nonzero(lengths == L).reshape(-1)
        gather = starts.index_select(0, sids).unsqueeze(1) + torch.arange(L, device=dev)
        seq = feats.index_select(0, gather.reshape(-1)).reshape(sids.numel(), L, H)
        out, _ = recurrent(seq)  # no padded timesteps: every row has exactly L real features
        ctx_parts.append(out[:, -1, :])
        order_parts.append(sids)
    order = torch.cat(order_parts)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(B, device=dev)
    ctx = torch.cat(ctx_parts, 0).index_select(0, inverse)

    logits = classifier(final_norm(ctx))
    if not isinstance(logits, Tensor) or logits.dim() != 2 or logits.shape[0] != B:
        raise ValueError('classifier must return [B,K] logits')
    return logits


def _need(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f'expected {exc.__name__}')


def _self_test_body():
    dt = torch.float64
    B, N, C, T, V, H, K = 4, 5, 2, 3, 4, 6, 3
    base = (nn.Sequential(nn.Flatten(), nn.Linear(C * T * V, H), nn.BatchNorm1d(H)).to(dt),
            nn.GRU(H, H, batch_first=True).to(dt), nn.LayerNorm(H).to(dt), nn.Linear(H, K).to(dt))

    def fresh(eval_mode=False):
        ms = tuple(copy.deepcopy(m) for m in base)
        if eval_mode:
            for m in ms:
                m.eval()
        return ms

    def flags(ms):
        return [s.training for m in ms for s in m.modules()]

    def stats_equal(a, b, exact):
        f = torch.equal if exact else torch.allclose
        return f(a.running_mean, b.running_mean) and f(a.running_var, b.running_var)

    x = torch.randn(B, N, C, T, V, dtype=dt)
    mask = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0]],
                        dtype=torch.bool)
    inv5 = ~mask[:, :, None, None, None]
    x0 = x.masked_fill(inv5, 0.0)

    # 1) all-true mask equals reference forward (train-mode BatchNorm), incl. running stats.
    a, r = fresh(), fresh()
    out = forward_valid_windows(*a, x, torch.ones(B, N, dtype=torch.bool))
    enc, gru, norm, cls = r
    ref = cls(norm(gru(F.relu(enc(x.flatten(0, 1))).reshape(B, N, H))[0][:, -1]))
    _need(torch.allclose(out, ref, rtol=0, atol=1e-12), 'all-true logits mismatch')
    _need(stats_equal(a[0][2], enc[2], exact=False), 'all-true BatchNorm stats mismatch')

    # 2) mixed lengths with noncontiguous valid positions: per-sample reference and order.
    e = fresh(eval_mode=True)
    out = forward_valid_windows(*e, x, mask)
    for b in range(B):
        fb = F.relu(e[0](x[b][mask[b]]))
        ctx = e[1](fb.unsqueeze(0))[0][:, -1]
        _need(torch.allclose(out[b], e[3](e[2](ctx))[0], rtol=0, atol=1e-12), f'sample {b} mismatch')
    perm = torch.tensor([2, 0, 3, 1])
    outp = forward_valid_windows(*e, x[perm], mask[perm])
    _need(torch.allclose(outp, out[perm], rtol=0, atol=1e-12), 'sample order mismatch')
    _need(not any(flags(e)), 'eval flags changed')

    # 3) finite poison in invalid windows changes neither outputs nor BatchNorm statistics.
    p, q = fresh(), fresh()
    o0 = forward_valid_windows(*p, x0, mask)
    o1 = forward_valid_windows(*q, x.masked_fill(inv5, 1e6), mask)
    _need(torch.equal(o0, o1), 'poison changed output')
    _need(stats_equal(p[0][2], q[0][2], exact=True), 'poison changed BatchNorm stats')
    _need(all(flags(p)), 'train flags changed')

    # 4) gradients: zero for invalid windows, nonzero for every valid window, all params reached.
    g = fresh()
    xg = x0.clone().requires_grad_(True)
    forward_valid_windows(*g, xg, mask).square().sum().backward()
    _need(bool((xg.grad[~mask] == 0).all()), 'invalid window received gradient')
    _need(bool((xg.grad[mask].flatten(1).abs().sum(1) > 0).all()), 'valid window lacks gradient')
    for m in g:
        for name, prm in m.named_parameters():
            _need(prm.grad is not None and bool(torch.isfinite(prm.grad).all()), f'no grad for {name}')

    # 5) explicit validation.
    bad = mask.clone()
    bad[1] = False
    _raises(ValueError, lambda: forward_valid_windows(*e, x, bad))
    xn = x0.clone()
    xn[1, 0, 0, 0, 0] = float('nan')  # NaN in an invalid window is still rejected
    _raises(ValueError, lambda: forward_valid_windows(*e, xn, mask))
    _raises(TypeError, lambda: forward_valid_windows(*e, x, mask.long()))
    _raises(ValueError, lambda: forward_valid_windows(*e, x, mask[:, :3]))
    _raises(ValueError, lambda: forward_valid_windows(*e, x[0], mask))
    return True


def self_test():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return _self_test_body()
