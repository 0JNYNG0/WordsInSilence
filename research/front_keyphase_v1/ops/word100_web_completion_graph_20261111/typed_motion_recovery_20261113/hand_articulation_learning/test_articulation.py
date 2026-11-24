"""Numeric-fixture tests only. No public video/old checkpoint/model re-evaluation."""
from articulation_core import *
from hand_core import HandObserver, original, load_parent_weights, torch, U, write, stamp, sha, preserve_failure
from pathlib import Path


def fixture():
    p = np.zeros((2, 21, 2), np.float32)
    p[:, 0] = [0, 0]
    for f, joints in enumerate(FINGERS):
        x = .4 * (f - 1)
        p[0, joints] = [[x, -1], [x, -1.4], [x, -1.8], [x, -2.2]]
        p[1, joints] = [[x, -1], [x, -1.4], [x + .4, -1.4], [x + .4, -1]]
    p[:, 1:5] = [[[-.5, -.3], [-.8, -.6], [-1, -.7], [-1.2, -.8]]]
    return np.c_[p.reshape(2, 42), np.ones((2, 21))].astype(np.float32)


def main():
    dest = U / 'PURE_TESTS.json'
    assert not dest.exists()
    checks = []
    x = fixture(); a = articulation24(x)
    assert np.allclose(a[0, :12], np.tile([-1, -1, 1], 4), atol=1e-6)
    assert np.allclose(a[1, :12], np.tile([0, 0, 1/3], 4), atol=1e-6)
    checks.append('Straight and bent projected finger fixtures have exact analytic cosines/ratios')
    for degrees in [-90, -60, -30, 30, 60, 90]:
        t = np.deg2rad(degrees); rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        y = x.copy(); y[:, :42] = (x[:, :42].reshape(2, 21, 2) @ rot.T).reshape(2, 42)
        assert np.allclose(articulation24(y), a, atol=1e-6)
    checks.append('All six fixed image-plane rotations preserve every feature and mask')
    y = x.copy(); y[:, :42] = (x[:, :42].reshape(2, 21, 2) * 2.7 + [4, -3]).reshape(2, 42)
    assert np.allclose(articulation24(y), a, atol=2e-6)
    y = x.copy(); y[:, :42:2] *= -1
    assert np.allclose(articulation24(y), a, atol=1e-6)
    checks.append('Positive scale, translation and anatomical reflection preserve descriptors')
    y = x.copy(); y[:, 42 + 7] = 0; b = articulation24(y)
    assert not b[:, :3].any() and not b[:, 12:15].any()
    assert np.array_equal(b[:, 3:12], a[:, 3:12]) and np.array_equal(b[:, 15:], a[:, 15:])
    checks.append('Missing index DIP masks only its three dependent descriptors; other fingers stay intact')
    y = x.copy(); y[:, 42 + 8] = 0; b = articulation24(y)
    assert np.array_equal(b[:, 0], a[:, 0]) and b[:, 12].all()
    assert not b[:, 1:3].any() and not b[:, 13:15].any()
    checks.append('Missing tip does not erase observable PIP cosine or invent DIP/ratio')
    y = x.copy(); y[:, 12:14] = y[:, 10:12]; b = articulation24(y)
    assert not b[:, 0].any() and not b[:, 12].any() and not b[:, 2].any() and not b[:, 14].any()
    checks.append('Degenerate bone length yields unavailable features, not zero-angle truth')
    y = x.copy(); y[:, 42:] = 0
    assert not articulation24(y).any()
    try:
        articulation24(np.full((1, 63), np.nan))
        raise AssertionError('Nonfinite input was silently accepted')
    except ValueError:
        pass
    checks.append('All-missing becomes all-zero value/mask, nonfinite input is rejected')
    train = a.copy(); train[1, 0] = 999; train[1, 12] = 0
    mean, std, n = fit_articulation_stats([train])
    assert mean[0] == a[0, 0] and n[0] == 1
    norm = normalize_articulation_numpy(train, mean, std)
    assert norm[1, 0] == 0 and np.array_equal(norm[:, 12:], train[:, 12:])
    checks.append('Only available FIT values enter stats; missing standardized values are explicitly zero')
    torch.manual_seed(1729)
    base = original.HandObserver(np.zeros(63, np.float32), np.ones(63, np.float32)).eval()
    extra_mean, extra_std, _ = fit_articulation_stats([a])
    new = HandObserver(np.zeros(63, np.float32), np.ones(63, np.float32), extra_mean, extra_std).eval()
    load_parent_weights(new, base.state_dict())
    raw = torch.tensor(np.stack([np.tile(x, (3, 1)), np.tile(x[::-1], (3, 1))]))
    extras = torch.tensor(np.stack([np.tile(a, (3, 1)), np.tile(a[::-1], (3, 1))]))
    lengths = torch.tensor([6, 4]); combined = torch.cat([raw, extras], -1)
    with torch.inference_mode():
        before, after = base(raw, lengths), new(combined, lengths)
    for key in before:
        assert torch.equal(before[key], after[key]), key
    checks.append('Zero residual preserves all original fixture logits/latents exactly with unequal padding')
    new.train(); oo = new(combined, lengths)
    loss = oo['attributes'].square().mean() + oo['transition'].square().mean(); loss.backward()
    assert torch.isfinite(new.articulation_residual.weight.grad).all()
    assert torch.count_nonzero(new.articulation_residual.weight.grad).item() > 0
    checks.append('Zero initialization permits finite nonzero learning gradient in the residual')
    write(dest, dict(at=stamp(), pass_=True, tests=len(checks), checks=checks,
                     numeric_fixtures_only=True, parent_checkpoint_loaded=False, training_executed=False,
                     pins={str(p): sha(p) for p in [Path(__file__), U/'articulation_core.py', U/'hand_core.py']}))
    print('Articulation numeric tests passed:', len(checks), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException: preserve_failure('PURE_TESTS'); raise
