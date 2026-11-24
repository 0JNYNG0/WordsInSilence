"""Single-case diagnostic: effect of manually masking visually reviewed false hand
observations on one clip's logits. NOT a fix, filter, or training step.
Usage: python counterfactual.py --config CFG.json --out NEW.json
"""
import copy, hashlib, json, pathlib, sys
import numpy as np
import torch

HAND = {'L': (33, 54), 'R': (54, 75)}
ALLOWED = {'L': [(18, 60), (102, 144), (195, 216), (237, 258)],
           'R': [(60, 102), (144, 186), (216, 237), (258, 279)]}
KEYS = ('web_src', 'pilot_src', 'observation', 'saved_input', 'saved_logits', 'clip_id',
        'model_key', 'model_sha256', 'observation_sha256', 'rows', 'side', 'target_label')


def die(msg):
    raise SystemExit('FAIL: ' + msg)


def check(cond, msg):
    if not cond:
        die(msg)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'))


def span(o):
    idx = np.flatnonzero(np.asarray(o.usable)[:, 33:75].any(axis=1))
    check(idx.size > 0, 'no usable hand rows')
    return int(idx[0]), int(idx[-1])


def main(argv):
    check(len(argv) == 4 and argv[0] == '--config' and argv[2] == '--out', 'usage: --config CFG --out NEW.json')
    cfg_path, out = pathlib.Path(argv[1]), pathlib.Path(argv[3])
    check(not out.exists(), 'output already exists')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    check(set(KEYS) <= set(cfg), 'missing config keys')
    for k in ('web_src', 'pilot_src', 'observation', 'saved_input', 'saved_logits'):
        check(pathlib.Path(cfg[k]).is_absolute(), f'{k} must be absolute')
    side = cfg['side']; check(side in HAND, 'side must be L/R')
    sys.path[:0] = [cfg['web_src'], cfg['pilot_src']]
    from wis_lab.core.observation import Observation
    from wis_lab.service.neural import _channels, _load
    from pilot.data import collate
    torch.set_num_threads(2)

    obs_sha0 = sha(cfg['observation'])
    check(obs_sha0 == cfg['observation_sha256'], 'observation sha mismatch (before)')
    src_sha = {k: sha(cfg[k]) for k in ('saved_input', 'saved_logits')}
    orig = Observation.load(cfg['observation'])
    usable0 = np.asarray(orig.usable)
    T = usable0.shape[0]
    rows = [int(r) for r in cfg['rows']]
    check(len(rows) > 0 and len(set(rows)) == len(rows), 'rows must be non-empty and unique')
    check(all(0 <= r < T for r in rows), 'row out of range')
    a, b = HAND[side]
    check(bool(usable0[rows, a:b].all()), 'selected hand points not all originally usable')

    cf = copy.deepcopy(orig)
    cf.usable[rows, a:b] = False
    removed = int(usable0.sum() - np.asarray(cf.usable).sum())
    check(removed == len(rows) * 21, f'flags removed {removed} != {len(rows) * 21}')
    diff = usable0 != np.asarray(cf.usable)
    exp = np.zeros_like(diff); exp[rows, a:b] = True
    check(bool((diff == exp).all()), 'usable changed outside selection')
    for k, v in vars(orig).items():
        if k == 'usable':
            continue
        w = getattr(cf, k)
        ok = same(v, w) if isinstance(v, np.ndarray) else v == w
        check(bool(ok), f'field {k} changed')
    lo, hi = span(orig)
    check(span(cf) == (lo, hi), 'active span changed')
    check(all(lo <= r <= hi for r in rows), 'row outside active span')

    ch0, err0, st0 = _channels(orig)
    ch1, err1, st1 = _channels(cf)
    check(err0 is None and err1 is None, f'channel error: {err0} / {err1}')
    x0, x1 = np.asarray(ch0['x']), np.asarray(ch1['x'])
    check(x0.shape == x1.shape and x0.shape == (hi - lo + 1, 280), 'x shape/span mismatch')
    with np.load(cfg['saved_input'], allow_pickle=False) as si:
        check(x0.dtype == si['x'].dtype and np.array_equal(x0, si['x']), 'original x != saved_input x (bitwise)')
        for k in ('timestamp_us', 'frame_id'):
            check(k in si.files, f'saved_input lacks {k}')
            check(same(si[k], np.asarray(getattr(orig, k))[lo:hi + 1]), f'saved_input {k} != active slice')
    check(ch0['frames'] == ch1['frames'] if np.isscalar(ch0['frames']) else same(ch0['frames'], ch1['frames']), 'frames differ')
    check(same(ch0['body_scale'], ch1['body_scale']), 'body_scale differs')
    ts = np.asarray(orig.timestamp_us)[lo:hi + 1]
    check(same(np.diff(ts), np.diff(np.asarray(cf.timestamp_us)[lo:hi + 1])), 'dt differs')
    for k in st0 if isinstance(st0, dict) else ():
        if 'span' in k or k == 'dt' or 'dt_' in k:
            check(json.dumps(st0[k], default=str) == json.dumps(st1.get(k), default=str), f'stats {k} differs')

    xd = ~(x0 == x1) & ~(np.isnan(x0) & np.isnan(x1))
    chg_rows = sorted(int(i) for i in np.flatnonzero(xd.any(axis=1)))
    check(chg_rows == sorted(r - lo for r in rows), f'changed input rows {chg_rows} != rows-lo')
    allowed = np.zeros(280, bool)
    for s, e in ALLOWED[side]:
        allowed[s:e] = True
    chg_cols = [int(i) for i in np.flatnonzero(xd.any(axis=0))]
    check(not xd[:, ~allowed].any(), f'changes outside allowed channels: {[c for c in chg_cols if not allowed[c]]}')

    m = _load(cfg['model_key'])
    check(m.get('status') == 'READY', 'model not READY')
    check(m['checkpoint_sha256'] == cfg['model_sha256'], 'model sha mismatch')
    classes = list(m['classes'])
    check(len(classes) == 93 and len(set(classes)) == 93, 'need 93 unique classes')
    check(cfg['target_label'] in classes, 'target_label not in classes')
    model = m['model']; model.eval(); model.requires_grad_(False)
    with np.load(cfg['saved_logits'], allow_pickle=False) as sl:
        ids = [str(i) for i in sl['ids']]
        check(ids.count(cfg['clip_id']) == 1, 'clip_id not found exactly once')
        check([str(c) for c in sl['classes']] == classes, 'saved class order != model')
        saved = np.asarray(sl['logits'][ids.index(cfg['clip_id'])], dtype=np.float64)

    def score(x):  # label is a dummy 0; source label is never fed to the model
        bx, pad, _ = collate([dict(x=x, frames=len(x), label=0)])
        with torch.inference_mode():
            out_ = model(torch.from_numpy(bx), torch.from_numpy(pad)).numpy()[0]
        check(out_.shape == (93,), 'logit shape')
        return out_.astype(np.float64)

    l0 = score(x0)
    dsaved = float(np.abs(l0 - saved).max())
    check(dsaved <= 1e-5, f'original vs saved logits maxabs {dsaved}')
    l1 = score(x1)

    t = classes.index(cfg['target_label'])
    def summ(l):
        o = np.argsort(-l, kind='stable')
        return dict(logits=l.tolist(), top3=[[classes[i], float(l[i])] for i in o[:3]],
                    target_rank=int(1 + (l > l[t]).sum()),
                    target_minus_best_other=float(l[t] - np.delete(l, t).max()))

    check(sha(cfg['observation']) == obs_sha0, 'observation sha changed (after)')
    check({k: sha(cfg[k]) for k in src_sha} == src_sha, 'saved source files changed')
    res = dict(
        config_sha256=sha(cfg_path), observation_sha256_before=obs_sha0, observation_sha256_after=obs_sha0,
        source_sha256=src_sha, model_sha256=m['checkpoint_sha256'], clip_id=cfg['clip_id'],
        side=side, rows=rows, target_label=cfg['target_label'],
        invariants=dict(active_span=[lo, hi], flags_removed=removed, x_bitequal_saved_input=True,
                        original_vs_saved_logits_maxabs=dsaved, coords_frames_times_unchanged=True,
                        body_scale_frames_dt_identical=True, changes_confined_to_selected_hand=True),
        changed_input_rows=chg_rows, changed_channels=chg_cols, classes=classes,
        original=summ(l0), intervention=summ(l1), maxabs_logit_delta=float(np.abs(l1 - l0).max()),
        interpretation_scope=('Single-clip counterfactual: manual masking of visually reviewed false '
                              'hand observations. Not a fix, automatic filter, or training change; '
                              'says nothing about aggregate accuracy.'))
    with open(out, 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1:])
