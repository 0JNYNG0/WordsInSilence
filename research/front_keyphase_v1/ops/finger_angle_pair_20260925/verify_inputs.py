"""Freeze the comparison and verify the geometry blocks before any training.

No model is trained here. The parent is loaded once only to assert that the zero-initialised
extra path reproduces its output exactly, which is what makes the two arms share a start point.
"""
import hashlib, json, sys, time, pathlib
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(H), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u                     # noqa: E402
import angle_block as AB                                     # noqa: E402
from generated_v1 import ExtraInputTCN, fit_channel_scale     # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CACHE = L / 'ops/motion_input_contract_20260922/training_motion_inputs.npz'
PARENT = L / 'runs/public_anchor_0p01_pair_v1/checkpoint_treatment_final.pt'
PARENT_SHA = 'b8d6045c0c154697a5d6d3440927a48cf1e9db3146421db0a920da9fa684870d'
CACHE_SHA = '66b768800ad70f78c482baeba95efe842c950b255376ff76a50af90c68e7b0dc'


def unpack(v, off):
    return [v[a:b].copy() for a, b in zip(off[:-1], off[1:])]


def main():
    started = time.time()
    c = {}
    c['parent_hash'] = sha(PARENT) == PARENT_SHA
    c['cache_exists'] = CACHE.exists()
    assert not (H / 'INPUT_READY.json').exists(), 'already prepared'

    with np.load(CACHE, allow_pickle=False) as z:
        classes = z['classes'].tolist()
        base_x = unpack(z['base_x'], z['base_offsets'])
        ctx_x = unpack(z['pair_context_x'], z['pair_context_offsets'])
        base_y, ctx_y = z['base_y'].copy(), z['pair_context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        bidx, cidx = z['base_indices'].copy(), z['context_indices'].copy()
    cache_sha = sha(CACHE)
    c['cache_hash_recorded'] = True
    c['counts_2122_224'] = len(base_x) == 2122 and len(ctx_x) == 224
    c['classes_93'] = len(classes) == 93
    c['schedule_660x16'] = bidx.shape == cidx.shape == (660, 16)
    c['teacher_shape'] = teacher.shape == (2122, 93)
    c['all_280_channels'] = all(a.shape[1] == 280 for a in base_x + ctx_x)

    # ---- build the two geometry blocks -----------------------------------
    blocks = {}
    for name, seqs in (('base', base_x), ('context', ctx_x)):
        ang, cho = [], []
        for a in seqs:
            g, k = AB.blocks(a)
            ang.append(g)
            cho.append(k)
        blocks[name] = {'angle': ang, 'chord': cho}
    c['block_width_76'] = all(g.shape[1] == AB.EXTRA_WIDTH for g in blocks['base']['angle'])
    c['block_frames_match_input'] = all(g.shape[0] == a.shape[0]
                                        for g, a in zip(blocks['base']['angle'], base_x))
    c['blocks_finite'] = all(bool(np.isfinite(g).all()) and bool(np.isfinite(k).all())
                             for g, k in zip(blocks['base']['angle'], blocks['base']['chord']))

    # the two blocks must share the validity mask exactly (width-matched control)
    c['shared_validity_mask'] = all(
        np.array_equal(g[:, AB.EXTRA_VALUES:], k[:, AB.EXTRA_VALUES:])
        for g, k in zip(blocks['base']['angle'], blocks['base']['chord']))

    # invalid entries must be exactly zero in BOTH blocks
    bad = 0
    for g, k in zip(blocks['base']['angle'], blocks['base']['chord']):
        m = g[:, AB.EXTRA_VALUES:] > 0.5
        bad += int(np.count_nonzero(g[:, :AB.EXTRA_VALUES][~m])) + \
               int(np.count_nonzero(k[:, :AB.EXTRA_VALUES][~m]))
    c['invalid_entries_are_zero'] = bad == 0

    # angles live in [0,1]; chords are non-negative
    c['angle_range_0_1'] = all(
        float(g[:, :AB.EXTRA_VALUES].min()) >= 0.0 and float(g[:, :AB.EXTRA_VALUES].max()) <= 1.0
        for g in blocks['base']['angle'])
    c['chord_non_negative'] = all(float(k[:, :AB.EXTRA_VALUES].min()) >= 0.0
                                  for k in blocks['base']['chord'])

    # ---- independent recomputation on a fixed sample ----------------------
    rng_free_pick = list(range(0, 2122, 211))                      # deterministic, 11 clips
    maxang = maxcho = 0.0
    checked = 0
    for i in rng_free_pick:
        x = base_x[i]
        pts, ok = AB.hand_points(x)
        g, k = blocks['base']['angle'][i], blocks['base']['chord'][i]
        for h in range(2):
            for j, (kind, a, b, cc) in enumerate(AB.TRIPLETS):
                uu, vv, jok = AB._vectors(pts[:, h], ok[:, h], kind, a, b, cc)
                nu, nv = np.linalg.norm(uu, axis=-1), np.linalg.norm(vv, axis=-1)
                good = jok & (nu > AB.EPS) & (nv > AB.EPS)
                idx = h * AB.N_ANGLES + j
                # independent path: atan2 based angle instead of arccos of the normalised dot
                cross = uu[:, 0] * vv[:, 1] - uu[:, 1] * vv[:, 0]
                dot = (uu * vv).sum(-1)
                alt = np.arctan2(np.abs(cross), dot) / np.pi
                maxang = max(maxang, float(np.max(np.abs(alt[good] - g[good, idx]))) if good.any() else 0.0)
                altc = np.sqrt(((uu - vv) ** 2).sum(-1))
                maxcho = max(maxcho, float(np.max(np.abs(altc[good] - k[good, idx]))) if good.any() else 0.0)
                checked += int(good.sum())
    c['independent_angle_recomputation'] = maxang < 1e-5
    c['independent_chord_recomputation'] = maxcho < 1e-5

    # ---- scale invariance of the angle block (property the control lacks) --
    probe = base_x[0].copy()
    hx = slice(AB.HAND_XY0, AB.HAND_XY0 + AB.CH_HAND_XY)
    probe[:, hx] = probe[:, hx] * 1.7
    g2, k2 = AB.blocks(probe)
    g0, k0 = blocks['base']['angle'][0], blocks['base']['chord'][0]
    c['angle_is_scale_invariant'] = float(np.max(np.abs(g2 - g0))) < 1e-5
    c['chord_is_not_scale_invariant'] = float(np.max(np.abs(k2 - k0))) > 1e-3

    # ---- train-only scales -------------------------------------------------
    fits = {f: fit_channel_scale(blocks['base'][f], base_y, AB.EXTRA_VALUES) for f in ('angle', 'chord')}
    c['scale_from_train_only'] = all(f['clip_count'] == 2122 for f in fits.values())
    c['scale_positive'] = all(bool((f['scale'] > 0).all()) for f in fits.values())

    # ---- zero-init residual reproduces the parent exactly ------------------
    ck = torch.load(PARENT, map_location='cpu', weights_only=True)
    parent, cl = u.rebuild_from_payload(ck)
    parent.eval()
    torch.set_num_threads(2)
    c['parent_classes_match'] = cl == classes
    xs = [base_x[i] for i in range(4)]
    es = [blocks['base']['angle'][i] for i in range(4)]
    t = max(a.shape[0] for a in xs)
    X = np.zeros((4, t, 280), dtype=np.float32); E = np.zeros((4, t, AB.EXTRA_WIDTH), dtype=np.float32)
    M = np.zeros((4, t), dtype=np.float32)
    for i, (a, g) in enumerate(zip(xs, es)):
        X[i, :len(a)] = a; E[i, :len(g)] = g; M[i, :len(a)] = 1.0
    model = ExtraInputTCN(parent, fits['angle']['scale'], AB.EXTRA_VALUES, AB.EXTRA_WIDTH).eval()
    with torch.inference_mode():
        got = model(torch.from_numpy(X), torch.from_numpy(E), torch.from_numpy(M)).numpy()
        ref = parent(torch.from_numpy(X), torch.from_numpy(M)).numpy()
    zero_err = float(np.max(np.abs(got - ref)))
    c['zero_init_reproduces_parent'] = zero_err < 1e-6
    c['extra_weight_starts_at_zero'] = bool(
        torch.count_nonzero(model.state_dict()['extra_weight']).item() == 0)

    failed = [k for k, v in c.items() if not v]
    arrays = {'classes': np.array(classes),
              'angle_scale': fits['angle']['scale'], 'chord_scale': fits['chord']['scale']}
    for g in ('base', 'context'):
        for f in ('angle', 'chord'):
            vals = np.concatenate(blocks[g][f], axis=0)
            off = np.cumsum([0] + [len(a) for a in blocks[g][f]], dtype=np.int64)
            arrays[f'{g}_{f}'] = vals
            arrays[f'{g}_{f}_offsets'] = off
    assert not failed, failed
    with (H / 'geometry_blocks.npz').open('xb') as fh:
        np.savez_compressed(fh, **arrays)

    out = {'status': 'PASS' if not failed else 'FAIL',
           'checks': c, 'checks_total': len(c), 'checks_failed': failed,
           'parent': {'path': str(PARENT), 'sha256': PARENT_SHA},
           'input_cache': {'path': str(CACHE), 'sha256': cache_sha},
           'blocks_sha256': sha(H / 'geometry_blocks.npz'),
           'module_sha256': {'angle_block.py': sha(H / 'angle_block.py'),
                             'generated_v1.py': sha(H / 'generated_v1.py')},
           'geometry': {'angles_per_hand': AB.N_ANGLES, 'value_channels': AB.EXTRA_VALUES,
                        'block_width': AB.EXTRA_WIDTH,
                        'definition': 'angle = arccos(u.v/|u||v|)/pi ; chord = |u-v| ; same u,v, same validity'},
           'scales': {f: {'min': float(fits[f]['scale'].min()), 'max': float(fits[f]['scale'].max()),
                          'never_observed_channels': fits[f]['never_observed_channels']}
                      for f in fits},
           'independent_values_checked': checked,
           'max_independent_angle_difference': maxang,
           'max_independent_chord_difference': maxcho,
           'zero_init_max_logit_difference': zero_err,
           'training_runs': 0, 'development_arrays_read': 0, 'user_arrays_read': 0,
           'reserved_arrays_read': 0, 'downloads': 0, 'new_extraction': 0,
           'seconds': round(time.time() - started, 2),
           'limits': ['2-D projected angles, not real 3-D flexion angles',
                      'not an expert handshape ground truth',
                      'no new observation: every value is a function of coordinates already input']}
    with (H / 'INPUT_READY.json').open('x', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write('\n')
    print('checks', len(c), 'failed', failed)
    print('independent values checked', checked,
          '| max angle diff %.2e | max chord diff %.2e' % (maxang, maxcho))
    print('zero-init max logit difference %.2e' % zero_err)
    print('angle scale [%.4f, %.4f] | chord scale [%.4f, %.4f]'
          % (fits['angle']['scale'].min(), fits['angle']['scale'].max(),
             fits['chord']['scale'].min(), fits['chord']['scale'].max()))
    print('blocks sha256', out['blocks_sha256'][:16], '| seconds', out['seconds'])


if __name__ == '__main__':
    main()
