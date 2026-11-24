import json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

OLD_W, KSL_PER_STEP, STEPS, ROLE = 0.125, 4, 660, 'PREPARED_TRAINING_CANDIDATE_NOT_REGISTERED'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _unchanged(m, snap):
    sd = m.state_dict()
    return sd.keys() == snap.keys() and all(torch.equal(sd[k], snap[k]) for k in snap)


def _save_pt(path, obj):
    with open(path, 'xb') as fh:
        torch.save(obj, fh)


def _progress(tag):
    def cb(arm, done, total):
        done, total = int(done), int(total)
        if done % 100 == 0 or done == total:
            print(f'[{tag}] {arm} {done}/{total}', flush=True)
    return cb


def _verify_inputs(E, cfg, old_run):
    for p, s in cfg['pins'].items():
        E.need(E.sha(p) == s, f'pin mismatch: {p}')
    E.verify_prepared()
    E.verify_receipt()
    tr = E.read(old_run / 'TRAINING.json')
    for name, s in tr['files'].items():
        E.need(E.sha(old_run / name) == s, f'old TRAINING file hash mismatch: {name}')
    return tr


def _check_out(E, out, psd, cache, sched, tag):
    need, sd = E.need, out['state_dict']
    losses = np.asarray(out['losses'])
    need(int(out['updates']) == STEPS, f'updates != {STEPS} {tag}')
    need(losses.shape == (STEPS, len(E.T.LOSS_COLUMNS)) and np.isfinite(losses).all(), f'losses shape/finite {tag}')
    need(_same(np.asarray(out['base_indices']), cache[5]), f'base_indices differ {tag}')
    need(_same(np.asarray(out['context_indices']), cache[6]), f'context_indices differ {tag}')
    need(_same(np.asarray(out['extra_indices']), sched), f'extra indices differ {tag}')
    need(sd.keys() == psd.keys(), f'state keys differ {tag}')
    need(all(sd[n].shape == psd[n].shape and sd[n].dtype == psd[n].dtype for n in psd), f'state shape/dtype differ {tag}')
    need(all(n.startswith('head.') for n in out['trainable_names']), f'non-head trainable {tag}')
    need(all(torch.equal(sd[n], psd[n]) for n in psd if not n.startswith('head.')), f'non-head tensor changed {tag}')
    need(any(not torch.equal(sd[n], psd[n]) for n in psd if n.startswith('head.')), f'head unchanged {tag}')
    need(bool(out['frozen_parameters_unchanged']) and bool(out['buffers_unchanged']), f'frozen/buffer change {tag}')
    return sd, losses


def run(E, cfg):
    G, T, need = E.G, E.T, E.need
    ops, run_dir, old_run = Path(cfg['ops_dir']), Path(cfg['run_dir']), Path(E.RUN)
    need(ops.is_dir(), 'ops_dir missing')
    need(not run_dir.exists(), 'run_dir already exists')
    need(run_dir.resolve() != old_run.resolve(), 'run_dir equals old run')
    tol, W, S = float(cfg['tolerance']), float(cfg['extra_weight']), int(cfg['support_per_step'])
    need(W == 0.25 and S == 4, 'unexpected cfg weight/support_per_step')
    need(W / (KSL_PER_STEP + S) == OLD_W / KSL_PER_STEP, 'mean coefficient .25/8 != .125/4')
    wrapper_sha = E.sha(__file__)
    timing = {'started_utc': _utc()}

    # 1. verify + load
    old_tr = _verify_inputs(E, cfg, old_run)
    pol, ck, parent, classes, ref, refmodel = E.load_models()
    classes = [str(c) for c in classes]
    cache = E.load_prior_cache(classes)
    new = E.load_new()
    psd0 = _snap(parent)
    old = dict(np.load(old_run / 'training_arrays.npz', allow_pickle=False))
    need(_same(cache[5], old['base_indices']) and _same(cache[6], old['context_indices']), 'cache indices differ from old arrays')

    need([str(c) for c in np.asarray(new['classes']).tolist()] == classes, 'new classes != model classes')
    ny, nf, ids = np.asarray(new['y']), np.asarray(new['folds']), list(new['id_list'])
    need(ny.dtype == np.int64 and ny.shape == (36,) and nf.shape == (36,) and len(ids) == 36, 'new source shape')
    class_order = sorted(set(ny.tolist()))
    need(len(class_order) == 2, 'new source is not two-label')
    new_labels = {classes[c] for c in class_order}

    m = E.read(cfg['support_manifest'])
    rows = m['rows'] if isinstance(m, dict) else m
    need(len(rows) == 113, 'support row count != 113')
    need(len({r['sample_id'] for r in rows}) == 113, 'sample_id not unique')
    need(len({r['clip_id'] for r in rows}) == 105, 'clip_id count != 105')
    need(len({r['label'] for r in rows}) == 69, 'label count != 69')
    need(all(r['role'] == ROLE for r in rows), 'disallowed role')
    need(all(int(r['signer_id']) in (1, 2) for r in rows), 'bad signer_id')
    need(all(r['label'] in classes for r in rows), 'support label not in classes')
    need(not ({r['label'] for r in rows} & new_labels), 'support labels overlap two-label set')
    rows = sorted(rows, key=lambda r: (classes.index(r['label']), int(r['signer_id']), str(r['clip_id']), str(r['sample_id'])))
    xs = []
    for r in rows:
        need(E.sha(r['input_path']) == r['input_sha256'], f"input hash mismatch: {r['sample_id']}")
        with np.load(r['input_path'], allow_pickle=False) as z:
            x = z['x']
        need(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280 and np.isfinite(x).all(), f"bad x: {r['sample_id']}")
        xs.append(x)
    sup_prefix = [np.asarray(p, dtype=np.float32) for p in G.extract_prefix(parent, xs)]
    need(len(sup_prefix) == 113 and _unchanged(parent, psd0), 'prefix extraction failed or modified parent')
    sup_y = np.array([classes.index(r['label']) for r in rows], dtype=np.int64)
    sup_ids = np.array([str(r['sample_id']) for r in rows])

    # 2. support schedule (labels from manifest only)
    need(old['extra_schedule__candidate_fold0'].shape[0] == STEPS, 'old schedule is not 660 steps')
    labs = sorted(set(sup_y.tolist()))
    starts = {c: int(np.flatnonzero(sup_y == c)[0]) for c in labs}
    sizes = {c: int((sup_y == c).sum()) for c in labs}
    sup_sched = np.empty((STEPS, S), dtype=np.int64)
    for j in range(STEPS * S):
        c, occ = labs[j % len(labs)], j // len(labs)
        sup_sched[j // S, j % S] = starts[c] + occ % sizes[c]
    draws = np.bincount(sup_y[sup_sched.ravel()], minlength=len(classes))[labs]
    need(set(draws.tolist()) <= {38, 39}, 'support draws not 38/39 per label')

    # 3. fold construction + preflight
    folds, checks = {}, {'support_draws_min': int(draws.min()), 'support_draws_max': int(draws.max()),
                         'new_class_order': class_order, 'new_labels': sorted(new_labels)}
    head_fn = lambda ps: (lambda h, mk: parent.head(G._masked_mean(parent.blocks[-1](h, mk), mk)))(*G._pad_batch(ps))
    for f in range(3):
        groups = T.fold_training_rows(ids, ny, nf, f, class_order)
        tr_rows = np.concatenate([np.asarray(g) for g in groups])
        osch = np.asarray(T.cyclic_schedule([len(g) for g in groups], STEPS, per_class=2))
        k = f'candidate_fold{f}'
        need(_same(tr_rows, old[f'train_rows__{k}']), f'train rows differ fold {f}')
        need(_same(osch, old[f'extra_schedule__{k}']), f'old schedule differs fold {f}')
        held = old[f'heldout_rows__{k}']
        need(not set(tr_rows.tolist()) & set(held.tolist()) and not (nf[tr_rows] == f).any(), f'train/eval overlap fold {f}')
        n_ksl = len(tr_rows)
        xp = [new['prefixes'][int(i)] for i in tr_rows] + sup_prefix
        xy = np.concatenate([ny[tr_rows], sup_y])
        xsch = np.concatenate([osch, sup_sched + n_ksl], axis=1)
        need(xsch.shape == (STEPS, 8) and _same(xsch[:, :KSL_PER_STEP], osch), f'combined schedule mismatch fold {f}')
        need(xsch.min() >= 0 and xsch.max() < len(xp), f'schedule index out of range fold {f}')
        md = 0.0
        with torch.no_grad():
            for t in range(STEPS):
                a = head_fn([xp[int(i)] for i in xsch[t, :4]])
                b = head_fn([xp[int(i)] for i in xsch[t]])[:4]
                md = max(md, float((a - b).abs().max()))
        need(md <= tol, f'padding logit diff {md} > tol fold {f}')
        checks[f'fold{f}'] = {'n_ksl_train': n_ksl, 'n_heldout': int(len(held)), 'max_padding_logit_abs_diff': md,
                              'old_rows_schedule_bytewise_equal': True, 'first4_equal_old': True}
        folds[f] = dict(rows=tr_rows, held=held, osch=osch, xp=xp, xy=xy, xsch=xsch, n=n_ksl)
    need(_unchanged(parent, psd0), 'preflight modified parent')
    checks['mean_coefficient'] = [W / 8, OLD_W / 4]

    npz = {f'support_prefix__{i:03d}': p for i, p in enumerate(sup_prefix)}
    npz.update(support_schedule=sup_sched, support_y=sup_y, support_ids=sup_ids)
    for f, d in folds.items():
        npz[f'old_extra_schedule__candidate_fold{f}'] = d['osch']
        npz[f'extra_schedule__candidate_fold{f}'] = d['xsch']
    E.save_npz(ops / 'SUPPORT_PREPARED.npz', npz)
    E.write_json(ops / 'SUPPORT_PREPARED_RECEIPT.json', {
        'rows': [{k: r[k] for k in ('sample_id', 'clip_id', 'label', 'signer_id', 'input_path', 'input_sha256')} for r in rows],
        'labels': [classes[c] for c in labs], 'npz_sha256': E.sha(ops / 'SUPPORT_PREPARED.npz'),
        'pins': cfg['pins'], 'author_wrapper_sha256': wrapper_sha, 'created_utc': _utc()})
    E.write_json(ops / 'PREFLIGHT.json', {'checks': checks, 'tolerance': tol, 'author_wrapper_sha256': wrapper_sha,
                                          'created_utc': _utc()})

    # 4. reproduce old fold0 before any candidate training
    need(not run_dir.exists(), 'run_dir appeared before reproduction')
    d0, psd = folds[0], parent.state_dict()
    timing['reproduction_start_utc'], t0 = _utc(), time.perf_counter()
    rep = T.train_arm(parent, *cache, extra_prefixes=d0['xp'][:d0['n']], extra_y=d0['xy'][:d0['n']],
                      extra_indices=d0['osch'], extra_weight=OLD_W, arm='candidate_fold0',
                      progress=_progress('repro0'), **T.TRAIN_KW)
    timing['reproduction_elapsed_s'], timing['reproduction_end_utc'] = time.perf_counter() - t0, _utc()
    rsd, rl = _check_out(E, rep, psd, cache, d0['osch'], 'repro0')
    osd = torch.load(old_run / 'candidate_fold0_final.pt', map_location='cpu', weights_only=True)['state_dict']
    keys_ok = osd.keys() == rsd.keys()
    tdiff = max((float((rsd[k].double() - osd[k].double()).abs().max()) if rsd[k].numel() else 0.0) for k in osd) if keys_ok else None
    teq = keys_ok and all(torch.equal(rsd[k], osd[k]) for k in osd)
    ol = old['losses__candidate_fold0']
    leq = np.array_equal(rl, ol)
    ldiff = float(np.abs(rl - ol).max()) if rl.shape == ol.shape else None
    E.write_json(ops / 'REPRODUCTION.json', {'state_equal': bool(teq), 'losses_equal': bool(leq), 'keys_equal': bool(keys_ok),
                                             'max_tensor_abs_diff': tdiff, 'max_loss_abs_diff': ldiff, 'steps': STEPS,
                                             'elapsed_s': timing['reproduction_elapsed_s'],
                                             'start_utc': timing['reproduction_start_utc'], 'end_utc': timing['reproduction_end_utc']})
    need(teq and leq, 'reproduction of old fold0 failed')
    need(_unchanged(parent, psd0), 'reproduction modified parent')

    # 5. candidate training
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(old_run / 'control_final.pt', 'rb') as src, open(run_dir / 'control_final.pt', 'xb') as dst:
        shutil.copyfileobj(src, dst)
    need(E.sha(run_dir / 'control_final.pt') == old_tr['files']['control_final.pt'], 'control copy hash mismatch')
    arrays = {'base_indices': old['base_indices'], 'context_indices': old['context_indices'],
              'support_ids': sup_ids, 'support_schedule': sup_sched}
    meta_ck = {k: v for k, v in ck.items() if k != 'state_dict'}
    for f, d in folds.items():
        k = f'candidate_fold{f}'
        st, t0 = _utc(), time.perf_counter()
        out = T.train_arm(parent, *cache, extra_prefixes=d['xp'], extra_y=d['xy'], extra_indices=d['xsch'],
                          extra_weight=W, arm=k, progress=_progress(k), **T.TRAIN_KW)
        timing[k] = {'start_utc': st, 'end_utc': _utc(), 'elapsed_s': time.perf_counter() - t0}
        sd, losses = _check_out(E, out, psd, cache, d['xsch'], k)
        _save_pt(run_dir / f'{k}_final.pt', {**meta_ck, 'state_dict': sd, 'mp_support_addition': {
            'fold': f, 'train_ids': [ids[int(i)] for i in d['rows']], 'support_ids': sup_ids.tolist(),
            'extra_weight': W, 'old_extra_weight': OLD_W, 'pins': cfg['pins'], 'author_wrapper_sha256': wrapper_sha,
            'timing': timing[k]}})
        arrays.update({f'losses__{k}': losses, f'extra_schedule__{k}': d['xsch'], f'old_extra_schedule__{k}': d['osch'],
                       f'train_rows__{k}': d['rows'], f'heldout_rows__{k}': d['held']})
    need(_unchanged(parent, psd0), 'training modified parent')
    E.save_npz(run_dir / 'training_arrays.npz', arrays)

    # 6. TRAINING.json
    _verify_inputs(E, cfg, old_run)
    timing['finished_utc'] = _utc()
    names = ['control_final.pt'] + [f'candidate_fold{f}_final.pt' for f in range(3)] + ['training_arrays.npz']
    E.write_json(run_dir / 'TRAINING.json', {
        'status': E.TRAINED_STATUS, 'prepared_sha256': E.sha(Path(E.PREP) / 'PREPARED.json'),
        'files': {n: E.sha(run_dir / n) for n in names}, 'scripts': E.script_hashes(),
        'author_wrapper_sha256': wrapper_sha, 'cfg': json.loads(json.dumps(cfg, default=str)), 'checks': checks,
        'parent_sha256': E.PARENT_SHA, 'reference_sha256': E.REF_SHA, 'old_extra_weight': OLD_W, 'new_extra_weight': W,
        'support_count': 113, 'folds': [0, 1, 2], 'control_reused': True, 'timing': timing,
        'old_training_sha256': E.sha(old_run / 'TRAINING.json'), 'old_recipe_is_new_recipe': False,
        'recipe': {'per_step': 'old 32 base/context + two-label KSL 4 + support 4', 'steps': STEPS,
                   'extra_weight': W, 'extra_mean_coefficient': W / 8, 'trainable': 'head only',
                   'train_kw': json.loads(json.dumps(T.TRAIN_KW, default=str))}})
    return run_dir
