"""Three-arm comparison: baseline (stored operating cache) / recovery candidate / arm-consistency candidate.

Coordinate sameness is REPORTED, never used as correctness. Every slot where arm differs from recovery
is listed with any existing video verdict; the rest are reviewed afterwards on video.
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
REC = H.parent / 'temporal_recovery_candidate_20260926'
OLD = H.parent / 'temporal_assignment_candidate_20260926'
SA = H.parent / 'side_assignment_candidate_20260925'
sys.path[:0] = [str(H), str(REC)]
import slot_review as sr                                          # noqa: E402  (loads REC run_compare as sr._rc)
from arm_landmarker import ArmLandmarker                          # noqa: E402
rc = sr._rc
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def side_state(xy, us, r, s):
    return sr.content(xy, us, r, s)


def diff_slots(axy, aus, bxy, bus):
    """slot-level differences arm(b) vs reference(a): list of (row, side, kind)"""
    out = []
    for r in range(axy.shape[0]):
        for s in (0, 1):
            ac, bc = side_state(axy, aus, r, s), side_state(bxy, bus, r, s)
            if ac is None and bc is None or rc.same(ac, bc):
                continue
            if ac is not None and bc is None:
                kind = 'MOVED_OUT' if rc.same(ac, side_state(bxy, bus, r, 1 - s)) else 'LOST'
            elif ac is None:
                kind = 'MOVED_IN' if rc.same(bc, side_state(axy, aus, r, 1 - s)) else 'NEW'
            else:
                kind = 'CHANGED_CONTENT'
            out.append((r, s, kind))
    return out


def eq_input(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(eq_input(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS'
    rec_stored = {r['clip']: r for r in json.loads((REC / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}
    old_stored = {r['clip']: r for r in json.loads((OLD / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}
    ck = rc.L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = rc.u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    rv = json.loads((SA / 'RUN_REVIEW.json').read_text(encoding='utf-8'))['rows']
    run_verdict = {(r['clip'], f): r['category'] for r in rv for f in range(r['run'][0], r['run'][-1] + 1)}
    started = time.time()

    rows, changes, control_ok, gaps = [], [], [], {'baseline': [], 'recovery': [], 'arm': []}
    per = {}
    for p in sorted(rc.CACHE.glob('*.npz')):
        clip = p.stem
        z = np.load(p, allow_pickle=False)
        src = json.loads((rc.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        o, _ = sr.run(sr.TemporalLandmarker, src, clip)
        c, _ = sr.run(sr.RecoveryLandmarker, src, clip)
        with ArmLandmarker(rc.CONTRACT) as lm:
            a = rc.extract(src, lm, clip)
            reasons, stats = dict(lm.reasons), dict(lm.stats)
        bxy, bus = z['xy_px'], z['usable']
        d_cb, _, _ = rc.decompose(bxy, bus, c.xy_px, c.usable)
        control_ok.append(d_cb == rec_stored[clip]['recov_vs_base'])     # recovery reproduced
        d_ab, _, _ = rc.decompose(bxy, bus, a.xy_px, a.usable)
        fids = list(map(int, z['frame_id']))
        for r, s, kind in diff_slots(c.xy_px, c.usable, a.xy_px, a.usable):
            changes.append({'clip': clip, 'frame': fids[r], 'side': 'LR'[s], 'kind_vs_recovery': kind,
                            'baseline_has': side_state(bxy, bus, r, s) is not None,
                            'arm_equals_baseline_slot': rc.same(side_state(bxy, bus, r, s), side_state(a.xy_px, a.usable, r, s)),
                            'existing_run_verdict': run_verdict.get((clip, fids[r]))})
        for k, us in (('baseline', bus), ('recovery', c.usable), ('arm', a.usable)):
            gaps[k] += rc.gaps(us)
        label = rc.NIKL_LABEL[clip.replace('NIKL_', '').rsplit('_', 1)[0]]
        A = rc.channels(a.xy_px, a.usable, a.timestamp_us)
        inputs = {'baseline': rc.channels(bxy, bus, z['timestamp_us']),
                  'original': rc.channels(o.xy_px, o.usable, o.timestamp_us),
                  'recovery': rc.channels(c.xy_px, c.usable, c.timestamp_us)}
        same_as = [k for k, v in inputs.items() if eq_input(A, v)]
        row = {'clip': clip, 'label': label, 'arm_vs_base': d_ab, 'recov_vs_base': d_cb,
               'sides': {'baseline': int(np.stack([bus[:, 33:54].any(1), bus[:, 54:75].any(1)], 1).sum()),
                         'recovery': int(np.stack([c.usable[:, 33:54].any(1), c.usable[:, 54:75].any(1)], 1).sum()),
                         'arm': int(np.stack([a.usable[:, 33:54].any(1), a.usable[:, 54:75].any(1)], 1).sum())},
                 'reasons': reasons, 'stats': stats, 'tcn_input_identical_to': same_as}
        if label is None:
            row['score_arm'] = 'NOT_SCORABLE_label_not_in_vocabulary'
        elif 'baseline' in same_as:
            row['score_arm'] = dict(old_stored[clip]['baseline'], source='reused: identical to baseline input')
        elif 'original' in same_as:
            row['score_arm'] = dict(old_stored[clip]['candidate'], source='reused: identical to original input')
        elif 'recovery' in same_as and 'score_recovery' in rec_stored[clip]:
            row['score_arm'] = dict(rec_stored[clip]['score_recovery'], source='reused: identical to recovery input')
        else:
            row['score_arm'] = dict(rc.score(model, classes, A, label), source='NEWLY_SCORED')
        if label is not None:
            row['score_baseline'] = old_stored[clip]['baseline']
            row['regression_vs_baseline'] = bool(row['score_baseline']['correct'] and not row['score_arm']['correct'])
        per[clip] = ({f: i for i, f in enumerate(fids)}, bus, c.usable, a.usable)
        rows.append(row)
        print(f"{clip:<24} sides {row['sides']}  input={same_as or 'NEW'}  "
              f"removed_pairs {reasons['pair_removed_arm_contradiction']}", flush=True)

    # ---- SEN0354 -------------------------------------------------------------
    sen = {}
    runs = {}
    for tag, cls in (('recovery', sr.RecoveryLandmarker), ('arm', ArmLandmarker)):
        with cls(rc.CONTRACT) as lm:
            runs[tag] = rc.extract(str(rc.SEN), lm, rc.SEN.stem)
    for rr in (38, 40, 44):
        e = {}
        for tag, o_ in runs.items():
            if o_.usable[rr, 54:75].any():
                xy = o_.xy_px[rr]
                dx = xy[[58, 62, 66, 70, 74], 0].mean() - xy[54, 0]
                e[tag] = {'R_assigned': True, 'finger_direction': 'RIGHT' if dx > 0 else 'LEFT', 'dx': round(float(dx), 1)}
            else:
                e[tag] = {'R_assigned': False}
        sen[f'row_{rr}'] = e
    sen_changes = [{'row': r, 'side': 'LR'[s], 'kind_vs_recovery': k}
                   for r, s, k in diff_slots(runs['recovery'].xy_px, runs['recovery'].usable,
                                             runs['arm'].xy_px, runs['arm'].usable)]

    def st(clip, frame, side):
        fid, bb, cc, aa = per[clip]
        i = fid[frame]
        return {'baseline': bool(bb[i, rc.SLOT[side]].any()), 'recovery': bool(cc[i, rc.SLOT[side]].any()),
                'arm': bool(aa[i, rc.SLOT[side]].any())}
    q_err = [dict(clip=c_, frame=f, side='LR'[s], **st(c_, f, s)) for c_, f, s in rc.ERRORS]
    q_keep = [dict(clip=c_, frame=f, side='LR'[s], **st(c_, f, s)) for c_, f, s in rc.KEEP]
    joe = {f: {'L': st('NIKL_JOESONGHADA_01', f, 0), 'R': st('NIKL_JOESONGHADA_01', f, 1)}
           for f in range(60, 76) if f in per['NIKL_JOESONGHADA_01'][0]}

    tot = lambda key, cat: sum(r[key][cat] for r in rows)
    g = {k: {'events': len(v), 'median': float(np.median(v)), 'max': int(max(v))} for k, v in gaps.items()}
    out = {'schema': 'wis_arm_consistency_result_v1', 'at_kst': '2026-09-26',
           'plan_sha256': sha(H / 'FROZEN_PLAN.json'), 'candidate_sha256': sha(H / 'arm_landmarker.py'),
           'operating_path_changed': False, 'coefficient_search': 0, 'new_coefficients': 0,
           'control_recovery_reproduced': {'clips_ok': sum(control_ok), 'clips': len(control_ok)},
           'totals_vs_baseline': {k: {c_: tot(key, c_) for c_ in ('KEPT_SAME', 'CHANGED_CONTENT', 'LOST', 'NEW', 'SWAPPED')}
                                  for k, key in (('recovery', 'recov_vs_base'), ('arm', 'arm_vs_base'))},
           'sides_total': {k: sum(r['sides'][k] for r in rows) for k in ('baseline', 'recovery', 'arm')},
           'changes_vs_recovery': changes, 'sen0354': sen, 'sen0354_changes_vs_recovery': sen_changes,
           'confirmed_errors': q_err, 'confirmed_normals': q_keep, 'joesonghada_f60_75': joe,
           'gaps': g, 'nikl': rows, 'seconds': round(time.time() - started, 1)}
    (H / 'RESULT.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('\ncontrol recovery reproduced', out['control_recovery_reproduced'])
    print('totals', out['totals_vs_baseline'], out['sides_total'])
    print('changes vs recovery', len(changes), '| SEN', sen, sen_changes)
    print('errors still prevented', sum(not x['arm'] for x in q_err), '/8 | normals kept', sum(x['arm'] for x in q_keep), '/8')
    print('gaps', g)
    print('scores', [(r['clip'][5:], r['score_arm'].get('correct') if isinstance(r['score_arm'], dict) else r['score_arm'],
                      r['score_arm'].get('source') if isinstance(r['score_arm'], dict) else '') for r in rows])


if __name__ == '__main__':
    main()
