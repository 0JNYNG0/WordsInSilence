"""Three-arm comparison: baseline (stored operating cache) / arm (frozen) / start (this candidate).

Correctness comes from existing video verdicts; coordinate sameness is only reported. Every slot that
changes vs arm is listed; those outside the 22 video-judged normal losses are reviewed afterwards.
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
ARM = H.parent / 'arm_consistency_candidate_20260926'
REC = H.parent / 'temporal_recovery_candidate_20260926'
OLD = H.parent / 'temporal_assignment_candidate_20260926'
sys.path[:0] = [str(H), str(ARM), str(REC)]
import slot_review as sr                                          # noqa: E402
import importlib.util                                             # noqa: E402
_sp = importlib.util.spec_from_file_location('arm_run_compare', ARM / 'run_compare.py')
arm_rc = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(arm_rc)   # diff_slots, eq_input
from arm_landmarker import ArmLandmarker                          # noqa: E402
from start_landmarker import StartLandmarker                      # noqa: E402
rc = sr._rc
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
MASKED5 = {'NIKL_ALDA_01': 'left', 'NIKL_GAM_01': 'left', 'NIKL_JADA_01': 'left',
           'NIKL_NA_01': 'left', 'NIKL_NEO_01': 'left'}


def masked_input(xy, us, ts, side):
    """reviewed-masked condition, exactly as tools_visibility_experiment builds it: intact span,
    then the reviewed hand marked absent in the native arrays, then build_channels."""
    sys.path.insert(0, str(rc.L))
    import tools_visibility_experiment as tve
    nat, pres = rc.to_native_like(np.asarray(xy, np.float32), np.asarray(us, bool))
    sp = rc.active_span(pres)
    lo, hi = sp
    nat2, pr2 = tve.mask_native(nat[lo:hi + 1], pres[lo:hi + 1], side)
    return rc.build_channels(nat2, pr2, np.asarray(ts, np.int64)[lo:hi + 1])


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS'
    arm_stored = json.loads((ARM / 'RESULT.json').read_text(encoding='utf-8'))
    arm_rows = {r['clip']: r for r in arm_stored['nikl']}
    old_stored = {r['clip']: r for r in json.loads((OLD / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}
    loss = json.loads((ARM / 'loss_review' / 'LOSS_LIST.json').read_text(encoding='utf-8'))['rows']
    normal22 = {(r['clip'], r['frame'], r['side']) for r in loss if r['verdict'].startswith('NORMAL_HAND_LOST')}
    wrongside7 = {(r['clip'], r['frame'], r['side']) for r in loss if r['verdict'].startswith('WRONG_SIDE_BINDING')}
    assert len(normal22) == 22 and len(wrongside7) == 7
    ck = rc.L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = rc.u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    started = time.time()

    rows, changes, control_ok, gaps, per = [], [], [], {'baseline': [], 'arm': [], 'start': []}, {}
    masked5 = []
    for p in sorted(rc.CACHE.glob('*.npz')):
        clip = p.stem
        z = np.load(p, allow_pickle=False)
        src = json.loads((rc.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        a, _ = sr.run(ArmLandmarker, src, clip)
        with StartLandmarker(rc.CONTRACT) as lm:
            s_ = rc.extract(src, lm, clip)
            reasons, stats = dict(lm.reasons), dict(lm.stats)
        bxy, bus = z['xy_px'], z['usable']
        d_ab, _, _ = rc.decompose(bxy, bus, a.xy_px, a.usable)
        control_ok.append(d_ab == arm_rows[clip]['arm_vs_base'])          # frozen arm reproduced
        d_sb, _, _ = rc.decompose(bxy, bus, s_.xy_px, s_.usable)
        fids = list(map(int, z['frame_id']))
        for r, s, kind in arm_rc.diff_slots(a.xy_px, a.usable, s_.xy_px, s_.usable):
            k = (clip, fids[r], 'LR'[s])
            changes.append({'clip': clip, 'frame': fids[r], 'side': 'LR'[s], 'kind_vs_arm': kind,
                            'start_equals_baseline_slot': rc.same(sr.content(bxy, bus, r, s),
                                                                  sr.content(s_.xy_px, s_.usable, r, s)),
                            'in_normal22': k in normal22, 'in_wrongside7': k in wrongside7})
        for k_, us in (('baseline', bus), ('arm', a.usable), ('start', s_.usable)):
            gaps[k_] += rc.gaps(us)
        label = rc.NIKL_LABEL[clip.replace('NIKL_', '').rsplit('_', 1)[0]]
        S = rc.channels(s_.xy_px, s_.usable, s_.timestamp_us)
        A = rc.channels(a.xy_px, a.usable, a.timestamp_us)
        B = rc.channels(bxy, bus, z['timestamp_us'])
        same_as = [n for n, v in (('baseline', B), ('arm', A)) if arm_rc.eq_input(S, v)]
        row = {'clip': clip, 'label': label, 'start_vs_base': d_sb, 'arm_vs_base': d_ab,
               'sides': {'baseline': int(np.stack([bus[:, 33:54].any(1), bus[:, 54:75].any(1)], 1).sum()),
                         'arm': int(np.stack([a.usable[:, 33:54].any(1), a.usable[:, 54:75].any(1)], 1).sum()),
                         'start': int(np.stack([s_.usable[:, 33:54].any(1), s_.usable[:, 54:75].any(1)], 1).sum())},
                 'reasons': reasons, 'stats': stats, 'tcn_input_identical_to': same_as}
        if label is None:
            row['score_start'] = 'NOT_SCORABLE_label_not_in_vocabulary'
        elif 'baseline' in same_as:
            row['score_start'] = dict(old_stored[clip]['baseline'], source='reused: identical to baseline input')
        elif 'arm' in same_as:
            row['score_start'] = dict({k: v for k, v in arm_rows[clip]['score_arm'].items() if k != 'source'},
                                      source='reused: identical to arm input')
        else:
            row['score_start'] = dict(rc.score(model, classes, S, label), source='NEWLY_SCORED')
        if label is not None:
            row['score_baseline'] = old_stored[clip]['baseline']
            row['regression_vs_baseline'] = bool(row['score_baseline']['correct'] and not row['score_start']['correct'])
        if clip in MASKED5:
            mb = masked_input(bxy, bus, z['timestamp_us'], MASKED5[clip])
            ms = masked_input(s_.xy_px, s_.usable, s_.timestamp_us, MASKED5[clip])
            m = {'clip': clip, 'side': MASKED5[clip], 'start_equals_baseline_masked_input': arm_rc.eq_input(ms, mb)}
            if not m['start_equals_baseline_masked_input']:
                m['score_baseline_masked'] = rc.score(model, classes, mb, label)
                m['score_start_masked'] = rc.score(model, classes, ms, label)
            masked5.append(m)
        per[clip] = ({f: i for i, f in enumerate(fids)}, bus, a.usable, s_.usable)
        rows.append(row)
        print(f"{clip:<24} sides {row['sides']}  input={same_as or 'NEW'}  provisional "
              f"{stats.get('provisional_assigned', 0)}  refusals "
              f"{ {k: v for k, v in reasons.items() if k.startswith('provisional')} }", flush=True)

    # ---- SEN0354 -------------------------------------------------------------
    runs = {}
    for tag, cls in (('arm', ArmLandmarker), ('start', StartLandmarker)):
        with cls(rc.CONTRACT) as lm:
            runs[tag] = rc.extract(str(rc.SEN), lm, rc.SEN.stem)
    sen = {}
    for rr in (38, 40, 44):
        e = {}
        for tag, o_ in runs.items():
            if o_.usable[rr, 54:75].any():
                dx = o_.xy_px[rr][[58, 62, 66, 70, 74], 0].mean() - o_.xy_px[rr][54, 0]
                e[tag] = {'R_assigned': True, 'finger_direction': 'RIGHT' if dx > 0 else 'LEFT', 'dx': round(float(dx), 1)}
            else:
                e[tag] = {'R_assigned': False}
        sen[f'row_{rr}'] = e
    sen_changes = [{'row': r, 'side': 'LR'[s], 'kind_vs_arm': k}
                   for r, s, k in arm_rc.diff_slots(runs['arm'].xy_px, runs['arm'].usable,
                                                    runs['start'].xy_px, runs['start'].usable)]

    def st(clip, frame, side):
        fid, bb, aa, ss = per[clip]
        i = fid[frame]
        return {'baseline': bool(bb[i, rc.SLOT[side]].any()), 'arm': bool(aa[i, rc.SLOT[side]].any()),
                'start': bool(ss[i, rc.SLOT[side]].any())}
    q_err = [dict(clip=c_, frame=f, side='LR'[s], **st(c_, f, s)) for c_, f, s in rc.ERRORS]
    q_joe = {f: st('NIKL_JOESONGHADA_01', f, 0) for f in (67, 68, 72, 73)}
    recovered = [c for c in changes if c['in_normal22'] and c['kind_vs_arm'] == 'NEW' and c['start_equals_baseline_slot']]
    ws7 = [dict(clip=c_, frame=f, side=s, **st(c_, f, 0 if s == 'L' else 1)) for c_, f, s in sorted(wrongside7)]
    g = {k: {'events': len(v), 'median': float(np.median(v)), 'max': int(max(v))} for k, v in gaps.items()}
    out = {'schema': 'wis_start_provisional_result_v1', 'at_kst': '2026-09-26',
           'plan_sha256': sha(H / 'FROZEN_PLAN.json'), 'candidate_sha256': sha(H / 'start_landmarker.py'),
           'operating_path_changed': False, 'coefficient_search': 0, 'new_coefficients': 0,
           'control_arm_reproduced': {'clips_ok': sum(control_ok), 'clips': len(control_ok)},
           'normal22_recovered_correct_side_baseline_content': len(recovered),
           'normal22_recovered': [(c['clip'], c['frame'], c['side']) for c in recovered],
           'wrongside7_state': ws7, 'changes_vs_arm': changes,
           'changes_outside_normal22': [c for c in changes if not c['in_normal22']],
           'totals_vs_baseline': {k: {c_: sum(r[key][c_] for r in rows) for c_ in ('KEPT_SAME', 'CHANGED_CONTENT', 'LOST', 'NEW', 'SWAPPED')}
                                  for k, key in (('arm', 'arm_vs_base'), ('start', 'start_vs_base'))},
           'sides_total': {k: sum(r['sides'][k] for r in rows) for k in ('baseline', 'arm', 'start')},
           'sen0354': sen, 'sen0354_changes_vs_arm': sen_changes, 'confirmed_errors': q_err,
           'joesonghada_L_f67_68_72_73': q_joe, 'gaps': g, 'masked5': masked5, 'nikl': rows,
           'seconds': round(time.time() - started, 1)}
    (H / 'RESULT.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('\ncontrol arm reproduced', out['control_arm_reproduced'])
    print('normal22 recovered', len(recovered), '| changes vs arm', len(changes), '| outside normal22', len(out['changes_outside_normal22']))
    for c in out['changes_outside_normal22']:
        print('   OUTSIDE', c)
    print('wrongside7', [(w['clip'][5:], w['frame'], w['side'], w['start']) for w in ws7])
    print('totals', out['totals_vs_baseline'], out['sides_total'])
    print('SEN', sen, sen_changes)
    print('errors unassigned', sum(not x['start'] for x in q_err), '/8 | JOE L', q_joe)
    print('gaps', g)
    print('masked5', masked5)
    print('scores', [(r['clip'][5:], r['score_start'] if isinstance(r['score_start'], str) else
                      (r['score_start']['correct'], r['score_start']['source'])) for r in rows])


if __name__ == '__main__':
    main()
