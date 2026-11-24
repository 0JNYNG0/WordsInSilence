"""Three-way comparison: baseline (stored cache) / original candidate / recovery candidate.

KEPT is decided by coordinate correspondence, never by "both slots are non-empty".
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
OLD = L / 'ops' / 'temporal_assignment_candidate_20260926'
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
sys.path.insert(0, str(H))
sys.path.insert(0, str(OLD))
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels, collate             # noqa: E402
from pilot.web_input import to_native_like, active_span    # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402
from temporal_landmarker import TemporalLandmarker         # noqa: E402
from recovery_landmarker import RecoveryLandmarker         # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
CACHE = L / 'cache' / '2683eadbc5f1c588'
SEN = L / 'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
SLOT = {0: slice(33, 54), 1: slice(54, 75)}
PALM = (0, 5, 9, 13, 17)
NIKL_LABEL = {'ALDA': '알다', 'GAMSAHAMNIDA': '감사합니다', 'GAM': '감', 'GONGBU': '공부',
              'INSA': '인사', 'JADA': '자다', 'JOESONGHADA': '죄송하다', 'JOTA': '좋다',
              'KKEUDEOGIDA': None, 'MAEKJU': '맥주', 'MEOKDA': '먹다', 'MOREUDA': '모르다',
              'MUL': '물', 'NA': '나', 'NEO': '너', 'PPANG': '빵', 'SARANG': '사랑',
              'SILTA': '싫다', 'UNDONG': '운동'}
ERRORS = [('NIKL_MUL_01', 59, 0), ('NIKL_NEO_01', 56, 0), ('NIKL_MOREUDA_01', 61, 0),
          ('NIKL_MEOKDA_01', 61, 0), ('NIKL_MAEKJU_01', 21, 0), ('NIKL_JOESONGHADA_01', 25, 0),
          ('NIKL_UNDONG_01', 67, 1), ('NIKL_GAM_01', 25, 0)]
KEEP = [('NIKL_GAMSAHAMNIDA_01', 57, 0), ('NIKL_INSA_01', 29, 0), ('NIKL_PPANG_01', 35, 1),
        ('NIKL_SILTA_01', 30, 1), ('NIKL_JOESONGHADA_01', 63, 0), ('NIKL_SARANG_01', 39, 0),
        ('NIKL_NA_01', 32, 0), ('NIKL_JOESONGHADA_01', 15, 0)]


def content(xy, usa, r, s):
    """the observation in one slot, or None"""
    if not usa[r, SLOT[s]].any():
        return None
    b = 33 if s == 0 else 54
    return xy[r, [b + j for j in PALM]]


def same(a, b):
    return a is not None and b is not None and np.allclose(a, b, atol=1.0)


def decompose(bxy, bus, cxy, cus):
    """coordinate-based: KEPT_SAME / CHANGED_CONTENT / LOST / NEW / SWAPPED"""
    t = {'KEPT_SAME': 0, 'CHANGED_CONTENT': 0, 'LOST': 0, 'NEW': 0, 'SWAPPED': 0}
    lost_slots, new_slots = [], []
    for r in range(bxy.shape[0]):
        bc = {s: content(bxy, bus, r, s) for s in (0, 1)}
        cc = {s: content(cxy, cus, r, s) for s in (0, 1)}
        swapped = set()
        for s in (0, 1):
            o = 1 - s
            if same(bc[s], cc[o]) and same(bc[o], cc[s]):
                swapped.add(s)
        for s in (0, 1):
            if s in swapped:
                t['SWAPPED'] += 1
                continue
            if bc[s] is not None and cc[s] is not None:
                t['KEPT_SAME' if same(bc[s], cc[s]) else 'CHANGED_CONTENT'] += 1
            elif bc[s] is not None:
                o = 1 - s
                if same(bc[s], cc[o]):
                    t['SWAPPED'] += 1
                else:
                    t['LOST'] += 1; lost_slots.append((r, s))
            elif cc[s] is not None:
                o = 1 - s
                if same(cc[s], bc[o]):
                    t['SWAPPED'] += 1
                else:
                    t['NEW'] += 1; new_slots.append((r, s))
    return t, lost_slots, new_slots


def gaps(us):
    st = np.stack([us[:, 33:54].any(1), us[:, 54:75].any(1)], 1)
    g = []
    for s in (0, 1):
        on = st[:, s]; r = 1
        while r < len(on):
            if on[r - 1] and not on[r]:
                q = r
                while q < len(on) and not on[q]:
                    q += 1
                if q < len(on):
                    g.append(q - r)
                r = q
            r += 1
    return g


def channels(xy, usa, ts):
    nat, pres = to_native_like(xy, usa)
    sp = active_span(pres)
    if sp is None:
        return None
    lo, hi = sp
    return build_channels(nat[lo:hi + 1], pres[lo:hi + 1], ts[lo:hi + 1])


def score(model, classes, clip, label):
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(label)
    order = sorted(range(93), key=lambda j: -float(logit[j]))
    rival = next(j for j in order if j != yi)
    return {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
            'margin': round(float(logit[yi] - logit[rival]), 5)}


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS'
    stored_old = {r['clip']: r for r in json.loads((OLD / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}
    ck = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    started = time.time()

    rows, per, control_ok = [], {}, []
    for p in sorted(CACHE.glob('*.npz')):
        z = np.load(p, allow_pickle=False)
        src = json.loads((CACHE / f'{p.stem}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        with TemporalLandmarker(CONTRACT) as lm:
            o = extract(src, lm, p.stem)
        with RecoveryLandmarker(CONTRACT) as lm:
            c = extract(src, lm, p.stem)
            rs, stt = dict(lm.reasons), dict(lm.stats)
        bxy, bus = z['xy_px'], z['usable']
        d_ob, lo_ob, _ = decompose(bxy, bus, o.xy_px, o.usable)      # original vs baseline
        d_rb, lo_rb, nw_rb = decompose(bxy, bus, c.xy_px, c.usable)  # recovery vs baseline
        d_ro, _, nw_ro = decompose(o.xy_px, o.usable, c.xy_px, c.usable)  # recovery vs original
        # of the slots the ORIGINAL lost, how many does recovery hold with the BASELINE content?
        lost_set = set(lo_ob)
        recovered_same = 0
        for (r, s) in lost_set:
            if same(content(bxy, bus, r, s), content(c.xy_px, c.usable, r, s)):
                recovered_same += 1
        got = int(np.stack([o.usable[:, 33:54].any(1), o.usable[:, 54:75].any(1)], 1).sum())
        control_ok.append(got == stored_old[p.stem]['sides_assigned']['candidate'])
        label = NIKL_LABEL[p.stem.replace('NIKL_', '').rsplit('_', 1)[0]]
        r_ = {'clip': p.stem, 'label': label,
              'sides': {'baseline': int(np.stack([bus[:, 33:54].any(1), bus[:, 54:75].any(1)], 1).sum()),
                        'original': got,
                        'recovery': int(np.stack([c.usable[:, 33:54].any(1), c.usable[:, 54:75].any(1)], 1).sum())},
              'orig_vs_base': d_ob, 'recov_vs_base': d_rb, 'recov_vs_orig': d_ro,
              'original_losses': len(lost_set), 'recovered_with_baseline_content': recovered_same,
              'reasons': rs, 'stats': stt,
              'gaps': {'baseline': gaps(bus), 'original': gaps(o.usable), 'recovery': gaps(c.usable)},
              'masks_changed_vs_original': bool(not np.array_equal(o.usable, c.usable))}
        if label is not None and r_['masks_changed_vs_original']:
            r_['score_baseline'] = score(model, classes, channels(bxy, bus, z['timestamp_us']), label)
            r_['score_recovery'] = score(model, classes, channels(c.xy_px, c.usable, c.timestamp_us), label)
            r_['regression_vs_baseline'] = bool(r_['score_baseline']['correct'] and
                                                not r_['score_recovery']['correct'])
        per[p.stem] = ({int(f): i for i, f in enumerate(z['frame_id'])}, bus, o.usable, c.usable)
        rows.append(r_)
        print(f"{p.stem:<24} sides {r_['sides']['baseline']:>4}/{r_['sides']['original']:>4}/"
              f"{r_['sides']['recovery']:<4} origLost {len(lost_set):>3} recovered "
              f"{recovered_same:>3} takeover {stt.get('recovered_takeover', 0):>3}", flush=True)

    # ---- SEN0354 ------------------------------------------------------------
    with TemporalLandmarker(CONTRACT) as lm:
        so = extract(str(SEN), lm, SEN.stem)
    with RecoveryLandmarker(CONTRACT) as lm:
        sc = extract(str(SEN), lm, SEN.stem)
        sst = dict(lm.stats)
    sen = {'stats': sst}
    for rr in (38, 40, 44):
        e = {}
        for tag, o_ in (('original', so), ('recovery', sc)):
            if o_.usable[rr, 54:75].any():
                xy = o_.xy_px[rr]
                d = xy[[54 + 4, 54 + 8, 54 + 12, 54 + 16, 54 + 20], 0].mean() - xy[54, 0]
                e[tag] = {'R_assigned': True, 'finger_direction': 'RIGHT' if d > 0 else 'LEFT',
                          'dx': round(float(d), 1)}
            else:
                e[tag] = {'R_assigned': False}
        sen[f'row_{rr}'] = e
    print(f"\nSEN0354 row38 {sen['row_38']}\n        row40 {sen['row_40']}\n        row44 {sen['row_44']}")

    def st3(clip, frame, side):
        fid, bb, oo, cc = per[clip]
        i = fid[frame]
        return (bool(bb[i, SLOT[side]].any()), bool(oo[i, SLOT[side]].any()),
                bool(cc[i, SLOT[side]].any()))
    q_err = [{'clip': c_, 'frame': f, 'side': 'LR'[s], 'baseline': st3(c_, f, s)[0],
              'original': st3(c_, f, s)[1], 'recovery': st3(c_, f, s)[2],
              'still_prevented': not st3(c_, f, s)[2]} for c_, f, s in ERRORS]
    q_keep = [{'clip': c_, 'frame': f, 'side': 'LR'[s], 'baseline': st3(c_, f, s)[0],
               'original': st3(c_, f, s)[1], 'recovery': st3(c_, f, s)[2],
               'no_new_loss_vs_original': st3(c_, f, s)[2] or not st3(c_, f, s)[1]}
              for c_, f, s in KEEP]

    tot = lambda key, cat: sum(r[key][cat] for r in rows)
    orig_lost = sum(r['original_losses'] for r in rows)
    rec_lost = tot('recov_vs_base', 'LOST')
    recovered = sum(r['recovered_with_baseline_content'] for r in rows)
    new_vs_orig = tot('recov_vs_orig', 'NEW')
    changed = tot('recov_vs_base', 'CHANGED_CONTENT')
    scored = [r for r in rows if 'score_recovery' in r]
    regs = sum(r['regression_vs_baseline'] for r in scored)
    gb = [g for r in rows for g in r['gaps']['baseline']]
    go = [g for r in rows for g in r['gaps']['original']]
    gr = [g for r in rows for g in r['gaps']['recovery']]

    passes = {
        'a_row38_right_hand_kept': (sen['row_38']['recovery'].get('R_assigned') and
                                    sen['row_38']['recovery'].get('finger_direction') == 'RIGHT'),
        'b_confirmed_errors_still_prevented': all(x['still_prevented'] for x in q_err),
        'c_no_new_normal_control_loss': all(x['no_new_loss_vs_original'] for x in q_keep),
        'd_losses_down_and_content_matches': rec_lost < orig_lost and recovered == (orig_lost - rec_lost),
        'e_no_recognition_regression': regs == 0,
    }
    print('\n--- pass conditions ---')
    for k, v in passes.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f"\n1. recovered: original lost {orig_lost} -> recovery lost {rec_lost}; "
          f"{recovered} recovered slots carry the BASELINE content")
    print(f"   recovery takeovers used: {sum(r['stats'].get('recovered_takeover', 0) for r in rows)}")
    print(f"   NEW vs original (content not in baseline): {new_vs_orig}   CHANGED_CONTENT vs baseline: {changed}")
    print(f"2. gap length frames  baseline med {np.median(gb):.1f}/max {max(gb)}  "
          f"original med {np.median(go):.1f}/max {max(go)}  recovery med {np.median(gr):.1f}/max {max(gr)}")
    print(f"   gap events {len(gb)} / {len(go)} / {len(gr)}")
    print(f"3. confirmed errors still prevented {sum(x['still_prevented'] for x in q_err)}/8 | "
          f"normal controls with no new loss {sum(x['no_new_loss_vs_original'] for x in q_keep)}/8")
    print(f"4. affected clips scored {len(scored)} | regressions vs baseline {regs}")
    print(f"\ncontrol: original reproduced its stored totals on {sum(control_ok)}/{len(control_ok)} clips")

    out = {'schema': 'wis_temporal_recovery_result_v1', 'at_kst': '2026-09-26',
           'plan_sha256': sha(H / 'FROZEN_PLAN.json'),
           'change': 'a live track no longer blocks re-initialisation on its own; takeover requires '
                     'the existing initialisation evidence',
           'operating_path_changed': False, 'coefficient_search': 0, 'interpolation': 0,
           'original_candidate_file_untouched': True,
           'nikl': rows, 'sen0354': sen,
           'Q1_recovery': {'original_lost_vs_baseline': orig_lost, 'recovery_lost_vs_baseline': rec_lost,
                           'recovered_with_baseline_content': recovered,
                           'takeovers': sum(r['stats'].get('recovered_takeover', 0) for r in rows),
                           'new_vs_original': new_vs_orig, 'changed_content_vs_baseline': changed},
           'Q2_reconnection': {'gap_events': {'baseline': len(gb), 'original': len(go), 'recovery': len(gr)},
                               'median_frames': {'baseline': float(np.median(gb)), 'original': float(np.median(go)),
                                                 'recovery': float(np.median(gr))},
                               'max_frames': {'baseline': int(max(gb)), 'original': int(max(go)),
                                              'recovery': int(max(gr))}},
           'Q3_errors_and_normals': {'confirmed_errors': q_err, 'confirmed_normals': q_keep},
           'Q4_recognition_affected_clips_only': {'scored': len(scored), 'regressions': regs,
                                                  'clips': [r['clip'] for r in scored]},
           'control_original_reproduced': {'clips_ok': sum(control_ok), 'clips': len(control_ok)},
           'pass_conditions': passes, 'all_passed': all(passes.values()),
           'seconds': round(time.time() - started, 1)}
    with (H / 'RESULT.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\nall_passed: {out['all_passed']} | {out['seconds']} s")


if __name__ == '__main__':
    main()
