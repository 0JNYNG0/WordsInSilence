"""Limited comparison of the causal palm-track candidate against the stored baseline.

Baseline results are reused; only material the candidate can affect is processed.
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
sys.path.insert(0, str(H))
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels, collate             # noqa: E402
from pilot.web_input import to_native_like, active_span    # noqa: E402
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402
from temporal_landmarker import TemporalLandmarker         # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
CACHE = L / 'cache' / '2683eadbc5f1c588'
SEN = L / 'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
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


def side_state(o):
    return np.stack([o.usable[:, 33:54].any(1), o.usable[:, 54:75].any(1)], 1)


def phantom(o):
    """a side assigned for >=2 consecutive frames while its own pose-wrist distance stays > 0.8 sw"""
    xy, usa = o.xy_px, o.usable
    sw = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    sh = usa[:, 11] & usa[:, 12]
    bad = np.zeros((xy.shape[0], 2), bool)
    for s, (hb, pw) in enumerate(((33, 15), (54, 16))):
        ok = usa[:, hb] & usa[:, pw] & sh
        i = np.flatnonzero(ok)
        if i.size:
            bad[i, s] = np.linalg.norm(xy[i, hb] - xy[i, pw], axis=1) / sw[i] > 0.8
    run = 0
    for r in range(bad.shape[0]):
        for s in (0, 1):
            pass
    total = 0
    for s in (0, 1):
        c = 0
        for r in range(bad.shape[0]):
            c = c + 1 if bad[r, s] else 0
            if c >= 2:
                total += 1
    return int(total)


def reconnect(base, cand):
    """after a side stops being assigned, how many frames until it is assigned again"""
    out = {}
    for name, o in (('baseline', base), ('candidate', cand)):
        st = side_state(o)
        gaps = []
        for s in (0, 1):
            on = st[:, s]
            r = 1
            while r < len(on):
                if on[r - 1] and not on[r]:
                    q = r
                    while q < len(on) and not on[q]:
                        q += 1
                    if q < len(on):
                        gaps.append(q - r)
                    r = q
                r += 1
        out[name] = {'gaps': len(gaps), 'median_frames': (float(np.median(gaps)) if gaps else None),
                     'max_frames': (int(max(gaps)) if gaps else None)}
    return out


def channels(o):
    nat, pres = to_native_like(o.xy_px, o.usable)
    sp = active_span(pres)
    if sp is None:
        return None
    lo, hi = sp
    return build_channels(nat[lo:hi + 1], pres[lo:hi + 1], o.timestamp_us[lo:hi + 1])


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
    assert plan['status'] == 'FROZEN_BEFORE_ANY_RESULT'
    ck = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    started = time.time()

    # ---- SEN0354: the counterexample and the two prevention cases -----------
    with Landmarker(CONTRACT) as lm:
        sb = extract(str(SEN), lm, SEN.stem)
    with TemporalLandmarker(CONTRACT) as lm:
        sc = extract(str(SEN), lm, SEN.stem)
        sen_stats = dict(lm.stats)
    bs, cs = side_state(sb), side_state(sc)
    sen = {'clip': SEN.stem, 'stats': sen_stats,
           'row_38_R_assigned': {'baseline': bool(bs[38, 1]), 'candidate': bool(cs[38, 1])},
           'row_40_R_assigned': {'baseline': bool(bs[40, 1]), 'candidate': bool(cs[40, 1])},
           'row_44_R_assigned': {'baseline': bool(bs[44, 1]), 'candidate': bool(cs[44, 1])},
           'phantom_frames': {'baseline': phantom(sb), 'candidate': phantom(sc)},
           'reconnect': reconnect(sb, sc),
           'sides_assigned_total': {'baseline': int(bs.sum()), 'candidate': int(cs.sum())}}
    # did row 38's R keep the RIGHT-pointing detection?
    for tag, o in (('baseline', sb), ('candidate', sc)):
        xy, usa = o.xy_px[38], o.usable[38]
        if usa[54]:
            d = xy[[54 + 4, 54 + 8, 54 + 12, 54 + 16, 54 + 20], 0].mean() - xy[54, 0]
            sen[f'row_38_R_finger_direction_{tag}'] = ('RIGHT' if d > 0 else 'LEFT', round(float(d), 1))
        else:
            sen[f'row_38_R_finger_direction_{tag}'] = ('not assigned', None)
    print(f"SEN0354  row38 R {sen['row_38_R_assigned']}   row40 {sen['row_40_R_assigned']}   "
          f"row44 {sen['row_44_R_assigned']}", flush=True)

    # ---- public NIKL -------------------------------------------------------
    rows, per = [], {}
    for p in sorted(CACHE.glob('*.npz')):
        src = json.loads((CACHE / f'{p.stem}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        with Landmarker(CONTRACT) as lm:
            b = extract(src, lm, p.stem)
        with TemporalLandmarker(CONTRACT) as lm:
            c = extract(src, lm, p.stem)
            st = dict(lm.stats)
        bs2, cs2 = side_state(b), side_state(c)
        fid = {int(f): i for i, f in enumerate(b.frame_id)}
        per[p.stem] = (fid, bs2, cs2)
        label = NIKL_LABEL[p.stem.replace('NIKL_', '').rsplit('_', 1)[0]]
        r = {'clip': p.stem, 'label': label, 'stats': st,
             'sides_assigned': {'baseline': int(bs2.sum()), 'candidate': int(cs2.sum())},
             'phantom_frames': {'baseline': phantom(b), 'candidate': phantom(c)},
             'reconnect': reconnect(b, c)}
        if label is not None:
            sbb = score(model, classes, channels(b), label)
            scc = score(model, classes, channels(c), label)
            r.update(baseline=sbb, candidate=scc,
                     gain=bool(scc['correct'] and not sbb['correct']),
                     regression=bool(sbb['correct'] and not scc['correct']))
        rows.append(r)
        print(f"{p.stem:<24} sides {r['sides_assigned']['baseline']:>4}->{r['sides_assigned']['candidate']:<4}"
              f" phantom {r['phantom_frames']['baseline']:>3}->{r['phantom_frames']['candidate']:<3}"
              + (f"  {r['baseline']['pred']:<8}{r['baseline']['margin']:+7.2f} -> "
                 f"{r['candidate']['pred']:<8}{r['candidate']['margin']:+7.2f}"
                 + ('  REGRESSION' if r['regression'] else '  GAIN' if r['gain'] else '')
                 if label else '  NOT_IN_MODEL'), flush=True)

    def st(clip, frame, side):
        fid, bb, cc = per[clip]
        i = fid[frame]
        return bool(bb[i, side]), bool(cc[i, side])
    q_err = [{'clip': c, 'frame': f, 'side': 'LR'[s], 'baseline': st(c, f, s)[0],
              'candidate': st(c, f, s)[1], 'prevented': not st(c, f, s)[1]} for c, f, s in ERRORS]
    q_keep = [{'clip': c, 'frame': f, 'side': 'LR'[s], 'baseline': st(c, f, s)[0],
               'candidate': st(c, f, s)[1], 'kept': st(c, f, s)[0] == st(c, f, s)[1]}
              for c, f, s in KEEP]

    scored = [r for r in rows if 'baseline' in r]
    gains = sum(r['gain'] for r in scored); regs = sum(r['regression'] for r in scored)
    ph_new = sum(max(0, r['phantom_frames']['candidate'] - r['phantom_frames']['baseline'])
                 for r in rows) + max(0, sen['phantom_frames']['candidate'] - sen['phantom_frames']['baseline'])
    passes = {
        'a_row38_right_hand_kept': sen['row_38_R_assigned']['candidate'] and
                                   sen['row_38_R_finger_direction_candidate'][0] == 'RIGHT',
        'b_row40_44_duplicate_not_assigned': (not sen['row_40_R_assigned']['candidate'] and
                                              not sen['row_44_R_assigned']['candidate']),
        'c_confirmed_errors_prevented': all(x['prevented'] for x in q_err),
        'd_confirmed_normals_kept': all(x['kept'] for x in q_keep),
        'e_no_new_phantom_tracking': ph_new == 0,
        'f_no_recognition_regression': regs == 0,
    }
    print('\n--- pass conditions ---')
    for k, v in passes.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'\nQ1 confirmed errors prevented {sum(x["prevented"] for x in q_err)}/8 | '
          f'Q2 confirmed normals kept {sum(x["kept"] for x in q_keep)}/8')
    print(f'Q3 new phantom frames {ph_new} | Q4 reconnect: see RESULT.json')
    print(f'recognition: scored {len(scored)} | gains {gains} | regressions {regs}')

    out = {'schema': 'wis_temporal_assignment_result_v1', 'at_kst': '2026-09-26',
           'plan_sha256': sha(H / 'FROZEN_PLAN.json'),
           'candidate': 'causal palm-track assignment', 'operating_path_changed': False,
           'model': '83212868 frozen', 'sen0354': sen, 'nikl': rows,
           'Q1_confirmed_errors': q_err, 'Q2_confirmed_normals': q_keep,
           'Q3_new_phantom_frames': ph_new,
           'Q_recognition': {'scored': len(scored), 'gains': gains, 'regressions': regs,
                             'baseline_correct': sum(r['baseline']['correct'] for r in scored),
                             'candidate_correct': sum(r['candidate']['correct'] for r in scored)},
           'pass_conditions': passes, 'all_passed': all(passes.values()),
           'training_runs': 0, 'downloads': 0, 'user_data': 0, 'interpolation': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'RESULT.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\nall_passed: {out['all_passed']} | {out['seconds']} s")


if __name__ == '__main__':
    main()
