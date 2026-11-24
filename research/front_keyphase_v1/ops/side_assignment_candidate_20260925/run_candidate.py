"""Isolated side-assignment candidate: mutual-nearest assignment.

A hand is bound to a side only if that side's pose wrist is also the hand's OWN nearest usable
wrist. Otherwise it stays unassigned - the path the code already has. Nothing is discarded on a
distance threshold, and closeness/overlap is never the trigger.

The operating Landmarker is NOT modified. The candidate is a subclass used only here.
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels, collate             # noqa: E402
from pilot.web_input import to_native_like, active_span    # noqa: E402
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
CACHE = L / 'cache' / '2683eadbc5f1c588'
NIKL_LABEL = {'ALDA': '알다', 'GAMSAHAMNIDA': '감사합니다', 'GAM': '감', 'GONGBU': '공부',
              'INSA': '인사', 'JADA': '자다', 'JOESONGHADA': '죄송하다', 'JOTA': '좋다',
              'KKEUDEOGIDA': None, 'MAEKJU': '맥주', 'MEOKDA': '먹다', 'MOREUDA': '모르다',
              'MUL': '물', 'NA': '나', 'NEO': '너', 'PPANG': '빵', 'SARANG': '사랑',
              'SILTA': '싫다', 'UNDONG': '운동'}


class MutualNearest(Landmarker):
    """Only change: a greedy pair is kept when the side is the hand's own nearest usable wrist."""

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        out = super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        from wis_lab.core.observation import POSE_L_WRIST, POSE_R_WRIST
        ref = {s: pose_xy[i] for s, i in ((0, POSE_L_WRIST), (1, POSE_R_WRIST)) if pose_usable[i]}
        if len(ref) < 2:
            return out                       # one wrist or none: out of scope, untouched
        n = len(hand_res.hand_landmarks)
        wrists = np.array([[hand_res.hand_landmarks[i][0].x * w,
                            hand_res.hand_landmarks[i][0].y * h] for i in range(n)])
        fixed = {}
        for i, (side, score) in out.items():
            if side is None:
                fixed[i] = (side, score)
                continue
            nearest = min(ref, key=lambda s: float(np.linalg.norm(wrists[i] - ref[s])))
            fixed[i] = (side, score) if nearest == side else (None, score)
        return fixed


def obs_for(video, cls):
    with cls(CONTRACT) as lm:
        return extract(str(video), lm, pathlib.Path(video).stem)


def channels(xy, usable, ts):
    nat, pres = to_native_like(xy, usable)
    sp = active_span(pres)
    if sp is None:
        return None
    lo, hi = sp
    return build_channels(nat[lo:hi + 1], pres[lo:hi + 1], ts[lo:hi + 1])


def score(model, classes, clip, label, idx18):
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(label)
    order = sorted(range(93), key=lambda j: -float(logit[j]))
    rival = next(j for j in order if j != yi)
    return {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
            'true_rank': order.index(yi) + 1,
            'margin': round(float(logit[yi] - logit[rival]), 5)}


def hand_state(usable):
    """per frame: which sides carry a hand."""
    return np.stack([usable[:, 33:54].any(axis=1), usable[:, 54:75].any(axis=1)], axis=1)


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_ANY_RESULT'
    ck = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    from wis_lab.service.neural import PROJECT_18
    idx18 = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]

    # cases frozen before results: (clip, frame_id, side_index, expectation)
    ERRORS = [('NIKL_MUL_01', 59, 0), ('NIKL_NEO_01', 56, 0), ('NIKL_MOREUDA_01', 61, 0),
              ('NIKL_MEOKDA_01', 61, 0), ('NIKL_MAEKJU_01', 21, 0), ('NIKL_JOESONGHADA_01', 25, 0),
              ('NIKL_UNDONG_01', 67, 1), ('NIKL_GAM_01', 25, 0)]
    KEEP = [('NIKL_GAMSAHAMNIDA_01', 57, 0), ('NIKL_INSA_01', 29, 0), ('NIKL_PPANG_01', 35, 1),
            ('NIKL_SILTA_01', 30, 1), ('NIKL_JOESONGHADA_01', 63, 0), ('NIKL_SARANG_01', 39, 0),
            ('NIKL_NA_01', 32, 0), ('NIKL_JOESONGHADA_01', 15, 0)]

    started, rows, per_clip = time.time(), [], {}
    for p in sorted(CACHE.glob('*.npz')):
        src = json.loads((CACHE / f'{p.stem}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        b, c = obs_for(src, Landmarker), obs_for(src, MutualNearest)
        assert np.array_equal(b.timestamp_us, c.timestamp_us)
        hb, hc = hand_state(b.usable), hand_state(c.usable)
        lost = np.argwhere(hb & ~hc)                       # assigned before, unassigned now
        gained = np.argwhere(~hb & hc)
        fid = {int(f): i for i, f in enumerate(b.frame_id)}
        label = NIKL_LABEL[p.stem.replace('NIKL_', '').rsplit('_', 1)[0]]
        r = {'clip': p.stem, 'label': label, 'frames': int(b.xy_px.shape[0]),
             'hands_unassigned_by_candidate': int(len(lost)),
             'hands_newly_assigned': int(len(gained)),
             'lost_rows': [[int(x), int(y)] for x, y in lost],
             'landmarks_identical_where_both_assigned': bool(np.array_equal(
                 np.nan_to_num(b.xy_px[b.usable & c.usable], nan=-1.0),
                 np.nan_to_num(c.xy_px[b.usable & c.usable], nan=-1.0)))}
        if label is not None:
            sb = score(model, classes, channels(b.xy_px, b.usable, b.timestamp_us), label, idx18)
            sc = score(model, classes, channels(c.xy_px, c.usable, c.timestamp_us), label, idx18)
            r.update(baseline=sb, candidate=sc,
                     gain=bool(sc['correct'] and not sb['correct']),
                     regression=bool(sb['correct'] and not sc['correct']))
        rows.append(r)
        per_clip[p.stem] = (fid, hb, hc)
        print(f"{p.stem:<24} unassigned {r['hands_unassigned_by_candidate']:>3}  "
              f"newly assigned {r['hands_newly_assigned']:>2}"
              + (f"  {r['baseline']['pred']:<8}{r['baseline']['margin']:+7.2f} -> "
                 f"{r['candidate']['pred']:<8}{r['candidate']['margin']:+7.2f}" if label else "  NOT_IN_MODEL"),
              flush=True)

    # --- Q1 on the 8 confirmed errors, Q2a on the 8 confirmed keeps ----------
    def state(clip, frame_id, side):
        fid, hb, hc = per_clip[clip]
        i = fid[frame_id]
        return bool(hb[i, side]), bool(hc[i, side])
    q1 = [{'clip': c, 'frame': f, 'side': 'L' if s == 0 else 'R',
           'assigned_baseline': state(c, f, s)[0], 'assigned_candidate': state(c, f, s)[1],
           'fixed': state(c, f, s)[0] and not state(c, f, s)[1]} for c, f, s in ERRORS]
    q2a = [{'clip': c, 'frame': f, 'side': 'L' if s == 0 else 'R',
            'assigned_baseline': state(c, f, s)[0], 'assigned_candidate': state(c, f, s)[1],
            'kept': state(c, f, s)[0] == state(c, f, s)[1]} for c, f, s in KEEP]

    # --- independent set ----------------------------------------------------
    B = [('ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL01_F.mp4', '나', 3.334, 4.585),
         ('ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL02_F.mp4', '나', 3.887, 4.840),
         ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL01_F.mp4', '죄송하다', 1.588, 2.94),
         ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL02_F.mp4', '죄송하다', 1.821, 3.001)]
    brows = []
    for rel, label, s0, s1 in B:
        v = L / rel
        b, c = obs_for(v, Landmarker), obs_for(v, MutualNearest)
        t = (b.timestamp_us - b.timestamp_us[0]) / 1e6
        sel = np.flatnonzero((t >= s0) & (t < s1))
        hb, hc = hand_state(b.usable[sel]), hand_state(c.usable[sel])
        sb = score(model, classes, channels(b.xy_px[sel], b.usable[sel], b.timestamp_us[sel]), label, idx18)
        sc = score(model, classes, channels(c.xy_px[sel], c.usable[sel], c.timestamp_us[sel]), label, idx18)
        brows.append({'clip': v.stem, 'label': label,
                      'hands_unassigned_by_candidate': int((hb & ~hc).sum()),
                      'baseline': sb, 'candidate': sc,
                      'gain': bool(sc['correct'] and not sb['correct']),
                      'regression': bool(sb['correct'] and not sc['correct'])})
        print(f"{v.stem:<24} unassigned {brows[-1]['hands_unassigned_by_candidate']:>3}  "
              f"{sb['pred']:<8}{sb['margin']:+7.2f} -> {sc['pred']:<8}{sc['margin']:+7.2f}", flush=True)

    scored = [r for r in rows + brows if 'baseline' in r]
    conf_frames = {(c, f) for c, f, _ in ERRORS}
    outside = sum(len([1 for i, s in r['lost_rows']]) for r in rows if True)   # recomputed below
    outside = 0
    for r in rows:
        fid, _, _ = per_clip[r['clip']]
        inv = {i: f for f, i in fid.items()}
        for i, _s in r['lost_rows']:
            if (r['clip'], inv[i]) not in conf_frames:
                outside += 1
    tot_lost = sum(r['hands_unassigned_by_candidate'] for r in rows + brows)
    gains = sum(r['gain'] for r in scored); regs = sum(r['regression'] for r in scored)
    passed = all(x['fixed'] for x in q1) and all(x['kept'] for x in q2a) and regs == 0

    print('\n--- Q1 confirmed side-assignment errors ---')
    for x in q1:
        print(f"  {x['clip']:<24}f{x['frame']:<4}{x['side']}  assigned {x['assigned_baseline']} -> "
              f"{x['assigned_candidate']}   {'FIXED' if x['fixed'] else 'NOT FIXED'}")
    print('--- Q2a confirmed normal / resting / under-observation ---')
    for x in q2a:
        print(f"  {x['clip']:<24}f{x['frame']:<4}{x['side']}  assigned {x['assigned_baseline']} -> "
              f"{x['assigned_candidate']}   {'KEPT' if x['kept'] else 'LOST'}")
    print(f"\nQ1 fixed {sum(x['fixed'] for x in q1)}/8 | Q2a kept {sum(x['kept'] for x in q2a)}/8")
    print(f"hands unassigned in total {tot_lost} (automatic count, not verified errors); "
          f"{outside} of them outside the 8 confirmed frames")
    print(f"scored {len(scored)} clips | gains {gains} | regressions {regs}")

    out = {'schema': 'wis_side_assignment_candidate_result_v1', 'at_kst': '2026-09-25',
           'plan_sha256': sha(H / 'FROZEN_PLAN.json'),
           'candidate': 'mutual-nearest assignment; unassigned when the side is not the hand own '
                        'nearest usable wrist. No distance filter, nothing discarded.',
           'model': '83212868 frozen', 'operating_path_changed': False,
           'nikl': rows, 'independent': brows,
           'Q1_confirmed_errors': q1, 'Q1_fixed': sum(x['fixed'] for x in q1),
           'Q2a_confirmed_keeps': q2a, 'Q2a_kept': sum(x['kept'] for x in q2a),
           'Q2b_hands_unassigned_total': tot_lost,
           'Q2b_outside_confirmed_frames': outside,
           'Q2b_status': 'AUTOMATIC COUNT. Only the 8 confirmed frames are verified errors; every '
                         'other unassignment is unconfirmed and is reported as risk, not as a fix.',
           'Q2c_scored_clips': len(scored), 'Q2c_gains': gains, 'Q2c_regressions': regs,
           'pass_condition_met': bool(passed),
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'RESULT.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\npass_condition_met: {passed} | {out['seconds']} s")


if __name__ == '__main__':
    main()
