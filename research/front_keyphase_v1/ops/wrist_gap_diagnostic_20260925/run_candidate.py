"""ONE comparison: baseline masks vs the frozen wrist-gap rule. Model frozen, operating path untouched."""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels, collate             # noqa: E402
from pilot.web_input import to_native_like, active_span    # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CACHE = L / 'cache' / '2683eadbc5f1c588'
SIDE = {'L': (33, 15, 54), 'R': (54, 16, 33)}
NIKL_LABEL = {'ALDA': '알다', 'GAMSAHAMNIDA': '감사합니다', 'GAM': '감', 'GONGBU': '공부',
              'INSA': '인사', 'JADA': '자다', 'JOESONGHADA': '죄송하다', 'JOTA': '좋다',
              'KKEUDEOGIDA': None, 'MAEKJU': '맥주', 'MEOKDA': '먹다', 'MOREUDA': '모르다',
              'MUL': '물', 'NA': '나', 'NEO': '너', 'PPANG': '빵', 'SARANG': '사랑',
              'SILTA': '싫다', 'UNDONG': '운동'}


def apply_rule(xy, usable, tau):
    """Clear a hand whose wrist sits tau shoulder-widths or more from its own pose wrist."""
    out = usable.copy()
    sw = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    sh = usable[:, 11] & usable[:, 12]
    removed = []
    for s, (hb, pw, ob) in SIDE.items():
        other_pw = SIDE['R' if s == 'L' else 'L'][1]
        for i in np.flatnonzero(usable[:, hb] & usable[:, pw] & sh):
            d = float(np.linalg.norm(xy[i, hb] - xy[i, pw]) / sw[i])
            if d < tau:
                continue
            out[i, hb:hb + 21] = False
            other = bool(usable[i, ob])
            gap = float(np.linalg.norm(xy[i, hb] - xy[i, ob]) / sw[i]) if other else None
            if other and gap < 0.4:
                kind = 'DUPLICATE'
            elif not other and not usable[i, other_pw]:
                kind = 'WRONG_SIDE_SINGLE'
            else:
                kind = 'NEITHER'
            removed.append({'row': int(i), 'side': s, 'd': round(d, 3),
                            'other_hand_usable': other,
                            'two_hand_gap': None if gap is None else round(gap, 3),
                            'kind': kind})
    return out, removed


def channels(xy, usable, ts):
    nat, pres = to_native_like(xy, usable)
    sp = active_span(pres)
    if sp is None:
        return None, None
    lo, hi = sp
    return build_channels(nat[lo:hi + 1], pres[lo:hi + 1], ts[lo:hi + 1]), (int(lo), int(hi))


def score(model, classes, clip, label, idx18):
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(label)
    out = {}
    for scope, inds in [('full93', list(range(93))), ('project18', idx18)]:
        order = sorted(inds, key=lambda j: -float(logit[j]))
        rival = next(j for j in order if j != yi)
        out[scope] = {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
                      'true_rank': order.index(yi) + 1,
                      'margin': round(float(logit[yi] - logit[rival]), 5)}
    return out


def kinds_of(removed):
    return {k: sum(1 for x in removed if x['kind'] == k)
            for k in ('DUPLICATE', 'WRONG_SIDE_SINGLE', 'NEITHER')}


def main():
    plan = json.loads((H / 'FROZEN_THRESHOLD.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_ANY_SCORE_IS_SEEN'
    tau = plan['tau']['value']
    ck = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ck) == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
    model, classes = u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    from wis_lab.service.neural import PROJECT_18
    idx18 = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]
    started, sets = time.time(), {}

    # ---- set A: public NIKL -------------------------------------------------
    rows = []
    for p in sorted(CACHE.glob('*.npz')):
        key = p.stem.replace('NIKL_', '').rsplit('_', 1)[0]
        label = NIKL_LABEL[key]
        z = np.load(p, allow_pickle=False)
        xy, usa, ts = z['xy_px'], z['usable'], z['timestamp_us']
        new, removed = apply_rule(xy, usa, tau)
        r = {'clip': p.stem, 'label': label, 'removed_hand_frames': len(removed),
             'removed_kinds': kinds_of(removed), 'removed_detail': removed}
        if label is None:
            r['scored'] = 'NOT_IN_MODEL'
        else:
            cb, spb = channels(xy, usa, ts)
            cc, spc = channels(xy, new, ts)
            sb = score(model, classes, cb, label, idx18)
            sc = score(model, classes, cc, label, idx18)
            r.update(baseline=sb['full93'], candidate=sc['full93'],
                     span_baseline=spb, span_candidate=spc,
                     gain=bool(sc['full93']['correct'] and not sb['full93']['correct']),
                     regression=bool(sb['full93']['correct'] and not sc['full93']['correct']))
        rows.append(r)
    sets['A_public_nikl'] = rows

    # ---- set B: independent AI Hub clips ------------------------------------
    from wis_lab.capture.landmarker import Landmarker
    from wis_lab.capture.video_source import extract
    contract = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
    B = [('ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL01_F.mp4', '나', 3.334, 4.585),
         ('ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL02_F.mp4', '나', 3.887, 4.840),
         ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL01_F.mp4', '죄송하다', 1.588, 2.94),
         ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL02_F.mp4', '죄송하다', 1.821, 3.001)]
    rows = []
    for rel, label, s0, s1 in B:
        v = L / rel
        with Landmarker(contract) as lm:
            obs = extract(str(v), lm, v.stem)
        t = (obs.timestamp_us - obs.timestamp_us[0]) / 1e6
        sel = np.flatnonzero((t >= s0) & (t < s1))
        xy, usa, ts = obs.xy_px[sel], obs.usable[sel], obs.timestamp_us[sel]
        new, removed = apply_rule(xy, usa, tau)
        cb, _ = channels(xy, usa, ts)
        cc, _ = channels(xy, new, ts)
        sb = score(model, classes, cb, label, idx18)
        sc = score(model, classes, cc, label, idx18)
        rows.append({'clip': v.stem, 'label': label, 'window_s': [s0, s1],
                     'removed_hand_frames': len(removed), 'removed_kinds': kinds_of(removed),
                     'removed_detail': removed,
                     'baseline': sb['full93'], 'candidate': sc['full93'],
                     'gain': bool(sc['full93']['correct'] and not sb['full93']['correct']),
                     'regression': bool(sb['full93']['correct'] and not sc['full93']['correct'])})
    sets['B_independent_aihub'] = rows

    scored = [r for rs in sets.values() for r in rs if isinstance(r.get('baseline'), dict)]
    tot_rm = sum(r['removed_hand_frames'] for rs in sets.values() for r in rs)
    kinds = {k: sum(r['removed_kinds'][k] for rs in sets.values() for r in rs)
             for k in ('DUPLICATE', 'WRONG_SIDE_SINGLE', 'NEITHER')}
    gains = sum(r['gain'] for r in scored)
    regs = sum(r['regression'] for r in scored)
    passed = regs == 0 and kinds['NEITHER'] == 0

    print(f"{'clip':<26}{'label':<7}{'rm':>4}  baseline                candidate")
    for rs in sets.values():
        for r in rs:
            if not isinstance(r.get('baseline'), dict):
                print(f"{r['clip']:<26}{'-':<7}{r['removed_hand_frames']:>4}  NOT_IN_MODEL")
                continue
            b, c = r['baseline'], r['candidate']
            flag = ' GAIN' if r['gain'] else ' REGRESSION' if r['regression'] else ''
            print(f"{r['clip']:<26}{r['label']:<7}{r['removed_hand_frames']:>4}  "
                  f"{b['pred']:<8}{b['margin']:+8.3f}      {c['pred']:<8}{c['margin']:+8.3f}{flag}")
    print(f"\nremoved hand-frames {tot_rm}  {kinds}")
    print(f"scored clips {len(scored)} | gains {gains} | regressions {regs} | "
          f"correct baseline {sum(r['baseline']['correct'] for r in scored)} -> "
          f"candidate {sum(r['candidate']['correct'] for r in scored)}")

    out = {'schema': 'wis_wrist_gap_candidate_result_v1', 'at_kst': '2026-09-25',
           'plan_sha256': sha(H / 'FROZEN_THRESHOLD.json'), 'tau': tau,
           'model': '83212868 frozen', 'operating_path_changed': False,
           'sets': sets, 'removed_hand_frames': tot_rm, 'removed_kinds': kinds,
           'scored_clips': len(scored), 'gains': gains, 'regressions': regs,
           'baseline_correct': sum(r['baseline']['correct'] for r in scored),
           'candidate_correct': sum(r['candidate']['correct'] for r in scored),
           'pass_condition_met': bool(passed),
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'CANDIDATE_RESULT.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f"\npass_condition_met: {passed} | {out['seconds']} s")


if __name__ == '__main__':
    main()
