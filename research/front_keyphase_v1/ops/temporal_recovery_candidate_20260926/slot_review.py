"""Per-slot review of the recovery candidate's open items. Decision logic is NOT changed.

Targets (deduplicated): recovery-vs-baseline CHANGED_CONTENT slots, the NEO slots the original lost
that recovery does not hold with baseline content, and the JOE slot recovery lost vs original.
Each target frame is recorded with BOTH sides in all three arms, the identity of every detection
(content-matched to the per-frame detection list), and arm-link geometry for reference only.
Existing frame/run verdicts from side_assignment_candidate_20260925 are attached where they exist.
"""
import json, pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
OLD = L / 'ops' / 'temporal_assignment_candidate_20260926'
SA = L / 'ops' / 'side_assignment_candidate_20260925'
sys.path[:0] = [str(H), str(L), str(L / 'src'), str(OLD)]
from wis_lab.capture.video_source import extract           # noqa: E402
from temporal_landmarker import TemporalLandmarker         # noqa: E402
from recovery_landmarker import RecoveryLandmarker         # noqa: E402
import importlib.util                                      # noqa: E402
_sp = importlib.util.spec_from_file_location('recovery_run_compare', H / 'run_compare.py')
_rc = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_rc)
CONTRACT, CACHE, content, same, decompose, ERRORS, KEEP = (
    _rc.CONTRACT, _rc.CACHE, _rc.content, _rc.same, _rc.decompose, _rc.ERRORS, _rc.KEEP)

PALM = (0, 5, 9, 13, 17)
HB = {0: 33, 1: 54}
POSE = {0: {'sh': 11, 'el': 13, 'wr': 15}, 1: {'sh': 12, 'el': 14, 'wr': 16}}


def det_id(palm, dets):
    if palm is None:
        return None
    for k, d in enumerate(dets):
        if np.allclose(palm, d, atol=1.0):
            return k
    return 'unmatched'


def geom(palm, xy, us):
    """reference-only link of one detection to each arm: wrist distance (shoulder widths) and
    cosine between forearm (elbow->pose wrist) and hand (wrist->mean MCP) directions"""
    if not (us[11] and us[12]):
        return None
    sw = float(np.linalg.norm(xy[11] - xy[12]))
    hand = palm[1:].mean(0) - palm[0]
    g = {}
    for s in (0, 1):
        p = POSE[s]
        e = {}
        if us[p['wr']]:
            e['d_wrist'] = round(float(np.linalg.norm(palm[0] - xy[p['wr']]) / sw), 3)
            if us[p['el']]:
                fa = xy[p['wr']] - xy[p['el']]
                e['cos_forearm'] = round(float(fa @ hand / (np.linalg.norm(fa) * np.linalg.norm(hand) + 1e-9)), 3)
        g['LR'[s]] = e
    g['hand_dir_dx'] = round(float(hand[0]), 1)
    return g


def run(cls, src, stem):
    with cls(CONTRACT) as lm:
        o = extract(src, lm, stem)
        log = {round(e['t'], 4): e for e in lm.frame_log}
    return o, log


def main():
    res = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))
    clips = [r['clip'] for r in res['nikl'] if r['recov_vs_base']['CHANGED_CONTENT'] or
             r['clip'] in ('NIKL_NEO_01', 'NIKL_JOESONGHADA_01')]
    review = json.loads((SA / 'RUN_REVIEW.json').read_text(encoding='utf-8'))['rows']
    prior = {}
    for rv in review:
        for f in range(rv['run'][0], rv['run'][-1] + 1):
            prior.setdefault((rv['clip'], f), []).append(
                {'source': 'RUN_REVIEW', 'side_unassigned_by_mutual_nearest': rv['side'],
                 'displayed': f in rv['frames_displayed_inside_the_run'],
                 'category': rv['category'], 'seen': rv['what_was_seen']})
    for c, f, s in ERRORS:
        prior.setdefault((c, f), []).append({'source': 'FROZEN_ERROR', 'side': 'LR'[s], 'displayed': True})
    for c, f, s in KEEP:
        prior.setdefault((c, f), []).append({'source': 'FROZEN_NORMAL', 'side': 'LR'[s], 'displayed': True})

    out, tally = [], {'CHANGED_CONTENT_vs_baseline': 0, 'neo_mismatch': 0, 'joe_new_loss': 0}
    for clip in clips:
        z = np.load(CACHE / f'{clip}.npz', allow_pickle=False)
        src = json.loads((CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        o, _ = run(TemporalLandmarker, src, clip)
        c, log = run(RecoveryLandmarker, src, clip)
        bxy, bus = z['xy_px'], z['usable']
        arms = {'baseline': (bxy, bus), 'original': (o.xy_px, o.usable), 'recovery': (c.xy_px, c.usable)}
        _, lost_ob, _ = decompose(bxy, bus, o.xy_px, o.usable)
        rb, lost_rb, _ = decompose(bxy, bus, c.xy_px, c.usable)
        _, lost_ro, _ = decompose(o.xy_px, o.usable, c.xy_px, c.usable)
        rec = res['nikl'][[r['clip'] for r in res['nikl']].index(clip)]
        assert rb == rec['recov_vs_base'], (clip, rb)          # identical to the stored run
        targets = {}
        for r in range(bxy.shape[0]):
            for s in (0, 1):
                bc, cc = content(bxy, bus, r, s), content(c.xy_px, c.usable, r, s)
                oc = content(o.xy_px, o.usable, r, s)
                oth_b, oth_c = content(bxy, bus, r, 1 - s), content(c.xy_px, c.usable, r, 1 - s)
                swapped = same(bc, oth_c) and same(oth_b, cc)
                if bc is not None and cc is not None and not same(bc, cc) and not swapped:
                    targets.setdefault(r, set()).add(('CHANGED_CONTENT_vs_baseline', s))
                if clip == 'NIKL_NEO_01' and (r, s) in lost_ob and (r, s) not in lost_rb and not same(bc, cc):
                    targets.setdefault(r, set()).add(('neo_mismatch', s))
            for s in (0, 1):
                if clip == 'NIKL_JOESONGHADA_01' and (r, s) in lost_ro:
                    targets.setdefault(r, set()).add(('joe_new_loss', s))
        fids = list(map(int, z['frame_id']))
        for r, tg in sorted(targets.items()):
            for k, _ in tg:
                tally[k] += 1
            t = round(float(c.timestamp_us[r]) / 1e6, 4)
            dets = [np.array(p) for p in log[t]['palms']] if t in log else []
            row = {'clip': clip, 'row': r, 'frame': fids[r],
                   'targets': sorted([{'kind': k, 'side': 'LR'[s]} for k, s in tg], key=str),
                   'n_detections': len(dets),
                   'detections': [geom(d, bxy[r], bus[r]) for d in dets],
                   'slots': {a: {'LR'[s]: det_id(content(xy, us, r, s), dets) for s in (0, 1)}
                             for a, (xy, us) in arms.items()},
                   'prior_evidence': prior.get((clip, fids[r]), [])}
            out.append(row)
            print(clip, fids[r], row['targets'], row['slots'], len(row['prior_evidence']), flush=True)
    doc = {'schema': 'wis_recovery_slot_review_v1', 'at_kst': '2026-09-26',
           'decision_logic_changed': False, 'slot_tally': tally, 'frames': out}
    (H / 'SLOT_REVIEW.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    print(tally, len(out), 'frames')


if __name__ == '__main__':
    main()
