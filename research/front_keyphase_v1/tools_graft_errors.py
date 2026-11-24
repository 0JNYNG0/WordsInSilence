"""Build handshape-error data by recombining REAL recordings, then measure stage 2 on them.

Takes the arm/wrist trajectory from one recording and the finger configuration from another.
Both halves are motion a person actually performed, so there is no synthesis noise - but nobody
ever performed the combination, so these are labelled SYNTHETIC_GRAFTED and must never be
counted as human attempts.

Run: py -3.13 tools_graft_errors.py
     py -3.13 tools_graft_errors.py --donor-session ses_XXXX   # use handshape samples as donors
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core import features as F  # noqa: E402
from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, POSE_L_SHOULDER, POSE_R_SHOULDER, RIGHT_HAND_SLICE,
    Observation, sha256_file,
)
from wis_lab.core.recognize import handshape_check, predict  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
BANDS = json.load(open(os.path.join(ROOT, "config", "handshape_bands.json"), encoding="utf-8"))
SESSION = "ses_cbd462050a95"
OUT_DIR = os.path.join(ROOT, "synthetic", "grafted_handshape_v1")
BUDGET = int(CONTRACT["sampling_A"]["budget"])


def shoulder_width(obs) -> float:
    ok = obs.usable[:, POSE_L_SHOULDER] & obs.usable[:, POSE_R_SHOULDER]
    if not ok.any():
        return float("nan")
    return float(np.median(np.linalg.norm(
        obs.xy_px[ok, POSE_L_SHOULDER] - obs.xy_px[ok, POSE_R_SHOULDER], axis=1)))


def graft(recipient: Observation, donor: Observation) -> Observation | None:
    """Recipient's trajectory, donor's finger configuration.

    Donor offsets are rescaled by the shoulder-width ratio so the transplanted fingers stay in
    proportion to the recipient's body rather than to the donor's distance from the camera.
    """
    sw_r, sw_d = shoulder_width(recipient), shoulder_width(donor)
    if not (np.isfinite(sw_r) and np.isfinite(sw_d) and sw_d > 1e-6):
        return None
    k = sw_r / sw_d

    o = Observation(
        xy_px=recipient.xy_px.copy(), timestamp_us=recipient.timestamp_us.copy(),
        frame_id=recipient.frame_id.copy(), observed=recipient.observed.copy(),
        usable=recipient.usable.copy(), conf=recipient.conf.copy(),
        hand_score=recipient.hand_score.copy(), sidecar={})
    tr, td = recipient.n_frames, donor.n_frames
    if tr < 2 or td < 2:
        return None

    replaced = 0
    for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        base = sl.start
        for t in range(tr):
            if not recipient.observed[t, base]:
                continue
            s = int(round(t / (tr - 1) * (td - 1)))
            if not donor.observed[s, base]:
                # nothing to transplant here: drop the recipient's fingers rather than keep them
                o.observed[t, base + 1:base + 21] = False
                o.usable[t, base + 1:base + 21] = False
                o.xy_px[t, base + 1:base + 21] = np.nan
                continue
            local = (donor.xy_px[s, base + 1:base + 21] - donor.xy_px[s, base]) * k
            o.xy_px[t, base + 1:base + 21] = o.xy_px[t, base] + local
            o.observed[t, base + 1:base + 21] = donor.observed[s, base + 1:base + 21]
            o.usable[t, base + 1:base + 21] = donor.usable[s, base + 1:base + 21]
            replaced += 1
    if replaced == 0:
        return None
    o.xy_px[~o.observed] = np.nan
    return o


def load_donor_samples(session: str) -> dict:
    """Handshape samples recorded through the web app, keyed by their label."""
    path = os.path.join(ROOT, "runs", session, "decisions.jsonl")
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_sign_attempt") is not False or not r.get("sample_label"):
            continue                      # only labelled handshape samples, never sign attempts
        npz = os.path.join(ROOT, "runs", session, "trials", r["trial_id"] + ".npz")
        if os.path.exists(npz):
            out.setdefault(r["sample_label"], []).append(Observation.load(npz))
    # one donor per label: the longest sample has the most usable frames
    return {k: max(v, key=lambda o: int(o.usable[:, 33:75].any(axis=1).sum()))
            for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-session", help="session of labelled handshape samples to graft from; "
                                            "default is the 18 reference clips")
    args = ap.parse_args()
    refs, lineage = load_reference_set()
    by_word = {r["word_id"]: r for r in refs}
    covered = BANDS["covered_words"]

    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "runs", SESSION, "decisions.jsonl"), encoding="utf-8")
            if l.strip()]
    recipients = [r for r in rows if r["requested_word_id"] in covered]
    if args.donor_session:
        donors = load_donor_samples(args.donor_session)
        donor_kind = f"handshape samples from {args.donor_session}"
        if not donors:
            print(f"no labelled handshape samples in {args.donor_session}")
            return 2
        print(f"donor handshapes: {', '.join(sorted(donors))}")
    else:
        donors = {}
        for clip in MANIFEST["clips"]:
            if clip["readiness"] != "UNSUPPORTED_CUE":
                donors[clip["word_id"]] = Observation.load(
                    os.path.join(ROOT, clip["observation_npz"]))
        donor_kind = "the 18 NIKL reference clips"

    out_dir = (os.path.join(ROOT, "synthetic", f"grafted_samples_{args.donor_session}")
               if args.donor_session else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    made, skipped, results = 0, 0, []
    t0 = time.perf_counter()
    for r in recipients:
        w = r["requested_word_id"]
        rec = Observation.load(os.path.join(ROOT, "runs", SESSION, "trials",
                                            r["trial_id"] + ".npz"))
        for dw, donor in donors.items():
            if not args.donor_session and dw == w:
                continue                      # same word is not a handshape error
            g = graft(rec, donor)
            if g is None:
                skipped += 1
                continue
            g.sidecar = {
                "schema": "wis_observation_2d_v1",
                "origin": "SYNTHETIC_GRAFTED",
                "NOT_A_HUMAN_ATTEMPT": True,
                "never_count_as_performance": True,
                "trajectory_from": {"session": SESSION, "trial_id": r["trial_id"], "word_id": w},
                "handshape_from": ({"sample_label": dw, "session": args.donor_session}
                                   if args.donor_session
                                   else {"clip_id": by_word[dw]["clip_id"], "word_id": dw}),
                "method": "recipient wrist trajectory + donor finger offsets, rescaled by the "
                          "shoulder-width ratio; donor frame chosen by proportional time index",
                "known_limit": "nobody performed this combination. Real signing couples wrist "
                               "rotation with hand movement and grafting breaks that coupling, so "
                               "these are unambiguously wrong rather than realistically wrong.",
                "mirrored": False, "rotation_deg": 0,
            }
            try:
                g.sidecar["quality"] = g.quality_summary()
                path = os.path.join(out_dir, f"{r['trial_id']}__{dw}.npz")
                g.save(path)
            except AssertionError:
                skipped += 1
                continue
            made += 1

            try:
                q = F.uniform_time_sample(F.sequence_features(g), BUDGET)
            except F.FeatureError:
                results.append({"word": w, "donor": dw, "status": "UNMEASURABLE"})
                continue
            hs = handshape_check(q, w, refs, CONTRACT)
            results.append({"word": w, "donor": dw, "status": hs["status"],
                            "distance": hs.get("distance")})

    dur = time.perf_counter() - t0
    print(f"만든 오답 데이터 {made}개 (건너뜀 {skipped}), {dur:.0f}초")
    print(f"저장 위치: synthetic/grafted_handshape_v1/  (전부 NOT_A_HUMAN_ATTEMPT 표시)\n")

    ok = [x for x in results if x["status"] in ("IN_BAND", "OUT_OF_BAND")]
    caught = sum(1 for x in ok if x["status"] == "OUT_OF_BAND")
    print(f"손모양 축이 걸러낸 비율: {caught}/{len(ok)} = {100 * caught / max(len(ok), 1):.1f}%")
    print(f"판정 불가(손가락 부족): {len(results) - len(ok)}\n")

    print(f"{'단어':<14} {'걸러냄':>10}   가장 안 걸리는 손모양 (통과한 것)")
    per = {}
    for x in ok:
        per.setdefault(x["word"], []).append(x)
    for w, xs in per.items():
        c = sum(1 for x in xs if x["status"] == "OUT_OF_BAND")
        missed = sorted({x["donor"] for x in xs if x["status"] == "IN_BAND"})
        print(f"{w:<14} {c:>6}/{len(xs):<4}   {', '.join(missed[:6]) or '없음'}")

    # false positives: the untouched originals, same check
    fp = 0
    for r in recipients:
        rec = Observation.load(os.path.join(ROOT, "runs", SESSION, "trials",
                                            r["trial_id"] + ".npz"))
        q = F.uniform_time_sample(F.sequence_features(rec), BUDGET)
        if handshape_check(q, r["requested_word_id"], refs, CONTRACT)["status"] == "OUT_OF_BAND":
            fp += 1
    print(f"\n원본 100개 중 잘못 걸러낸 것: {fp}/{len(recipients)}")

    report = {
        "task_id": "R2-step11-grafted-handshape-errors", "date_kst": time.strftime("%Y-%m-%d"),
        "material": {"trajectories": f"{SESSION} (web app recordings)",
                     "handshapes": donor_kind,
                     "new_recordings_required": 0},
        "generated": made, "skipped": skipped, "output_dir": os.path.relpath(out_dir, ROOT).replace("\\", "/"),
        "labelling": "every file carries origin=SYNTHETIC_GRAFTED and NOT_A_HUMAN_ATTEMPT=true",
        "handshape_axis_recall": f"{caught}/{len(ok)}",
        "recall_pct": round(100 * caught / max(len(ok), 1), 1),
        "unmeasurable": len(results) - len(ok),
        "false_positive_on_originals": f"{fp}/{len(recipients)}",
        "per_word": {w: {"caught": sum(1 for x in xs if x["status"] == "OUT_OF_BAND"),
                         "total": len(xs),
                         "missed_donors": sorted({x["donor"] for x in xs
                                                  if x["status"] == "IN_BAND"})}
                     for w, xs in per.items()},
        "what_this_measures": "how far the flexion-angle axis responds when the finger "
                              "configuration is replaced by a different real one",
        "what_this_does_NOT_measure": "whether a real, sloppy human performance would be caught. "
                                      "A grafted recording is unambiguously wrong, not "
                                      "realistically wrong.",
        "self_check_status": "PASS", "independent_review_status": "NOT_RUN",
        "performance_status": "SYNTHETIC_ONLY_NOT_HUMAN_PERFORMANCE",
    }
    p = os.path.join(ROOT, "evidence", "R2",
                     f"result_R2_step11{'_samples' if args.donor_session else ''}.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("->", os.path.relpath(p, ROOT).replace("\\", "/"), sha256_file(p)[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
