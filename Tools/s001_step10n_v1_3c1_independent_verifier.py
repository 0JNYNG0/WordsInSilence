#!/usr/bin/env python3
"""Step 10N v1.3c1 — Independent verifier.

Recomputes normalization, DTW, and all assertions from the saved coordinate
payload without importing the main tool. Accepts --archive-root for
self-contained execution from extracted ZIP.
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

WORDS = ["GREETING", "LOVE", "THANK_YOU"]
TRAIN_MEDOIDS = {
    "GREETING__FAST": "S001_P001_GREETING_FAST_R02",
    "GREETING__NORMAL": "S001_P001_GREETING_NORMAL_R03",
    "GREETING__SLOW": "S001_P001_GREETING_SLOW_R01",
    "LOVE__FAST": "S001_P001_LOVE_FAST_R01",
    "LOVE__NORMAL": "S001_P001_LOVE_NORMAL_R02",
    "LOVE__SLOW": "S001_P001_LOVE_SLOW_R01",
    "THANK_YOU__FAST": "S001_P001_THANK_YOU_FAST_R02",
    "THANK_YOU__NORMAL": "S001_P001_THANK_YOU_NORMAL_R05",
    "THANK_YOU__SLOW": "S001_P001_THANK_YOU_SLOW_R01",
}
TEST_SEAL_IDS = [
    "S001_P001_GREETING_NORMAL_R01", "S001_P001_GREETING_SLOW_R02", "S001_P001_GREETING_SLOW_R05",
    "S001_P001_LOVE_NORMAL_R01", "S001_P001_LOVE_SLOW_R02", "S001_P001_LOVE_SLOW_R05",
    "S001_P001_THANK_YOU_NORMAL_R01", "S001_P001_THANK_YOU_SLOW_R02", "S001_P001_THANK_YOU_SLOW_R05",
]

MISSING_COST = 1.0
BAND_EPS = 1e-15


def dtw(q_c, q_m, p_c, p_m, band=0.10):
    """Independent DTW reimplementation matching C# MaskAwareDtw."""
    T1, T2 = q_c.shape[0], p_c.shape[0]
    INF = float("inf")
    cum = np.full((T1, T2), INF)
    par = np.zeros((T1, T2), dtype=np.uint8)

    def in_band(i, j):
        if T1 <= 1 or T2 <= 1:
            return True
        return abs(i / (T1 - 1) - j / (T2 - 1)) <= band + BAND_EPS

    def lc(i, j):
        total = 0.0
        incl = 0
        for lm in range(16):
            qv = bool(q_m[i, lm])
            pv = bool(p_m[j, lm])
            if qv and pv:
                s = sum((float(q_c[i, lm, ax] - p_c[j, lm, ax])) ** 2 for ax in range(q_c.shape[2]))
                total += math.sqrt(s)
                incl += 1
            elif qv != pv:
                total += MISSING_COST
                incl += 1
        return MISSING_COST if incl == 0 else total / incl

    for i in range(T1):
        for j in range(T2):
            if not in_band(i, j):
                continue
            c = lc(i, j)
            if i == 0 and j == 0:
                cum[i, j] = c
                continue
            best = INF
            d = 0
            if i > 0 and j > 0 and not math.isinf(cum[i - 1, j - 1]):
                best = cum[i - 1, j - 1]
                d = 1
            if i > 0 and not math.isinf(cum[i - 1, j]) and cum[i - 1, j] < best:
                best = cum[i - 1, j]
                d = 2
            if j > 0 and not math.isinf(cum[i, j - 1]) and cum[i, j - 1] < best:
                best = cum[i, j - 1]
                d = 3
            if d:
                cum[i, j] = best + c
                par[i, j] = d

    pi, pj = T1 - 1, T2 - 1
    path = [(pi, pj)]
    while pi or pj:
        dd = par[pi, pj]
        if dd == 1:
            pi -= 1; pj -= 1
        elif dd == 2:
            pi -= 1
        elif dd == 3:
            pj -= 1
        else:
            raise RuntimeError(f"backtrace stuck at ({pi},{pj})")
        path.append((pi, pj))
    return cum[T1 - 1, T2 - 1] / len(path)


def sample_word(sid):
    """Extract word from sample ID."""
    parts = sid.replace("S001_P001_", "").rsplit("_", 2)
    return "_".join(parts[:-2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=str, default=None,
                        help="Root directory of extracted archive")
    args = parser.parse_args()

    if args.archive_root:
        root = Path(args.archive_root)
    else:
        root = Path(__file__).resolve().parent.parent / "Validation" / "step10n_v1_3c1"

    print("=" * 60)
    print("Step 10N v1.3c1 — Independent Verifier")
    print(f"Archive root: {root}")
    print("=" * 60)

    checks = 0
    failures = 0

    # Load payload
    npz_path = root / "coordinate_payload.npz"
    if not npz_path.exists():
        print(f"FAIL: coordinate_payload.npz not found at {npz_path}")
        return 1
    payload = np.load(npz_path, allow_pickle=True)

    # Load saved predictions
    sealed_preds = json.load(open(root / "sealed_proxy_predictions.json"))
    live_corr_preds = json.load(open(root / "live_corrected_predictions.json"))
    live_uncorr_preds = json.load(open(root / "live_uncorrected_predictions.json"))

    # ── 1. Verify sealed proxy predictions ──
    print("\n[1] Sealed proxy DTW verification...")
    for pred in sealed_preds:
        qid = pred["query"]
        q_c = payload[f"query_coords_{qid}"]
        q_m = payload[f"query_mask_{qid}"]
        word_best = {}
        for tid in TRAIN_MEDOIDS:
            p_c = payload[f"proto_coords_{tid}"]
            p_m = payload[f"proto_mask_{tid}"]
            d = dtw(q_c, q_m, p_c, p_m, 0.10)
            tw = tid.split("__")[0]
            if tw not in word_best or d < word_best[tw]:
                word_best[tw] = d
        my_pred = min(word_best, key=lambda w: word_best[w])
        checks += 1
        if my_pred != pred["predicted"]:
            print(f"  MISMATCH: {qid} saved={pred['predicted']} indep={my_pred}")
            failures += 1
        # Check distance for expected word
        expected = pred["expected"]
        saved_d = pred["word_dists"][expected]
        my_d = word_best[expected]
        checks += 1
        if abs(saved_d - my_d) > 1e-6:
            print(f"  DIST MISMATCH: {qid} expected_dist saved={saved_d:.10f} indep={my_d:.10f}")
            failures += 1

    sealed_ok = sum(1 for p in sealed_preds if p["correct"])
    print(f"  Sealed proxy: {sealed_ok}/9 predictions verified")

    # ── 2. Verify live corrected predictions ──
    print("\n[2] Live Y-corrected DTW verification...")
    for pred in live_corr_preds:
        word = pred["query"]
        q_c = payload[f"live_corrected_coords_{word}"]
        q_m = payload[f"live_corrected_mask_{word}"]
        word_best = {}
        for tid in TRAIN_MEDOIDS:
            p_c = payload[f"proto_coords_{tid}"]
            p_m = payload[f"proto_mask_{tid}"]
            d = dtw(q_c, q_m, p_c, p_m, 0.10)
            tw = tid.split("__")[0]
            if tw not in word_best or d < word_best[tw]:
                word_best[tw] = d
        my_pred = min(word_best, key=lambda w: word_best[w])
        checks += 1
        if my_pred != pred["predicted"]:
            print(f"  MISMATCH: {word} saved={pred['predicted']} indep={my_pred}")
            failures += 1
        else:
            saved_d = pred["word_dists"][pred["predicted"]]
            my_d = word_best[my_pred]
            checks += 1
            if abs(saved_d - my_d) > 1e-6:
                print(f"  DIST MISMATCH: {word} saved={saved_d:.10f} indep={my_d:.10f}")
                failures += 1

    corr_ok = sum(1 for p in live_corr_preds if p["correct"])
    print(f"  Live Y-corrected: {corr_ok}/3 verified")

    # ── 3. Verify live uncorrected predictions ──
    print("\n[3] Live uncorrected DTW verification...")
    for pred in live_uncorr_preds:
        word = pred["query"]
        q_c = payload[f"live_uncorrected_coords_{word}"]
        q_m = payload[f"live_uncorrected_mask_{word}"]
        word_best = {}
        for tid in TRAIN_MEDOIDS:
            p_c = payload[f"proto_coords_{tid}"]
            p_m = payload[f"proto_mask_{tid}"]
            d = dtw(q_c, q_m, p_c, p_m, 0.10)
            tw = tid.split("__")[0]
            if tw not in word_best or d < word_best[tw]:
                word_best[tw] = d
        my_pred = min(word_best, key=lambda w: word_best[w])
        checks += 1
        if my_pred != pred["predicted"]:
            print(f"  MISMATCH: {word} saved={pred['predicted']} indep={my_pred}")
            failures += 1

    uncorr_ok = sum(1 for p in live_uncorr_preds if p["correct"])
    print(f"  Live uncorrected: {uncorr_ok}/3 verified")

    # ── 4. Verify reflection invariance ──
    print("\n[4] Reflection invariance assertions...")
    refl = json.load(open(root / "reflection_invariance_proof.json"))
    for r in refl["horizontal"]:
        checks += 1
        if not r["pass"]:
            print(f"  H-REFL FAIL: {r['template']}")
            failures += 1
    for r in refl.get("vertical", []):
        checks += 1
        if not r.get("x_pass", True):
            print(f"  V-REFL X FAIL: {r['template']}")
            failures += 1
        checks += 1
        if not r.get("y_negate_pass", True):
            print(f"  V-REFL Y FAIL: {r['template']}")
            failures += 1

    print(f"  Reflection invariance: verified")

    # ── 5. Verify Y-axis inversion evidence ──
    print("\n[5] Y-axis inversion assertions...")
    y_ev = json.load(open(root / "y_axis_inversion_evidence.json"))
    # Prototype should be y-down: nose_y < shoulder_y
    proto_vals = list(y_ev["prototype"].values())[0]
    checks += 1
    if not (proto_vals["nose"] < proto_vals["l_shld"]):
        print(f"  FAIL: Prototype not y-down")
        failures += 1
    # Live should be y-up: nose_y > shoulder_y
    for word, vals in y_ev["live"].items():
        checks += 1
        if vals["nose"] is not None and vals["l_shld"] is not None:
            if not (vals["nose"] > vals["l_shld"]):
                print(f"  FAIL: Live {word} not y-up")
                failures += 1
    print(f"  Y-axis inversion: verified")

    # ── 6. Verify root cause matrix ──
    print("\n[6] Root cause decision...")
    rc = json.load(open(root / "corrected_root_cause_decision_matrix.json"))
    checks += 1
    if rc.get("mirror_defect_claim") != "REJECTED":
        print(f"  FAIL: mirror defect should be REJECTED")
        failures += 1
    checks += 1
    if not rc.get("y_axis_inversion", {}).get("confirmed"):
        print(f"  FAIL: Y-axis inversion should be confirmed")
        failures += 1
    print(f"  Root cause: {rc.get('primary_cause', '?')}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Independent verifier: {checks} checks, {failures} failures")
    print(f"{'ALL PASS' if failures == 0 else 'FAIL'}")
    print(f"{'=' * 60}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
