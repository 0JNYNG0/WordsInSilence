"""Derive per-word two-sided bands from correct attempts, then test them honestly.

Two rules keep this from being circular:

* the band for a given attempt is derived from the OTHER attempts of that word (leave-one-out),
  so the normal-rejection rate is not measured against a band that already contains it;
* the error trials come from an earlier session that contributes nothing to any band.

Nothing here trains a model. It computes two numbers per word and freezes them with the hashes
of the recordings they came from.

Run: py -3.13 tests/derive_bands.py --session ses_XXXX [--errors ses_cde9d19d75f2]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core.observation import Observation, sha256_file  # noqa: E402
from wis_lab.core.recognize import predict  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
# protocol order of the earlier error session
ERROR_LABELS = {16: "주먹", 17: "주먹", 18: "순서반대", 19: "순서반대",
                20: "정지", 21: "정지", 22: "다른단어", 23: "다른단어"}


def load_session(sid):
    p = os.path.join(ROOT, "runs", sid, "decisions.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def distances(sid, rows, refs, only_indices=None):
    """Distance from each attempt to the word it was asked to perform, plus its top-1."""
    out = []
    for i, r in enumerate(rows):
        if only_indices is not None and i not in only_indices:
            continue
        npz = os.path.join(ROOT, "runs", sid, "trials", r["trial_id"] + ".npz")
        if not os.path.exists(npz):
            continue
        res = predict(Observation.load(npz), refs, CONTRACT)
        w = r["requested_word_id"]
        d = next((x["distance"] for x in res["ranking"] if x["word_id"] == w), None)
        out.append({"i": i, "trial_id": r["trial_id"], "word": w, "d": d,
                    "top1": res.get("top1_word_id"), "status": res["observation_status"],
                    "sha256": sha256_file(npz)})
    return out


BANDS = {
    "min_max": lambda v: (float(min(v)), float(max(v))),
    "p5_p95": lambda v: (float(np.percentile(v, 5)), float(np.percentile(v, 95))),
    "median_2.5mad": lambda v: _mad(v, 2.5),
    "median_3mad": lambda v: _mad(v, 3.0),
}


def _mad(v, k):
    med = float(np.median(v))
    mad = float(np.median(np.abs(np.asarray(v) - med))) or 1e-6
    return med - k * 1.4826 * mad, med + k * 1.4826 * mad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session of CORRECT attempts")
    ap.add_argument("--errors", default="ses_cde9d19d75f2", help="earlier session with error trials")
    ap.add_argument("--out", default="reports/bands_v1")
    args = ap.parse_args()

    refs, lineage = load_reference_set()
    norm_rows = load_session(args.session)
    norm = [x for x in distances(args.session, norm_rows, refs) if x["d"] is not None]
    err_rows = load_session(args.errors)
    err = distances(args.errors, err_rows, refs, only_indices=set(ERROR_LABELS))
    for e in err:
        e["kind"] = ERROR_LABELS[e["i"]]

    by_word = {}
    for x in norm:
        by_word.setdefault(x["word"], []).append(x)

    print(f"정상 {len(norm)}회 / 단어 {len(by_word)}개, 오류 {len(err)}회 (별도 세션)")
    print(f"{'단어':<14} {'n':>3} {'거리 범위':>21} {'중앙값':>8} {'top1':>7}")
    for w, xs in by_word.items():
        v = [x["d"] for x in xs]
        hit = sum(1 for x in xs if x["top1"] == w)
        print(f"{w:<14} {len(v):>3} {min(v):>9.4f}~{max(v):<10.4f} {np.median(v):>8.4f} "
              f"{hit:>4}/{len(v)}")

    report = {"task_id": "R3-bands", "session_normal": args.session, "session_errors": args.errors,
              "contract_id": CONTRACT["contract_id"], "reference_set_id": lineage["set_id"],
              "n_normal": len(norm), "n_error": len(err), "per_word": {}, "candidates": {}}

    print(f"\n{'대역 정의':<16} {'정상 거부(LOO)':>15} {'오류 거부':>11} {'주먹':>7} {'순서':>7} "
          f"{'정지':>7} {'다른단어':>9}")
    for name, fn in BANDS.items():
        rejected_normal = 0
        for w, xs in by_word.items():
            v = [x["d"] for x in xs]
            for k in range(len(v)):
                others = v[:k] + v[k + 1:]          # leave-one-out: no attempt judges itself
                lo, hi = fn(others)
                if not (lo <= v[k] <= hi):
                    rejected_normal += 1
        caught, per_kind = 0, {}
        for e in err:
            v = [x["d"] for x in by_word.get(e["word"], [])]
            if not v or e["d"] is None:
                continue
            lo, hi = fn(v)
            # accepted only when the ranking agrees AND the distance sits inside the band
            ok = (e["top1"] == e["word"]) and (lo <= e["d"] <= hi)
            per_kind.setdefault(e["kind"], [0, 0])
            per_kind[e["kind"]][1] += 1
            if not ok:
                caught += 1
                per_kind[e["kind"]][0] += 1
        report["candidates"][name] = {
            "normal_rejected_leave_one_out": rejected_normal,
            "normal_total": len(norm),
            "error_rejected": caught, "error_total": len(err),
            "by_kind": {k: f"{a}/{b}" for k, (a, b) in per_kind.items()},
            "bands": {w: [round(x, 5) for x in fn([y["d"] for y in xs])]
                      for w, xs in by_word.items()},
        }
        bk = report["candidates"][name]["by_kind"]
        print(f"{name:<16} {rejected_normal:>8}/{len(norm):<6} {caught:>6}/{len(err):<4} "
              f"{bk.get('주먹','-'):>7} {bk.get('순서반대','-'):>7} {bk.get('정지','-'):>7} "
              f"{bk.get('다른단어','-'):>9}")

    report["per_word"] = {
        w: {"n": len(xs), "min": round(min(x["d"] for x in xs), 5),
            "max": round(max(x["d"] for x in xs), 5),
            "median": round(float(np.median([x["d"] for x in xs])), 5),
            "top1_correct": sum(1 for x in xs if x["top1"] == w),
            "trial_sha256": [x["sha256"][:16] for x in xs]}
        for w, xs in by_word.items()}
    report["honesty"] = {
        "normal_rejection_is_leave_one_out": True,
        "errors_from_a_separate_session": args.errors != args.session,
        "nothing_was_trained": "the recognition model, features, weights and DTW are unchanged; "
                               "this computes two numbers per word",
        "one_participant": "these bands describe this tester only and are not transferable",
        "still_needed": "a frozen re-test with NEW correct recordings, to check the band does not "
                        "reject good performances it has never seen",
    }
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "bands.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n->", os.path.relpath(p, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
