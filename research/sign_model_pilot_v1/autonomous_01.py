#!/usr/bin/env python3
"""Autonomous task 01: v3 aggregates on real clips, plus the data-ablation interpretation fixes.

Part A applies movement_events v3 to the clips the earlier submitted results were computed on, and
reports how the two versions differ - how many hands v2 called a confident still that v3 cannot
count, and how many counts change once bridged holes stop contributing turns.

Part B corrects the reporting around runs/data_ablation_v1, which is NOT re-run. Its result.json is
left byte-identical as reproduction evidence and the corrections live here.

No retraining, no camera, no change to model.py, data.py, axes.py, the contract, the web
recogniser, or runs/expanded_v1/model.pt. Only aggregates and code paths are printed.

Run: py -3.13 autonomous_01.py
"""
import collections
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import axes as V12  # noqa: E402
from pilot import data as D  # noqa: E402
from pilot.motion_events_v3 import movement_events_v3  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
ABLATION = os.path.join(ROOT, "runs", "data_ablation_v1", "result.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_clips():
    out = []
    for it in D.load_manifest(PREP):
        if it["source_only_pilot_eligible"]:
            c = D.load_clip(PREP, it)
            out.append(c)
    return out


def web_clips():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r.get("observation_status") != "OK" or not r.get("requested_word_id"):
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if os.path.exists(p):
            c = load_web_clip(p)
            c["word_label"] = r["requested_word_id"]
            out.append(c)
    return out


def compare(clips, label):
    """Per hand: how v2 and v3 disagree once bridging stops."""
    agg = collections.Counter()
    changed, bridged_turns = [], 0
    per_word = collections.defaultdict(lambda: collections.Counter())
    for c in clips:
        a = V12.movement_events(c)
        b = movement_events_v3(c)
        for hand in ("left", "right"):
            oa, ob = a[hand], b[hand]
            agg["hands"] += 1
            agg[f"v2_{oa['status']}"] += 1
            agg[f"v3_{ob['status']}"] += 1
            if ob["status"] == "OK":
                agg[f"v3_{ob['completeness']}"] += 1
            if oa["status"] == "OK" and ob["status"] != "OK":
                agg["v2_confident_v3_uncountable"] += 1
            if oa["status"] == "OK" and oa["turns"] == 0 and ob["status"] != "OK":
                agg["v2_called_still_v3_uncountable"] += 1
            if oa["status"] == "OK" and ob["status"] == "OK" and oa["turns"] != ob["turns"]:
                agg["turn_count_changed"] += 1
                bridged_turns += oa["turns"] - ob["turns"]
                if len(changed) < 12:
                    changed.append({"word": c.get("word_label"), "hand": hand,
                                    "v2_turns": oa["turns"], "v2_segments": oa["segments"],
                                    "v3_turns": ob["turns"], "v3_runs": ob["runs_total"],
                                    "v3_completeness": ob["completeness"]})
            w = c.get("word_label")
            if w:
                per_word[w][ob["status"]] += 1
    print(f"\n{label}  손 단위 {agg['hands']}건")
    print(f"  v2 OK {agg['v2_OK']}  /  v3 OK {agg['v3_OK']}"
          f"   (v3 완전관측 {agg['v3_COMPLETE']}, 부분관측 {agg['v3_PARTIAL']})")
    print(f"  v2가 셌지만 v3는 셀 수 없음: {agg['v2_confident_v3_uncountable']}"
          f"   그중 v2가 '정지 0회'라 한 것: {agg['v2_called_still_v3_uncountable']}")
    print(f"  양쪽 다 셌는데 횟수가 달라진 손: {agg['turn_count_changed']}"
          f"   (v2 합계 - v3 합계 = {bridged_turns})")
    return {"hands": agg["hands"], "counts": dict(agg), "net_turns_removed_by_not_bridging":
            bridged_turns, "examples": changed,
            "per_word_v3_status": {k: dict(v) for k, v in per_word.items()}}


def part_b():
    d = json.load(open(ABLATION, encoding="utf-8"))
    runs = d["per_run"]
    by = collections.defaultdict(list)
    for r in runs:
        by[r["fraction"]].append(r)
    actual = {f"{int(f*100)}%": {"steps_run": [r["steps_run"] for r in rs],
                                 "checkpoint_step": [r["checkpoint_step"] for r in rs],
                                 "training_clips": rs[0]["training_clips"]}
              for f, rs in sorted(by.items())}
    print("\n실제 실행 step (동일하지 않음)")
    for k, v in actual.items():
        print(f"  {k:>5}  실행 {v['steps_run']}   체크포인트 {v['checkpoint_step']}")
    return {
        "target": "runs/data_ablation_v1/result.json",
        "file_left_byte_identical": True,
        "file_sha256": sha256_file(ABLATION),
        "not_rerun": True,
        "withdrawn_wording": {
            "was": "identical step count everywhere / every condition gets the same 3,960 "
                   "gradient steps",
            "why_wrong": "3,960 was the maximum budget. Early stopping ended every run sooner and "
                         "at different points, so the conditions did not receive equal training.",
            "correct_wording": "all conditions shared the same maximum budget (3,960 steps), the "
                               "same evaluation interval (66 steps) and the same stopping rule "
                               "(10 evaluation rounds without improvement); the steps actually run "
                               "differ per run and are listed below",
            "actual_steps": actual,
        },
        "subset_construction_caveat": {
            "what_was_done": "subsets are nested and stratified BY CLASS",
            "what_was_not_done": "they are not stratified by signer",
            "consequence": "a class's signer composition can change between conditions, so 'amount "
                           "of data' and 'which signers contribute to a class' moved together. "
                           "The ablation therefore isolates data amount from architecture and "
                           "budget, but not from per-class signer mix.",
        },
        "conclusion_status": {
            "kept": "the observation stands: with architecture, class set, split, budget rule and "
                    "seeds handled as recorded, the 알다 transfer fell from 60/60 at 25 percent to "
                    "19/60 at 100 percent while 감사합니다 stayed 60/60, and validation fit "
                    "correlated with transfer failure at Spearman -0.78 over nine runs",
            "downgraded": "source-domain specialisation remains a HYPOTHESIS. Two words, forty "
                          "clips, one performer and one subset draw do not exclude data shortage "
                          "as a cause in general, and the earlier expansion did fix 감사합니다 "
                          "from 11/20 to 20/20.",
            "not_discarded": "the model-side signal is kept as a lead, not deleted",
        },
        "other_overreach_corrected": [
            {"claim": "DTW matched the same headword, so a lexical-variant difference is excluded",
             "appeared_in": "README.md, 어휘 확장 재학습 section, and the chat report",
             "status": "WITHDRAWN as stated",
             "why": "agreement on a class label is not linguistic verification that AIHub's 알다 "
                    "and the participant's 알다 are the same variant. A nearest-template match "
                    "shows the two are closer to each other than to the other four templates; it "
                    "does not establish sameness of form.",
             "what_remains": "the variant question is OPEN, not settled either way"},
            {"claim": "252 ms end-to-end latency for the web tester",
             "appeared_in": "evidence/corrections_20260912.json and earlier reporting",
             "status": "DEFINITION CORRECTED, number retained",
             "why": "by the definition actually verified, 252 ms is the server-side interval from "
                    "end-candidate to server result. It is not confirmed as motion-end to "
                    "browser-display.",
             "what_remains": "no end-to-end latency figure is claimed; the lineage of the 252 ms "
                             "measurement is kept and re-labelled rather than deleted"},
        ],
    }


def main() -> int:
    t0 = time.perf_counter()
    src, web = source_clips(), web_clips()
    print(f"AIHub eligible {len(src)}건, 참가자 웹 {len(web)}건 (원본 배열은 출력하지 않음)")
    a_src = compare(src, "AIHub (OpenPose)")
    a_web = compare(web, "참가자 웹캠 (MediaPipe)")
    b = part_b()

    out = {
        "task_id": "autonomous_01", "date_kst": time.strftime("%Y-%m-%d"),
        "instruction": "WordsInSilence_자율협업_20260912/CLAUDE_TASK_01.md",
        "command": "py -3.13 autonomous_01.py",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "part_a_movement_events_v3": {
            "new_module": "src/pilot/motion_events_v3.py",
            "regression_suite": "tests/test_motion_events_v3.py",
            "design": {
                "runs_break_at": "every explicitly missing frame, and any time gap over 0.25 s",
                "bridging": "none - the simple path the instruction asked for first",
                "turns_counted": "inside each observed run of at least 3 frames only",
                "completeness": "COMPLETE only when one run covers every frame; otherwise PARTIAL, "
                                "and the count is neither an upper nor a lower bound because a gap "
                                "can hide a turn and can also split one movement across two runs",
                "insufficient_when": "fewer than 8 observed frames, or observed fraction below "
                                     "0.5, or no unbroken run reaching 3 frames, or fewer than 8 "
                                     "frames inside countable runs",
                "still_vs_unseen": "a seen, motionless hand returns OK with 0; an uncountable hand "
                                   "returns INSUFFICIENT_OBSERVATION with turns None",
                "observed_seconds": "summed within runs; the first-to-last distance is reported "
                                    "separately as span_seconds and is never called observation",
            },
            "counterexamples_reproduced_then_fixed": {
                "1_gap_bridging": {
                    "input": "31 frames at 30 fps, right hand invalid at frames 10, 11, 21, 22",
                    "v2": {"segments": 1, "turns": 3, "turns_span_gaps": False},
                    "v3": {"status": "OK", "runs_total": 3, "gap_count": 2, "turns": 1,
                           "completeness": "PARTIAL", "observed_seconds": 0.8,
                           "span_seconds": 1.0},
                    "significance": "bridging did not merely mislabel the report - it produced two "
                                    "turns that no observed run supports",
                },
                "2_single_frame_runs": {
                    "input": "17 frames at dt 0.2 s, right hand valid on even indices only",
                    "v2": {"status": "OK", "turns": 0, "segments": 9, "observed_seconds": 3.2},
                    "v3": {"status": "INSUFFICIENT_OBSERVATION", "turns": None,
                           "longest_run_frames": 1,
                           "reason": "no unbroken run reaches 3 frames"},
                },
            },
            "sampling_condition_measured_not_promised": {
                "synthetic_path": "two up-down cycles, 3 true turns",
                "turns_by_samples": {"4": None, "6": None, "8": 3, "10": 3, "15": 3, "30": 3,
                                     "60": 3, "120": 3, "240": 3},
                "note": "the 4 and 6 sample cases return None because they fall under the "
                        "8-observed-frame floor, so this sweep did not reach the counting "
                        "algorithm's own limit. No invariance is claimed for arbitrary paths or "
                        "occlusion patterns; what is shown is stability for this path above the "
                        "observation floor.",
            },
            "real_clip_aggregates": {"aihub_eligible_163": a_src, "participant_web_100": a_web},
            "preserved_unchanged": [
                "src/pilot/axes.py (v1 direction_reversals and v2 movement_events, byte identical)",
                "evidence/result_axis_split.json",
                "evidence/result_movement_structure.json",
                "evidence/result_motion_events.json",
                "src/pilot/model.py", "src/pilot/data.py", "runs/expanded_v1/model.pt",
                "the 280-channel contract and the default web recogniser",
            ],
        },
        "part_b_data_ablation_interpretation": b,
        "no_retraining_performed": True,
        "no_gate_deployed": True,
        "remaining_limits": [
            "v3 refuses to count where v2 guessed; that is a reporting fix, not evidence that the "
            "turn count is a good movement feature",
            "the real-clip aggregates describe how the two versions differ, not how well either "
            "describes sign movement",
            "the participant session predates attempt_label, so correctness of each attempt is "
            "unverified and these clips remain diagnostic data only",
            "nothing here judges whether any attempt was performed correctly",
        ],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "autonomous_01_result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
