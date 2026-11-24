#!/usr/bin/env python3
"""Re-frame the interpreter's clips into the participant's capture geometry, and see what breaks.

The participant cannot be asked to sit at a fixed distance - the game has to work from any webcam
at any seating distance - so this asks the same question as the recording experiment without any
recording: take the NIKL clips the seven models recognise, place the signer where the participant
actually sits, and let whatever leaves the frame be genuinely lost.

Three conditions on the same clips and the same models. No training.

  S0  as recorded                     control; 알다 is known to be 7/7
  S1  moved to the participant's geometry, and any landmark outside the frame is marked
      unobserved - the honest consequence of sitting close
  S2  the same geometric move WITHOUT losing anything outside the frame

S2 exists to separate two things that would otherwise be confounded. The features divide by
shoulder width, so pure scale should change nothing; S2 must therefore match S0. If it does not,
the transform is doing something unintended and the run is void. Any difference between S1 and S2
is then attributable to data that was never captured, not to the scaling.

What this cannot reproduce: close-range perspective distortion, which 2D landmarks cannot simulate.
Frame rate is not a suspect here - the participant's ALDA ran at 31.9 fps against NIKL's 30.0.

Pre-registered rule, fixed before the numbers are seen:
  validity   S0 알다 must be 7/7, and S2 must equal S0 on every clip
  <= 2/7     S1 알다: capture framing (losing the hand off-frame) is implicated
  >= 5/7     capture framing is ruled out; execution remains the suspect
  3-4/7      inconclusive
Both the five-candidate and the eighty-candidate readings are reported; neither is chosen after
the fact.

Run: py -3.13 capture_simulation_run.py
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.model_masked_v1 import MaskedSignTCN  # noqa: E402
from pilot.web_input import active_span, to_native_like  # noqa: E402

FRONT = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1"))
NIKL_DIR = os.path.join(FRONT, "cache", "2683eadbc5f1c588")
RUN = os.path.join(ROOT, "runs", "capture_simulation_v1")
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
NIKL_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다", "NA": "나",
                 "JOTA": "좋다", "MOREUDA": "모르다"}
L_SH, R_SH = 11, 12                       # MediaPipe pose shoulders
# the participant's measured geometry, from 100 clips of session ses_cbd462050a95
TARGET = {"frame_w": 1280.0, "frame_h": 720.0, "shoulder_over_height": 0.42,
          "mid_x_fraction": 0.529, "mid_y_fraction": 0.546}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reframe(xy, usable, drop_offscreen: bool):
    """Place the signer where the participant sits. Optionally lose what leaves the frame."""
    ok = usable[:, L_SH] & usable[:, R_SH]
    if ok.sum() < 3:
        return None
    sw = float(np.median(np.linalg.norm(xy[ok, L_SH] - xy[ok, R_SH], axis=1)))
    mid = np.median((xy[ok, L_SH] + xy[ok, R_SH]) / 2.0, axis=0)
    k = (TARGET["shoulder_over_height"] * TARGET["frame_h"]) / sw
    target_mid = np.array([TARGET["mid_x_fraction"] * TARGET["frame_w"],
                           TARGET["mid_y_fraction"] * TARGET["frame_h"]], dtype=np.float64)
    out = (xy.astype(np.float64) - mid) * k + target_mid
    use = usable.copy()
    inside = ((out[..., 0] >= 0) & (out[..., 0] <= TARGET["frame_w"]) &
              (out[..., 1] >= 0) & (out[..., 1] <= TARGET["frame_h"]))
    lost = float((usable & ~inside).sum() / max(int(usable.sum()), 1))
    if drop_offscreen:
        use = use & inside
    return out.astype(np.float32), use, {"scale_applied": round(k, 4),
                                         "source_shoulder_px": round(sw, 1),
                                         "fraction_of_observed_points_lost": round(lost, 4)}


def build(xy, usable, ts):
    nat, present = to_native_like(xy, usable)
    span = active_span(present)
    if span is None:
        return None
    lo, hi = span
    return D.build_channels(nat[lo:hi + 1], present[lo:hi + 1], ts[lo:hi + 1])


def nikl_conditions():
    clips = {"S0": [], "S1": [], "S2": []}
    meta = {}
    for word_id, label in NIKL_TO_LABEL.items():
        p = os.path.join(NIKL_DIR, f"NIKL_{word_id}_01.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        xy, us, ts = z["xy_px"].astype(np.float32), z["usable"].astype(bool), z["timestamp_us"]
        base = build(xy, us, ts)
        r1 = reframe(xy, us, drop_offscreen=True)
        r2 = reframe(xy, us, drop_offscreen=False)
        if base is None or r1 is None or r2 is None:
            continue
        c1, c2 = build(r1[0], r1[1], ts), build(r2[0], r2[1], ts)
        for key, c in (("S0", base), ("S1", c1), ("S2", c2)):
            if c is None:
                continue
            c.update(clip_id=f"NIKL_{word_id}_01", true_label=label, label=0, condition=key)
            clips[key].append(c)
        meta[label] = {"npz_sha256": sha_file(p), **r1[2],
                       "hand_points_lost_offscreen": round(float(
                           (us[:, 33:75] & ~((r1[0][:, 33:75, 0] >= 0) &
                                             (r1[0][:, 33:75, 0] <= TARGET["frame_w"]) &
                                             (r1[0][:, 33:75, 1] >= 0) &
                                             (r1[0][:, 33:75, 1] <= TARGET["frame_h"]))).sum()
                           / max(int(us[:, 33:75].sum()), 1)), 4)}
    return clips, meta


def model_list():
    out = []
    for cond in ("A", "B"):
        for seed in (20260912, 20260913, 20260914):
            p = os.path.join(ROOT, "runs", "observation_gap_v1", f"{cond}_seed{seed}_final.pt")
            if os.path.exists(p):
                out.append((f"obsgap_{cond}_seed{seed}", p, "masked"))
    p = os.path.join(ROOT, "runs", "expanded_v1", "model.pt")
    if os.path.exists(p):
        out.append(("expanded_v1_best", p, "plain"))
    return out


def load_model(path, kind):
    ck = torch.load(path, weights_only=True)
    cfg = ck.get("config", {})

    def need(k):
        if k in ck:
            return ck[k]
        if k in cfg:
            return cfg[k]
        raise KeyError(k)

    cls = MaskedSignTCN if kind == "masked" else SignTCN
    m = cls(in_channels=ck.get("in_channels", D.TOTAL_CHANNELS), width=need("width"),
            dilations=tuple(need("dilations")), kernel_size=need("kernel_size"),
            dropout=need("dropout"), n_classes=len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["classes"]


def predict_one(model, clip, classes, five_idx):
    x, pad, _ = D.collate([clip])
    with torch.no_grad():
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    order = np.argsort(-logit)
    return {"pred5": PROJECT_5[int(np.argmax([logit[i] for i in five_idx]))],
            "pred80": classes[int(order[0])],
            "rank_of_true_in_80": int(order.tolist().index(
                classes.index(clip["true_label"]))) + 1}


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RUN, exist_ok=True)
    torch.set_num_threads(4)
    clips, meta = nikl_conditions()
    models = model_list()
    assert len(models) == 7, f"expected 7 models, found {len(models)}"
    print(f"모델 {len(models)}개 / NIKL {len(clips['S0'])}단어 x 3조건  (학습 없음)")
    print(f"목표 기하: 어깨폭/화면높이 {TARGET['shoulder_over_height']}, "
          f"어깨중점 ({TARGET['mid_x_fraction']}, {TARGET['mid_y_fraction']}), "
          f"프레임 {int(TARGET['frame_w'])}x{int(TARGET['frame_h'])}")
    print("\n재배치로 화면 밖으로 사라진 손 좌표 비율")
    for w, m in meta.items():
        print(f"  {w:<8} 배율 {m['scale_applied']:.2f}   전체 좌표 {m['fraction_of_observed_points_lost']:.3f}"
              f"   손 좌표만 {m['hand_points_lost_offscreen']:.3f}")

    per_model = {}
    for name, path, kind in models:
        model, classes = load_model(path, kind)
        five = [classes.index(w) for w in PROJECT_5]
        entry = {"checkpoint_sha256": sha_file(path)}
        for cond in ("S0", "S1", "S2"):
            entry[cond] = {c["true_label"]: predict_one(model, c, classes, five)
                           for c in clips[cond]}
        per_model[name] = entry

    def tally(cond, word, key):
        return sum(1 for m in per_model.values()
                   if m[cond].get(word, {}).get(key.replace("correct", "pred")) == word)

    n = len(per_model)
    s2_matches_s0 = all(m["S2"][w] == m["S0"][w] for m in per_model.values()
                        for w in m["S0"] if w in m["S2"])
    valid = {"S0_alda_five": tally("S0", "알다", "pred5") == n,
             "S2_identical_to_S0": s2_matches_s0}
    a5, a80 = tally("S1", "알다", "pred5"), tally("S1", "알다", "pred80")

    def verdict(c):
        if c <= 2:
            return "CAPTURE_FRAMING_IMPLICATED"
        if c >= 5:
            return "CAPTURE_FRAMING_RULED_OUT"
        return "INCONCLUSIVE"

    ok = all(valid.values())
    v5 = verdict(a5) if ok else "VOID_VALIDITY_FAILED"
    v80 = verdict(a80) if ok else "VOID_VALIDITY_FAILED"

    print(f"\n유효성 확인")
    print(f"  S0 알다 7/7 재현: {'통과' if valid['S0_alda_five'] else '실패'}")
    print(f"  S2 == S0 (크기만 바꾼 것은 무영향): {'통과' if valid['S2_identical_to_S0'] else '실패'}")
    print(f"\n본 판정 — S1(참가자 기하 + 화면 이탈 손실) 에서 알다를 맞힌 모델 수")
    print(f"  5후보 {a5}/{n}  ->  {v5}")
    print(f"  80후보 {a80}/{n}  ->  {v80}")

    print(f"\n{'모델':<22}{'S0 알다':>10}{'S1 알다':>10}{'S2 알다':>10}{'S1 감사합니다':>14}")
    for name, m in per_model.items():
        print(f"{name:<22}{m['S0']['알다']['pred5']:>10}{m['S1']['알다']['pred5']:>10}"
              f"{m['S2']['알다']['pred5']:>10}{m['S1']['감사합니다']['pred5']:>14}")

    out = {
        "task_id": "capture_simulation_v1", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does placing the interpreter at the participant's seating distance, and "
                    "losing whatever leaves the frame, break recognition",
        "why_no_recording": "the participant cannot be asked to sit at a fixed distance; the game "
                            "must work from any webcam. This asks the same question from data "
                            "that already exists.",
        "command": "py -3.13 capture_simulation_run.py",
        "no_training": True, "no_new_recording": True, "reserved_test_loaded": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "target_geometry_measured_from": "front_keyphase_v1/runs/ses_cbd462050a95, 100 clips",
        "target_geometry": TARGET,
        "conditions": {"S0": "as recorded", "S1": "participant geometry, off-frame points lost",
                       "S2": "participant geometry, nothing lost"},
        "reframing_effect_per_word": meta,
        "pre_registered_rule": {
            "validity": "S0 알다 must be 7/7 and S2 must equal S0 on every clip",
            "implicated": "S1 알다 <= 2/7", "ruled_out": "S1 알다 >= 5/7",
            "inconclusive": "3-4/7",
            "both_readings_reported": True},
        "validity_check": valid,
        "verdict": {"five_candidate": v5, "eighty_candidate": v80,
                    "s1_alda_counts": {"five": f"{a5}/{n}", "eighty": f"{a80}/{n}"}},
        "per_model": per_model,
        "cannot_reproduce": ["close-range perspective distortion, which 2D landmarks cannot "
                             "simulate", "any change in frame rate; the participant's ALDA ran at "
                             "31.9 fps against NIKL's 30.0, so frame rate is not a suspect here"],
        "limits": ["one NIKL clip per word; seven models bound model variation, not clip variation",
                   "the target geometry is the participant's median, not any single recording",
                   "agreement with a word name is not a verdict that a performance was well formed"],
        "web_recogniser_unchanged": True, "not_a_correctness_claim": True,
    }
    p = os.path.join(RUN, "result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
