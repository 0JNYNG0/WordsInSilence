#!/usr/bin/env python3
"""Separate the performer from the other suspects, using models that already exist.

Design: design/performer_separation_experiment.md, written before this script.

The participant's ALDA fails. AIHub differs from it in three ways at once - extractor, sentence-cut
versus isolated, and fluent signer versus learner - so nothing could be attributed. The NIKL
interpreter clips sit between: MediaPipe, isolated, and the very video the participant imitated, so
they share the participant's extractor, clip format and lexical variant, and differ only in skill.

No training happens here. Seven already-trained models are loaded and asked about 45 clips.

Decision rule, fixed in the design:
  validity  NIKL 감사합니다 must be recognised by at least 5 of 7 models, otherwise the run is
            recorded as VOID and no performer verdict is given
  >= 5/7    the performer is implicated
  <= 2/7    the performer is ruled out
  3-4/7     inconclusive
The design did not say whether to apply the count to the five-candidate or the eighty-candidate
reading, so BOTH are evaluated and both verdicts are reported. Picking one now would be choosing a
criterion with the numbers already in view.

Run: py -3.13 performer_separation_run.py
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

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.model_masked_v1 import MaskedSignTCN  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

FRONT = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1"))
NIKL_DIR = os.path.join(FRONT, "cache", "2683eadbc5f1c588")
SESSION = os.path.join(FRONT, "runs", "ses_cbd462050a95")
RUN = os.path.join(ROOT, "runs", "performer_separation_v1")
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
NIKL_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다", "NA": "나",
                 "JOTA": "좋다", "MOREUDA": "모르다"}
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}
VALIDITY_WORD = "감사합니다"
VALIDITY_MIN = 5
IMPLICATE_MIN = 5
EXONERATE_MAX = 2


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_paths():
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
    """The two runs saved different key layouts: expanded_v1 put the architecture at the top
    level, observation_gap_v1 put it under "config". Read either without guessing a default."""
    ck = torch.load(path, weights_only=True)
    classes = ck["classes"]
    cfg = ck.get("config", {})

    def need(key):
        if key in ck:
            return ck[key]
        if key in cfg:
            return cfg[key]
        raise KeyError(f"{key} missing from {os.path.basename(path)}")

    cls = MaskedSignTCN if kind == "masked" else SignTCN
    m = cls(in_channels=ck.get("in_channels", D.TOTAL_CHANNELS), width=need("width"),
            dilations=tuple(need("dilations")), kernel_size=need("kernel_size"),
            dropout=need("dropout"), n_classes=len(classes))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, classes


def nikl_clips():
    out = []
    for word_id, label in NIKL_TO_LABEL.items():
        p = os.path.join(NIKL_DIR, f"NIKL_{word_id}_01.npz")
        if not os.path.exists(p):
            continue
        c = load_web_clip(p)
        c.update(clip_id=f"NIKL_{word_id}_01", true_label=label, source="nikl_interpreter",
                 label=0, npz_sha256=sha_file(p))
        out.append(c)
    return out


def web_clips():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        w = r.get("requested_word_id")
        if r.get("observation_status") != "OK" or w not in WEB_TO_LABEL:
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if os.path.exists(p):
            c = load_web_clip(p)
            c.update(clip_id=r["trial_id"], true_label=WEB_TO_LABEL[w],
                     source="participant_web", label=0)
            out.append(c)
    return out


def score(model, clips, classes):
    five = [classes.index(w) for w in PROJECT_5]
    rows = []
    for c in clips:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        order = np.argsort(-logit)
        rows.append({
            "clip_id": c["clip_id"], "source": c["source"], "true": c["true_label"],
            "pred5": PROJECT_5[int(np.argmax([logit[i] for i in five]))],
            "pred80": classes[int(order[0])],
            "rank_of_true_in_80": int(order.tolist().index(classes.index(c["true_label"]))) + 1,
            "top3_of_80": [classes[i] for i in order[:3]],
        })
    return rows


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RUN, exist_ok=True)
    torch.set_num_threads(4)
    models = model_paths()
    assert len(models) == 7, f"expected 7 trained models, found {len(models)}"
    nikl, web = nikl_clips(), web_clips()
    print(f"모델 {len(models)}개, NIKL {len(nikl)}건, 참가자 {len(web)}건  (학습 없음)")

    per_model = {}
    for name, path, kind in models:
        model, classes = load_model(path, kind)
        rows = score(model, nikl, classes) + score(model, web, classes)
        n = {r["true"]: r for r in rows if r["source"] == "nikl_interpreter"}
        wc = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            if r["source"] != "participant_web":
                continue
            v = wc[r["true"]]
            v[2] += 1
            v[0] += int(r["pred5"] == r["true"])
            v[1] += int(r["pred80"] == r["true"])
        per_model[name] = {
            "checkpoint": os.path.relpath(path, ROOT).replace("\\", "/"),
            "checkpoint_sha256": sha_file(path), "class_count": len(classes),
            "nikl": {w: {"pred5": r["pred5"], "pred80": r["pred80"],
                         "rank_of_true_in_80": r["rank_of_true_in_80"],
                         "top3_of_80": r["top3_of_80"],
                         "correct5": r["pred5"] == w, "correct80": r["pred80"] == w}
                     for w, r in n.items()},
            "participant": {w: {"five": f"{v[0]}/{v[2]}", "eighty": f"{v[1]}/{v[2]}"}
                            for w, v in wc.items()},
            "per_clip": rows,
        }

    def tally(word, key):
        return sum(1 for m in per_model.values() if m["nikl"].get(word, {}).get(key))

    valid5, valid80 = tally(VALIDITY_WORD, "correct5"), tally(VALIDITY_WORD, "correct80")
    alda5, alda80 = tally("알다", "correct5"), tally("알다", "correct80")

    def verdict(count):
        if count >= IMPLICATE_MIN:
            return "PERFORMER_IMPLICATED"
        if count <= EXONERATE_MAX:
            return "PERFORMER_RULED_OUT"
        return "INCONCLUSIVE"

    n_models = len(per_model)
    validity_ok = {"five_candidate": valid5 >= VALIDITY_MIN,
                   "eighty_candidate": valid80 >= VALIDITY_MIN}
    result_five = (verdict(alda5) if validity_ok["five_candidate"] else "VOID_VALIDITY_FAILED")
    result_eighty = (verdict(alda80) if validity_ok["eighty_candidate"]
                     else "VOID_VALIDITY_FAILED")

    print(f"\n유효성 확인 — NIKL {VALIDITY_WORD} 를 맞힌 모델 수")
    print(f"  5후보 {valid5}/{n_models}  (기준 {VALIDITY_MIN} 이상: "
          f"{'통과' if validity_ok['five_candidate'] else '실패'})")
    print(f"  80후보 {valid80}/{n_models}  (기준 {VALIDITY_MIN} 이상: "
          f"{'통과' if validity_ok['eighty_candidate'] else '실패'})")
    print(f"\n본 판정 — NIKL 알다 를 맞힌 모델 수")
    print(f"  5후보 {alda5}/{n_models}  ->  {result_five}")
    print(f"  80후보 {alda80}/{n_models}  ->  {result_eighty}")

    print(f"\n{'모델':<22} {'NIKL 알다':>22} {'NIKL 감사':>22} {'참가자 알다(5후보)':>18}")
    for name, m in per_model.items():
        a = m["nikl"].get("알다", {})
        g = m["nikl"].get("감사합니다", {})
        print(f"{name:<22} {str(a.get('pred5')) + '/' + str(a.get('pred80')):>22} "
              f"{str(g.get('pred5')) + '/' + str(g.get('pred80')):>22} "
              f"{m['participant'].get('알다', {}).get('five', '-'):>18}")

    print(f"\nNIKL 나머지 3단어 (맥락)")
    for w in ("나", "좋다", "모르다"):
        c5 = tally(w, "correct5")
        c80 = tally(w, "correct80")
        print(f"  {w:<6} 5후보 {c5}/{n_models}   80후보 {c80}/{n_models}")

    out = {
        "task_id": "performer_separation_v1", "date_kst": time.strftime("%Y-%m-%d"),
        "design": "design/performer_separation_experiment.md",
        "design_sha256": sha_file(os.path.join(ROOT, "design",
                                               "performer_separation_experiment.md")),
        "command": "py -3.13 performer_separation_run.py",
        "no_training": True, "no_new_recording": True, "reserved_test_loaded": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "logic": {
            "why_nikl": "the NIKL interpreter clips share the participant's extractor "
                        "(MediaPipe), clip format (isolated) and lexical variant (the participant "
                        "imitated these very videos), and differ only in skill",
            "asymmetry": "a success is decisive because skill is the only remaining difference; a "
                         "failure is NOT, because extractor, sentence-cut versus isolated, and a "
                         "possible NIKL-versus-AIHub variant difference all remain",
        },
        "inputs": {"nikl_clips": len(nikl), "nikl_source": "front_keyphase_v1/cache/"
                   "2683eadbc5f1c588", "nikl_per_word": 1,
                   "participant_clips": len(web), "participant_words": sorted(WEB_TO_LABEL.values()),
                   "nikl_sha256": {c["clip_id"]: c["npz_sha256"] for c in nikl}},
        "models": [{"name": n, "checkpoint": per_model[n]["checkpoint"],
                    "sha256": per_model[n]["checkpoint_sha256"]} for n, _, _ in models],
        "pre_registered_rule": {
            "validity": f"NIKL {VALIDITY_WORD} recognised by at least {VALIDITY_MIN} of "
                        f"{n_models} models",
            "implicated": f">= {IMPLICATE_MIN}/{n_models}",
            "ruled_out": f"<= {EXONERATE_MAX}/{n_models}",
            "inconclusive": "anything between",
            "metric_ambiguity": "the design did not fix whether to count the five-candidate or the "
                                "eighty-candidate reading, so both are reported and neither is "
                                "chosen after the fact",
        },
        "validity_check": {"word": VALIDITY_WORD, "five_candidate": f"{valid5}/{n_models}",
                           "eighty_candidate": f"{valid80}/{n_models}", "passed": validity_ok},
        "verdict": {"five_candidate": result_five, "eighty_candidate": result_eighty,
                    "alda_counts": {"five": f"{alda5}/{n_models}",
                                    "eighty": f"{alda80}/{n_models}"}},
        "context_other_nikl_words": {w: {"five": f"{tally(w, 'correct5')}/{n_models}",
                                         "eighty": f"{tally(w, 'correct80')}/{n_models}"}
                                     for w in ("나", "좋다", "모르다")},
        "per_model": per_model,
        "exposure": {
            "nikl_clips": "these 19 clips are the web tester's reference set and were used to "
                          "derive the per-word flexion bands. They have never been in any TCN "
                          "training set, but they are project-exposed data and cannot serve as an "
                          "independent test for anything designed from them",
            "participant_clips": "attempt_label is unset for this session, so it is not "
                                 "established that each attempt was a correct performance",
        },
        "limits": [
            "one NIKL clip per word; the seven-model agreement bounds model-to-model variation, "
            "not clip-to-clip variation",
            "a failure leaves three suspects entangled, not one",
            "agreement with a word name is not a verdict that any attempt was well formed",
        ],
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
