"""functional_14: a reproducible consistency list for the 19 project words.

Four separate questions per word, deliberately not merged:

  1. demonstration source and hand-form review state - where the reference clip came from and
     whether its form has been checked;
  2. whether the existing training material holds it;
  3. whether the PINNED model was trained on it;
  4. whether more video has to be looked at before anything can be said.

"the current model never learned it" and "AIHub has no such source label" are DIFFERENT facts and
are reported in different fields. The existing vocabulary scans are consulted for the second, and
where they cannot settle it the word is UNDETERMINED rather than absent.

For the five shared words a demonstration comparison sample is pinned in advance - split=train,
REAL01, first by ascending clip_id - so the choice cannot drift after seeing anything. These are
COMPARISON SAMPLES, not a performance report: no new attempt, no validation split and no reserved
test is read.

The form review and the source-vocabulary answer are an AI ASSISTANT reviewer's results
(ChatGPT Work / Codex), not a person's and not a sign-language expert's - the record carries
reviewer.kind=ai_assistant and human_expert_validation=false. They are read from their lineage
copies under evidence/external/; this generator adds no verdict of its own.

Reads only: config/words.json, config/source_manifest.json, the pinned checkpoint's class list
(hash-verified, weights_only=True, never run), expand/manifest.json, expand/vocab_scan.json,
evidence/aihub_vocab_scan.json and the evidence/external copies.
No video is re-extracted, no model is run, no training, no per-word rule, no threshold, no web
change, no download. Public aggregates only; raw arrays and source video stay local.

Run: py -3.13 tools_training_demo_consistency.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1")
sys.path.insert(0, os.path.join(ROOT, "src"))

# split=train, REAL01, first by ascending clip_id. Fixed here BEFORE any comparison is made.
PINNED_SAMPLES = {
    "GAMSAHAMNIDA": [{"sentence": "NIA_SL_SEN0355", "start_s": 1.253, "end_s": 2.333}],
    "MOREUDA": [{"sentence": "NIA_SL_SEN0195", "start_s": 2.781, "end_s": 4.071}],
    "ALDA": [{"sentence": "NIA_SL_SEN0196", "start_s": 1.837, "end_s": 2.951}],
    "JOTA": [{"sentence": "NIA_SL_SEN0189", "start_s": 1.356, "end_s": 2.262}],
    "NA": [{"sentence": "NIA_SL_SEN0019", "start_s": 1.422, "end_s": 2.484},
           {"sentence": "NIA_SL_SEN0020", "start_s": 1.319, "end_s": 2.322},
           {"sentence": "NIA_SL_SEN0199", "start_s": 2.914, "end_s": 3.568}],
}
# The form review and the source-vocabulary answer are ASSISTANT-REVIEWED external observations -
# the reviewer is an AI assistant (ChatGPT Work / Codex), NOT a person and NOT a sign-language
# expert, and no human expert has validated them. They are read from their evidence copies under
# evidence/external/ so this generator carries no verdict of its own; each copy records where it
# came from and the origin's hash.
EXTERNAL = os.path.join(ROOT, "evidence", "external")
VISUAL_REVIEW = os.path.join(EXTERNAL, "five_word_visual_review.json")
VOCAB_AUDIT = os.path.join(EXTERNAL, "full_source_vocabulary_audit_v2.json")
VOCAB_AUDIT_FAILED_FIRST = os.path.join(EXTERNAL, "full_source_vocabulary_audit_v1_failed.json")
ACQUISITION = os.path.join(EXTERNAL, "four_word_source_acquisition.json")

# the pinned checkpoint, verified by hash before its class list is read
CHECKPOINT_SHA256 = "56a17a4d0a4795a2463add5572acc0d985c3cb7a1cbaee2853698f106d5b34b5"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def checkpoint_classes() -> tuple:
    """The pinned checkpoint's class names. Hash-verified, tensors only, never run."""
    import torch

    from wis_lab.service import neural
    got = sha256_file(neural.CHECKPOINT)
    if got != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint hash mismatch: {got}")
    # weights_only=True refuses to unpickle arbitrary objects; the class list is plain strings
    ck = torch.load(neural.CHECKPOINT, map_location="cpu", weights_only=True)
    return list(ck["classes"]), got


def form_review_rows() -> dict:
    """The external visual observation, as recorded by an AI-assistant reviewer.

    Not a person's judgement and not a sign-language expert's: the reviewer's own record says
    kind=ai_assistant and human_expert_validation=false, and both travel with every row so no
    reader can mistake the standing of these observations. This tool adds no judgement of its own.
    """
    review = load(VISUAL_REVIEW)
    reviewer = review.get("reviewer") or {}
    out = {}
    for r in review["rows"]:
        out[r["word_id"]] = {
            "state": r["visual_form_status"],
            "observation": r["observation"],
            "source": "evidence/external/five_word_visual_review.json",
            "method": review["method"],
            "reviewer_kind": reviewer.get("kind"),
            "reviewer_name": reviewer.get("name"),
            "human_expert_validation": reviewer.get("human_expert_validation"),
            "is_correctness_verdict": review["is_correctness_verdict"],
            "is_training_approval": review["is_training_approval"],
        }
    return out


def main() -> int:
    words = load(os.path.join(ROOT, "config", "words.json"))
    src = load(os.path.join(ROOT, "config", "source_manifest.json"))
    manifest = load(os.path.join(PILOT, "expand", "manifest.json"))
    scan = load(os.path.join(PILOT, "expand", "vocab_scan.json"))
    scan2 = load(os.path.join(PILOT, "evidence", "aihub_vocab_scan.json"))
    audit = load(VOCAB_AUDIT)
    audit_by_word = {w["word_id"]: w for w in audit["words"]}
    reviewed = form_review_rows()
    classes, checkpoint_hash = checkpoint_classes()

    clips = {c["word_id"]: c for c in src["clips"]}
    by_label = {v["label"]: v for v in scan["vocabulary"]}
    listed = len(scan["vocabulary"])
    distinct = scan["totals"]["distinct_labels"]
    min_listed_instances = min(v["instances"] for v in scan["vocabulary"])

    train_counts, split_counts = {}, {}
    for m in manifest:
        train_counts.setdefault(m["word_label"], {}).setdefault(m["split_role"], 0)
        train_counts[m["word_label"]][m["split_role"]] += 1
        split_counts.setdefault(m["split_role"], 0)
        split_counts[m["split_role"]] += 1

    rows = []
    for w in words["words"]:
        wid, label = w["word_id"], w["display_name"]
        clip = clips.get(wid)
        scanned = by_label.get(label)
        splits = train_counts.get(label)

        # the full untruncated audit is the authority; the older filtered scan is kept beside it
        a = audit_by_word.get(wid, {})
        presence = {
            "status": a.get("status"),
            "exact_labels_checked": a.get("exact_labels_checked"),
            "annotations_in_this_archive_front_view": a.get("all_annotation_count"),
            "signers": a.get("all_signer_count"),
            "annotations_in_training_signers_01_to_09": a.get(
                "training_signers_01_to_09_annotation_count"),
            "scope": audit["scope"],
            "means": "this answers ONLY the exact headword and the configured aliases, front view, "
                     "in THIS one archive. It is not absence from all of AIHub and not absence of "
                     "a synonym or a differently written form.",
            "source": "evidence/external/full_source_vocabulary_audit_v2.json",
            "earlier_filtered_scan": (
                {"status": "PRESENT_IN_SCAN", "instances": scanned["instances"],
                 "signers": scanned["signers"], "clips": scanned["clips"]} if scanned else
                {"status": "NOT_IN_THE_FILTERED_LIST",
                 "note": f"that scan listed {listed} of {distinct} labels with a smallest listed "
                         f"count of {min_listed_instances}; it could never settle a single word, "
                         f"which is why the untruncated audit above is used"}),
        }

        rows.append({
            "word_id": wid, "display_name": label,
            # 1. demonstration source and form review
            "demonstration": {
                "clip_id": (clip or {}).get("clip_id"),
                "source_kind": (clip or {}).get("source_kind"),
                "source_url": (clip or {}).get("source_url"),
                "source_verification": (clip or {}).get("source_verification"),
                "readiness": (clip or {}).get("readiness"),
                "readiness_reason": (clip or {}).get("readiness_reason"),
                "cue_flags": w.get("cue_flags"),
                "form_review": reviewed.get(wid, {
                    "state": "NOT_COMPARED_ON_VIDEO",
                    "observation": "no frame comparison between the demonstration clip and any "
                                   "training sample has been made for this word",
                    "source": None}),
            },
            # 2. existing training material
            "training_material": {
                "in_expand_manifest": bool(splits),
                "samples_by_split": splits or {},
                "aihub_source_label": presence,
            },
            # 3. the pinned model
            "pinned_model": {
                "trained_on_this_word": label in classes,
                "note": "not trained is NOT the same as no source label existing; those are the "
                        "two fields above and to the left",
            },
            # 4. what still needs video
            "needs_more_video": (
                "PARTIAL - fixed frame samples were compared; full movement, timing and semantic "
                "equivalence remain unverified" if wid in reviewed else
                "YES - no demonstration-versus-training form comparison exists"
                if splits else
                "NOT POSSIBLE FROM THIS ARCHIVE - no exact-label source material was found in it"),
            "pinned_comparison_samples": PINNED_SAMPLES.get(wid, []),
        })

    out = {
        "task_id": "functional_14_training_demo_consistency",
        "purpose": "a reproducible list separating four questions per word: demonstration source "
                   "and form review, existing training material, the pinned model's vocabulary, "
                   "and what still needs video",
        "is_not": ["a performance report", "a recognition result", "a per-word rule",
                   "a threshold", "a reason to retrain"],
        "sample_selection_rule": "split=train, REAL01, first by ascending clip_id, FIXED before "
                                 "any comparison. These are demonstration comparison samples; no "
                                 "new attempt, no validation split and no reserved test is read.",
        "inputs": {
            "config/words.json": sha256_file(os.path.join(ROOT, "config", "words.json")),
            "config/source_manifest.json": sha256_file(
                os.path.join(ROOT, "config", "source_manifest.json")),
            "expand/manifest.json": sha256_file(os.path.join(PILOT, "expand", "manifest.json")),
            "expand/vocab_scan.json": sha256_file(os.path.join(PILOT, "expand",
                                                               "vocab_scan.json")),
            "evidence/aihub_vocab_scan.json": sha256_file(
                os.path.join(PILOT, "evidence", "aihub_vocab_scan.json")),
            "evidence/external/five_word_visual_review.json": sha256_file(VISUAL_REVIEW),
            "evidence/external/full_source_vocabulary_audit_v2.json": sha256_file(VOCAB_AUDIT),
            "evidence/external/full_source_vocabulary_audit_v1_failed.json": sha256_file(
                VOCAB_AUDIT_FAILED_FIRST),
            "evidence/external/four_word_source_acquisition.json": sha256_file(ACQUISITION),
        },
        "checkpoint_classes": {"count": len(classes), "read_only_no_forward_pass": True,
                               "sha256_verified": checkpoint_hash,
                               "torch_load": "weights_only=True"},
        "external_evidence": {
            "visual_form_review": "evidence/external/five_word_visual_review.json",
            "source_acquisition_receipt": "evidence/external/four_word_source_acquisition.json",
            "full_vocabulary_audit": "evidence/external/full_source_vocabulary_audit_v2.json",
            "first_audit_attempt_FAILED": "evidence/external/"
                                          "full_source_vocabulary_audit_v1_failed.json",
            "first_attempt_note": "the first pass could not handle the _REAL11_U_F and _REAL14_U_F "
                                  "name variants and is preserved as a failure; v2 handles both "
                                  "and nothing was re-scanned",
            "reviewer_owns_these": "the form review and the acquisition are the reviewer's work; "
                                   "this generator copies their lineage and adds no verdict",
            "reviewer_kind": "ai_assistant (ChatGPT Work / Codex). NOT a person and NOT a "
                             "sign-language expert; human_expert_validation is false"},
        "vocab_scan_scope": {
            "listed_labels": listed, "distinct_labels": distinct,
            "smallest_listed_instances": min_listed_instances,
            "filters": scan.get("filters"),
            "second_scan_has_no_per_word_list": "aihub_vocab_scan.json carries threshold "
                                                "aggregates only, so it cannot settle a single "
                                                "word either",
            "second_scan_totals": scan2.get("by_threshold", {}).get("signers>=16&instances>=1"),
        },
        "training_split_totals": split_counts,
        "words": rows,
        "summary": {
            "total": len(rows),
            "trained_by_pinned_model": sum(1 for r in rows
                                           if r["pinned_model"]["trained_on_this_word"]),
            "in_training_material": sum(1 for r in rows
                                        if r["training_material"]["in_expand_manifest"]),
            "exact_label_present_in_this_archive": sum(
                1 for r in rows
                if (r["training_material"]["aihub_source_label"]
                    ["annotations_in_this_archive_front_view"] or 0) > 0),
            "exact_label_absent_in_this_archive": sum(
                1 for r in rows
                if (r["training_material"]["aihub_source_label"]
                    ["annotations_in_this_archive_front_view"] or 0) == 0),
            "form_observations_are_assistant_reviewed_not_human_expert": True,
            "form_reviewed_on_fixed_frames": sum(
                1 for r in rows
                if r["demonstration"]["form_review"]["state"] != "NOT_COMPARED_ON_VIDEO"),
            "form_difference_observed": sum(
                1 for r in rows
                if r["demonstration"]["form_review"]["state"] == "DIFFERENCE_OBSERVED"),
        },
        "no_training": True, "no_inference": True, "no_web_change": True,
        "no_per_word_exception": True, "no_threshold": True, "no_download": True,
        "no_video_re_extraction": True,
    }
    path = os.path.join(ROOT, "evidence", "functional_14_training_demo_consistency.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"summary": out["summary"], "vocab_scan_scope": {
        k: out["vocab_scan_scope"][k] for k in ("listed_labels", "distinct_labels",
                                                "smallest_listed_instances")},
        "written": os.path.relpath(path, ROOT)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
