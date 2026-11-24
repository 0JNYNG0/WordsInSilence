"""Read existing public metadata only; no video/private access or model execution."""
import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
OPS = OUT.parent
ROOT = OPS.parents[2]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

inputs = {
    "sources": OPS / "word100_rtmw_joint_learning_20261108/SOURCES.json",
    "input_verification": OPS / "word100_rtmw_joint_learning_20261108/INPUT_VERIFICATION.json",
    "extraction": OPS / "word100_rtmw_joint_learning_20261108/EXTRACTION_COMPLETE.json",
    "guides": OPS / "word100_guide_video_feedback_20261028/GUIDE_CARDS.json",
    "prior_label_audit": OPS / "word100_feedback_learning_plan_20261109/AUDIT.json",
}
sources = read(inputs["sources"])
verification = read(inputs["input_verification"])
extraction = read(inputs["extraction"])
guides = read(inputs["guides"])
labels = read(inputs["prior_label_audit"])["annotation_counts"]["ym"]
roles = dict(collections.Counter(x["role"] for x in sources))
feedback = [x for x in sources if x["feedback"]]
candidate_rows = []
for card in guides["cards"]:
    rows = [x for x in sources if x["word_id"] == card["word_id"]]
    candidate_rows.append({
        "word_id": card["word_id"], "label": card["label"],
        "roles": dict(collections.Counter(x["role"] for x in rows)),
        "candidate_entry_ids": [x["entry_id"] for x in card["candidates"]],
        "historical_card_fully_verified": card.get("full_guide_match_verified"),
        "warning": "Historical source candidates, not new verified temporal labels; consult later variant-specific evidence.",
    })
checks = {
    "1199_unique_clips": len(sources) == len({x["clip_id"] for x in sources}) == 1199,
    "100_words": len({x["word_id"] for x in sources}) == 100,
    "899_fit_300_dev": roles == {"fit": 899, "development": 300},
    "feedback_36_fit_12_dev": dict(collections.Counter(x["role"] for x in feedback)) == {"fit": 36, "development": 12},
    "feedback_4_words": len({x["word_id"] for x in feedback}) == 4,
    "9_fit_people": len({x["person"] for x in feedback if x["role"] == "fit"}) == 9,
    "3_dev_people": len({x["person"] for x in feedback if x["role"] == "development"}) == 3,
    "disjoint_people": not ({x["person"] for x in sources if x["role"] == "fit"} & {x["person"] for x in sources if x["role"] == "development"}),
    "150476_three_metadata_totals": sum(x["frames_header"] for x in sources) == extraction["frames"] == sum(x["frames"] for x in verification["rows"]) == 150476,
    "prior_arithmetic_verification_pass": verification["status"] == "PASS",
    "12_guide_candidates": len(candidate_rows) == 12,
    "all_candidates_9_fit_3_dev": all(c["roles"] == {"fit": 9, "development": 3} for c in candidate_rows),
    "left_open_fit_7": labels["fit"]["2:1"]["labels"] == 7,
    "left_open_fit_3_people_1_expression": len(labels["fit"]["2:1"]["people"]) == 3 and len(labels["fit"]["2:1"]["words"]) == 1,
    "left_open_dev_1": labels["development"]["2:1"]["labels"] == 1,
}
audit = {
    "status": "REVIEW_ONLY_NO_TRAINING", "roles": roles,
    "feedback_roles": dict(collections.Counter(x["role"] for x in feedback)),
    "person_ids_by_role": {r: sorted({x["person"] for x in sources if x["role"] == r}) for r in roles},
    "frames": extraction["frames"], "candidate_guides": candidate_rows,
    "left_open_positive": {r: labels[r]["2:1"] for r in ("fit", "development")},
    "inputs": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in inputs.items()},
    "checks": checks,
    "verification_scope": "Metadata reconciliation and prior label-count audit; not visual joint accuracy, independent human labels or new performance measurement.",
    "new_training": False, "new_inference": False, "private_data_access": False,
}
(OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
assert all(checks.values()), {k:v for k,v in checks.items() if not v}
print(json.dumps({"status":"PASS","metadata_checks":len(checks),"candidate_words":len(candidate_rows),"model_runs":0},ensure_ascii=False))
