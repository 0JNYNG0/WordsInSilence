"""Versioned conservative motion-mask audit. Never authorizes model training.

The original dataset, label drafts, and failed reliability audit are read-only.
This removes remaining binary negatives from unresolved *categorical* motion
judgments. It neither adjudicates motion nor infers a sequence from endpoints.
Only Python's standard library is used; no model or former runner is imported.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import datetime
import hashlib
import json
import sys

U = Path(__file__).resolve().parents[1]
VERSION = "motion_category_guard_v1"
STATUS = "NOT_TRAINING_READY_ORIGINAL_RELIABILITY_FAILED"
MOTIONS = ("up", "down", "horizontal", "still", "mixed")


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def unique_rows(rows):
    result = {row["sample_id"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate sample IDs")
    return result


def motion_judgment(annotation, hand):
    category = annotation["motion"][hand]
    roi = annotation["roi_window"][hand]
    require(category is None or category in MOTIONS, "Invalid motion category")
    require(roi is None or type(roi) is bool, "ROI must be bool or null")
    return category, roi


def guard_hand(primary, secondary, hand):
    """Only explicit whole-window categorical judgments can retain motion GT."""
    if secondary is None:
        return {"decision": "unchecked_no_new_secondary", "withhold": False}
    require(primary is not None, "Secondary annotation lacks corresponding primary")
    p, pr = motion_judgment(primary, hand)
    q, qr = motion_judgment(secondary, hand)
    reasons = []
    if p is None or q is None:
        reasons.append("category_unknown")
    if p is not None and q is not None and p != q:
        reasons.append("category_disagreement")
    if pr is not True or qr is not True:
        reasons.append("whole_window_roi_not_confirmed_by_both")
    return dict(decision="withheld_entire_hand_motion" if reasons else "retained_category_agreement",
                withhold=bool(reasons), reasons=reasons, primary=p, secondary=q,
                primary_roi_window=pr, secondary_roi_window=qr)


def transform(dataset, primary, secondary):
    """Preserve every row; only withdraw known motion cells, never add labels."""
    result = deepcopy(dataset)
    targets = dataset["targets"]
    require(len(targets) == len(set(targets)), "Duplicate target names")
    motion_indices = {h: [targets.index(f"{h}_motion_{m}") for m in MOTIONS] for h in "RL"}
    original_by_id = unique_rows(dataset["rows"])
    require(set(secondary) <= set(primary) <= set(original_by_id), "Label IDs outside dataset")
    changes, decisions = [], []
    for row in result["rows"]:
        sid = row["sample_id"]
        original = original_by_id[sid]
        require(len(row["y"]) == len(row["target_mask"]) == len(targets), "Target width mismatch")
        require(all(type(m) is bool for m in row["target_mask"]), "Target mask is not boolean")
        require(all(v in (0, 1) for v in row["y"]), "Non-binary stored value")
        require(all(m or v == 0 for v, m in zip(row["y"], row["target_mask"])),
                "Unknown storage must be zero with mask false")
        row_decisions = {}
        for hand in "RL":
            decision = guard_hand(primary.get(sid), secondary.get(sid), hand)
            decision.update(sample_id=sid, hand=hand, role=row["role"], clip_id=row["clip_id"],
                            reused_prior_targets=bool(row.get("reused_prior_targets")))
            row_decisions[hand] = deepcopy(decision)
            decisions.append(decision)
            if not decision["withhold"]:
                continue
            for j in motion_indices[hand]:
                name = targets[j]
                if row["target_mask"][j]:
                    changes.append(dict(sample_id=sid, hand=hand, role=row["role"], target=name,
                                        previous_value=row["y"][j], previous_mask=True,
                                        new_value=0.0, new_mask=False,
                                        previous_reason=row["reasons"].get(name),
                                        new_reason="motion_category_guard_unresolved_whole_window"))
                    row["reasons"][name] = "motion_category_guard_unresolved_whole_window"
                row["y"][j] = 0.0
                row["target_mask"][j] = False
            require(all(not row["target_mask"][j] for j in motion_indices[hand]),
                    "Unresolved motion retained a known binary negative")
        row["motion_category_guard"] = row_decisions
        for key, value in original.items():
            if key not in ("y", "target_mask", "reasons"):
                require(row[key] == value, f"Changed original row field: {sid}/{key}")
        for j, name in enumerate(targets):
            require(not row["target_mask"][j] or original["target_mask"][j], "Added known label")
            if "_motion_" not in name:
                require((row["y"][j], row["target_mask"][j], row["reasons"].get(name)) ==
                        (original["y"][j], original["target_mask"][j], original["reasons"].get(name)),
                        "Static shape/relation changed")
        if sid not in secondary:
            require(all(row[k] == v for k, v in original.items()), "Unchecked original label changed")
    require([r["sample_id"] for r in result["rows"]] == [r["sample_id"] for r in dataset["rows"]],
            "Rows reordered, added or removed")
    require(result["pins"] == dataset["pins"], "Original pins changed")
    return result, changes, decisions


def support(rows, targets, role):
    result = {}
    for j, name in enumerate(targets):
        counts = {}
        for value, label in ((1, "positive"), (0, "negative")):
            selected = [r for r in rows if r["role"] == role and r["target_mask"][j] and r["y"][j] == value]
            counts[label] = dict(windows=len(selected), clips=len({r["clip_id"] for r in selected}),
                                 people=len({r["person"] for r in selected}), words=len({r["word_id"] for r in selected}))
        counts["unknown_windows"] = sum(r["role"] == role and not r["target_mask"][j] for r in rows)
        result[name] = counts
    return result


def coverage(rows, targets, decisions, role):
    selected = [r for r in rows if r["role"] == role]
    dd = [d for d in decisions if d["role"] == role]
    mi = [j for j, n in enumerate(targets) if "_motion_" in n]
    return dict(windows=len(selected), clips=len({r["clip_id"] for r in selected}),
                people=len({r["person"] for r in selected}), words=len({r["word_id"] for r in selected}),
                total_hand_windows=len(dd), decisions=dict(Counter(d["decision"] for d in dd)),
                decision_counts_describe_guard_policy_not_original_known_bits=True,
                checked_new_hand_windows=sum(d["decision"] != "unchecked_no_new_secondary" for d in dd),
                unchecked_reused_hand_windows=sum(d["decision"] == "unchecked_no_new_secondary" and d["reused_prior_targets"] for d in dd),
                unchecked_new_hand_windows=sum(d["decision"] == "unchecked_no_new_secondary" and not d["reused_prior_targets"] for d in dd),
                known_motion_cells=sum(r["target_mask"][j] for r in selected for j in mi),
                unknown_motion_cells=sum(not r["target_mask"][j] for r in selected for j in mi),
                all_windows_retained_for_coverage=True)


def selftest():
    """Small synthetic policy cases; no video/model/previous-score execution."""
    checks = []
    def annotation(r="still", l="still", rr=True, lr=True):
        return dict(motion={"R": r, "L": l}, roi_window={"R": rr, "L": lr})
    require(guard_hand(annotation(), annotation(), "R")["decision"] == "retained_category_agreement", "Stationary rejected")
    checks.append("valid_full_window_stationary_agreement_retained")
    for a, b in [(annotation("up"), annotation("down")), (annotation(None), annotation()),
                 (annotation(), annotation(None)), (annotation(None), annotation(None)),
                 (annotation(rr=False), annotation()), (annotation(), annotation(rr=False)),
                 (annotation(rr=None), annotation())]:
        require(guard_hand(a, b, "R")["withhold"], "Unresolved hand motion accepted")
    checks.append("category_mismatch_unknown_and_roi_fail_withhold")
    # Identical endpoints can describe a hold or an excursion with return.
    hold = [0] * 13
    excursion = [0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
    require((hold[0], hold[-1]) == (excursion[0], excursion[-1]) and hold != excursion, "Bad endpoint test")
    a, b = annotation("still"), annotation("mixed")
    a["endpoint_positions"] = b["endpoint_positions"] = [0, 0]
    require(guard_hand(a, b, "R")["withhold"], "Endpoints overrode whole-window disagreement")
    a = annotation(None)
    a.update(endpoint_positions=[0, 0], center_motion="still")
    require(guard_hand(a, annotation(), "R")["withhold"], "Endpoint/center proxy filled unknown sequence")
    checks.append("same_endpoints_cannot_infer_intermediate_sequence_or_still")
    names = ["R_open_four", "R_higher"] + [f"{h}_motion_{m}" for h in "RL" for m in MOTIONS]
    vals = [1, 0] + [0, 0, 0, 1, 0] * 2
    sample = dict(sample_id="synthetic", role="fit", clip_id="clip", person=1, word_id="word",
                  y=vals, target_mask=[True] * len(names), reasons={n: "original" for n in names},
                  input="unchanged.npz", input_sha256="unchanged", frame_ids=list(range(13)), center_frame=6)
    data = dict(rows=[sample], targets=names, pins={"source": "unchanged"})
    # Emulate legacy predicate-level resolution: two disputed bits were already
    # unknown but three common negatives remained known.
    data["rows"][0]["y"][5] = 0
    data["rows"][0]["target_mask"][2] = False
    data["rows"][0]["target_mask"][5] = False
    before = deepcopy(data)
    out, changed, _ = transform(data, {"synthetic": annotation("up")}, {"synthetic": annotation("still")})
    require(len(changed) == 3 and not any(out["rows"][0]["target_mask"][2:7]), "Known negative leftovers")
    require(out["rows"][0]["y"][:2] == sample["y"][:2] and out["rows"][0]["target_mask"][7:] == [True] * 5,
            "Static or opposite hand changed")
    require(data == before, "In-place input mutation")
    checks.append("all_five_unresolved_bits_unknown_static_and_other_hand_unchanged")
    unreviewed, changed, decisions = transform(data, {}, {})
    require(not changed and all(d["decision"] == "unchecked_no_new_secondary" for d in decisions), "Unchecked treated as agreement")
    require(all(unreviewed["rows"][0][k] == v for k, v in sample.items()), "Unreviewed original changed")
    checks.append("unchecked_old_or_new_rows_unchanged_and_explicitly_marked")
    stationary_data = deepcopy(data)
    stationary_data["rows"][0]["y"] = vals.copy()
    stationary_data["rows"][0]["y"][5] = 1
    stationary_data["rows"][0]["target_mask"] = [True] * len(names)
    stationary, changed, _ = transform(stationary_data, {"synthetic": annotation()}, {"synthetic": annotation()})
    require(not changed and stationary["rows"][0]["target_mask"][5] and stationary["rows"][0]["y"][5] == 1,
            "Valid still label lost")
    checks.append("valid_stationary_one_hot_remains_known_positive")
    for bad in [annotation("sideways"), annotation(rr=1)]:
        try:
            guard_hand(bad, annotation(), "R")
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed category/ROI accepted")
    checks.append("invalid_category_and_nonboolean_roi_rejected")
    return dict(status="PASS_SYNTHETIC_CATEGORY_GUARD", checks=checks, count=len(checks),
                model_inference=0, training=0, endpoint_sequence_inference=False)


def run():
    output = U / "DATASET_MOTION_GUARDED.json"
    audit_output = U / "MOTION_GUARD_AUDIT.json"
    require(not output.exists() and not audit_output.exists(), "Audit outputs already exist; inspect and reuse, never overwrite")
    tests = selftest()
    original = read(U / "DATASET.json")
    first_audit = read(U / "LABEL_AUDIT.json")
    first_review = read(U / "review/PRETRAIN_FINAL_REVIEW.json")
    rubric = read(U / "RUBRIC.json")
    require(first_audit["status"] == "NOT_READY_SUPPORT_OR_REVIEW", "Expected original failed readiness")
    require(first_audit["known_conflict_rate"] > .05, "Original reliability failure not present")
    for name in ("DATASET.json", "LABEL_AUDIT.json"):
        require(sha(U / name) == first_review["source_sha256"][str(U / name)], "First failed artifact changed")
    sources = [U / n for n in ("DATASET.json", "LABEL_AUDIT.json", "RUBRIC.json", "SELECTION.json",
                "review/PRETRAIN_FINAL_REVIEW.json", "review/INPUT_LABEL_VERIFICATION.json",
                "review/LABELS_A.json", "review/LABELS_B.json", "review/LABELS_C.json",
                "review/LABELS_SECONDARY.json", "review/SECONDARY_SELECTION.json")]
    sources.append(Path(__file__).resolve())
    source_hashes = {str(p): sha(p) for p in sources}
    require(read(U / "review/INPUT_LABEL_VERIFICATION.json")["status"] == "PASS_INPUT_LABEL_ARITHMETIC", "Saved original arithmetic verification absent")
    for p, expected in original["pins"].items():
        require(sha(p) == expected, f"Original source pin changed: {p}")
    protected = read(U / "SELECTION.json")["protected"]
    for p, expected in protected.items():
        require(sha(p) == expected, f"Protected artifact changed: {p}")
    primary_rows = [r for name in ("LABELS_A.json", "LABELS_B.json", "LABELS_C.json") for r in read(U / "review" / name)["rows"]]
    primary = unique_rows(primary_rows)
    secondary = unique_rows(read(U / "review/LABELS_SECONDARY.json")["rows"])
    require(set(secondary) == set(read(U / "review/SECONDARY_SELECTION.json")["sample_ids"]), "Secondary selection differs")
    input_hashes = {r["input"]: r["input_sha256"] for r in original["rows"]}
    for p, expected in input_hashes.items():
        require(sha(p) == expected, f"Input file changed: {p}")
    guarded, changes, decisions = transform(original, primary, secondary)
    targets = original["targets"]
    roles = sorted({r["role"] for r in original["rows"]})
    before_support = {role: support(original["rows"], targets, role) for role in roles}
    after_support = {role: support(guarded["rows"], targets, role) for role in roles}
    gate = rubric["class_gate"]
    fit_passed = {n: all(after_support["fit"][n][s][m] >= gate[f"fit_{s}_{m}_min"]
                        for s in ("positive", "negative") for m in ("windows", "clips", "words", "people")) for n in targets}
    old_known = sum(sum(r["target_mask"]) for r in original["rows"])
    new_known = sum(sum(r["target_mask"]) for r in guarded["rows"])
    require(old_known == first_audit["known_cells"], "Original known-cell count differs from first audit")
    require(old_known - new_known == len(changes), "Changed cell ledger/count mismatch")
    require(all(c["previous_value"] == 0 for c in changes), "Unexpected retained positive under categorical conflict")
    before_categories = Counter()
    withheld_categories = Counter()
    for d in decisions:
        if d["decision"] == "unchecked_no_new_secondary":
            continue
        pair = (d["role"], d["hand"], d["primary"], d["secondary"])
        before_categories[pair] += 1
        if d["withhold"]:
            withheld_categories[pair] += 1
    guarded.update(status=STATUS, training_allowed=False, audit_only=True,
                   derivation=dict(version=VERSION, source_dataset=str(U / "DATASET.json"),
                                   source_dataset_sha256=source_hashes[str(U / "DATASET.json")],
                                   source_hashes=source_hashes, original_failure_preserved=True,
                                   no_temporal_adjudication=True, no_center_or_endpoint_sequence_inference=True,
                                   unchecked_labels_not_newly_validated=True))
    preserved_fields = [{k: v for k, v in r.items() if k not in ("y", "target_mask", "reasons")} for r in original["rows"]]
    retained_fields = [{k: r[k] for k in p} for r, p in zip(guarded["rows"], preserved_fields)]
    require(digest(preserved_fields) == digest(retained_fields), "Nonlabel row data changed")
    for p, expected in source_hashes.items():
        require(sha(p) == expected, f"Source changed during audit: {p}")
    # Exclusive creation ensures an existing output is never replaced.
    with output.open("x", encoding="utf8") as dest:
        json.dump(guarded, dest, ensure_ascii=False, indent=2, allow_nan=False)
    report = dict(status=STATUS, version=VERSION, audit_only=True, training_allowed=False,
                  created_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                  original_failure=dict(path=str(U / "LABEL_AUDIT.json"), sha256=source_hashes[str(U / "LABEL_AUDIT.json")],
                      status=first_audit["status"], known_cross_conflicts=first_audit["known_cross_conflicts"],
                      known_cross_comparisons=first_audit["known_cross_comparisons"], known_conflict_rate=first_audit["known_conflict_rate"],
                      category_compared=first_audit["category_compared"], category_conflicts=first_audit["category_conflicts"],
                      inherited_threshold=.05, failure_not_recalculated_as_pass=True),
                  source_hashes=source_hashes, original_dataset_pins_unchanged=guarded["pins"] == original["pins"],
                  input_sha_verified=input_hashes, protected_sha_verified=protected,
                  guarded_dataset=dict(path=str(output), sha256=sha(output), rows=len(guarded["rows"])),
                  invariants=dict(all_original_rows_in_original_order=True, nonlabel_row_fields_unchanged=True,
                      nonlabel_row_fields_sha256=digest(preserved_fields), static_shape_and_relation_unchanged=True,
                      no_secondary_rows_unchanged=True, no_new_known_labels=True,
                      no_all_negative_leftovers_for_unresolved_motion=True, unknown_storage_zero_is_not_negative=True),
                  exact_changed_cells=changes, changed_cells=len(changes), changed_selected_cells=sum(c["target"] in original["selected_targets"] for c in changes),
                  known_cells_before=old_known, known_cells_after=new_known,
                  unknown_cells_after=len(targets) * len(guarded["rows"]) - new_known,
                  coverage_before={role: coverage(original["rows"], targets, decisions, role) for role in roles},
                  coverage_after={role: coverage(guarded["rows"], targets, decisions, role) for role in roles},
                  support_before=before_support, support_after=after_support,
                  selected_fit_support_passed_after={n: fit_passed[n] for n in original["selected_targets"]},
                  fit_support_passed_after=fit_passed,
                  motion_category_coverage=[dict(role=k[0], hand=k[1], primary=k[2], secondary=k[3],
                      hand_windows=count, withheld_hand_windows=withheld_categories[k]) for k, count in before_categories.items()],
                  all_hand_window_decisions=decisions, tests=tests,
                  interpretation="Conservative masking only. Agreement-selected labels are not validated sequence truth. All original challenge windows remain in coverage. Original reliability fails; support counts do not authorize training or easier-subset scoring.",
                  actual_training_runs=0, actual_model_inference_calls=0, previous_scores_rerun=0,
                  saved_label_verification_reused_without_rerun=True)
    with audit_output.open("x", encoding="utf8") as dest:
        json.dump(report, dest, ensure_ascii=False, indent=2, allow_nan=False)
    # Final read-back and original-source hashes are checks, not prior reruns.
    require(read(output)["status"] == read(audit_output)["status"] == STATUS, "Output status mismatch")
    for p, expected in source_hashes.items():
        require(sha(p) == expected, f"Original source changed: {p}")
    print(json.dumps(dict(status=STATUS, tests=tests["count"], rows=len(guarded["rows"]), changed_cells=len(changes),
                         known_cells_before=old_known, known_cells_after=new_known,
                         selected_fit_passed=report["selected_fit_support_passed_after"],
                         dataset_sha256=sha(output), audit_sha256=sha(audit_output))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(selftest(), indent=2))
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("Usage: motion_category_guard.py selftest | run (audit only; never trains)")
