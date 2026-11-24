#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F10 finger-preserving candidate: the F9 45/48 features in the F3 TRAIN comparison.

ONE representation candidate. This is KNOWN TRAIN DEVELOPMENT material, not independent
final performance and not a general correctness model. The people and captures here were
already used to develop the system, so the leave-one-signer-out numbers measure CLOSED-SET
RANKING and technical failures only. No acceptance threshold, no ACCEPT, no enrollment, no
promotion and no game authority is produced here.

    1 dtw_only   nearest reference by the UNCHANGED grouped geometric path cost
    2 ml_only    pair compatibility on the F9 45 fixed-time features (no alignment)
    3 hybrid     pair compatibility on the F9 48 aligned features

Everything except the feature vector is the approved F3 machinery, imported and CALLED,
never monkeypatched: ``load_records`` (schema, sha256, bounded npz paths, TRAIN-only
labels), ``training_pairs`` (unordered, canonically sorted, always cross-signer),
``build_training_matrix``, ``fit_pair_model`` (logistic L2 C=1 balanced lbfgs
max_iter=2000 random_state=0 with the TRAIN-only nan-mean/std scaler, a missing
standardized feature becoming the training mean while its explicit coverage feature
stays), ``compatibility``, ``rank_classes`` and ``score_query``. The only new object is
``FingerPairCache``, a duck-typed adapter with the same surface as the F3 ``PairCache``
that returns F9 pair results and caches by ORDERED key, since DTW symmetry is not assumed.

TRAIN ONLY. This CLI reads no validation, test or live material. The audit found 22
TRAIN word-labelled sequences from 11 people plus one-user requested-condition pilot
captures, and NO expert-certified multi-person quality labels. So this candidate tests
only whether preserving finger identity changes the existing technical failures and
ranking mistakes. It does not learn or claim verified KSL execution quality, and word
labels are never turned into quality labels. Game integration needs a later
evidence-based decision, not a low threshold or a hardcoded greeting/thanks rule.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_PACKET = _HERE.parent.parent
for _directory in (_PACKET / "input", _PACKET / "output"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import full_hand_pair_features_v1 as fh              # noqa: E402  approved F3R1 baseline
import full_hand_comparison_v1 as cmp                # noqa: E402  approved F3 comparison
import finger_pair_features_v2 as f9                 # noqa: E402  approved F9 features

OUTPUT_SCHEMA = "wis_finger_candidate_experiment_v2"
CONCLUSION = "TRAIN_DEVELOPMENT_DIAGNOSTIC_ONLY_NO_PROMOTION"
METHODS = cmp.METHODS
ComparisonError = cmp.ComparisonError
QUERY_KEY = "__scored_query__"

THRESHOLD_POLICY = {
    "applied_in_this_run": False,
    "reason": "this fit CLI produces ranking diagnostics only; no ACCEPT decision exists "
              "here and runtime evaluation is a separate operator action",
    "declared_policy": "one globally applied threshold, chosen on the development set "
                       "only, maximizing correct positive accepts subject to ZERO "
                       "development false accepts",
    "no_per_word_adjustment": True,
    "not_tuned_on_final_material": True,
}


# ---- the only new machinery: an adapter, not a patch ------------------------

class FingerPairCache:
    """The F3 ``PairCache`` surface, filled by the F9 45/48 feature functions.

    Pure: the pair functions fit nothing and are given SEQUENCES ONLY, never an intended
    target, a word or a user id. Keys are ORDERED, so no DTW symmetry is assumed."""

    def __init__(self, sequences: Dict[str, fh.FeatureSequence]):
        self.sequences = sequences
        self.fixed: Dict = {}
        self.aligned: Dict = {}
        self.calls = {"fixed": 0, "aligned": 0}

    def fixed_pair(self, a: str, b: str):
        if (a, b) not in self.fixed:
            self.calls["fixed"] += 1
            self.fixed[(a, b)] = f9.fixed_time_features(self.sequences[a],
                                                        self.sequences[b])
        return self.fixed[(a, b)]

    def aligned_pair(self, a: str, b: str):
        if (a, b) not in self.aligned:
            self.calls["aligned"] += 1
            self.aligned[(a, b)] = f9.aligned_pair_features(self.sequences[a],
                                                            self.sequences[b])
        return self.aligned[(a, b)]


# ---- the one public scoring entry point -------------------------------------

def _check_parameters(parameters: Dict, names, width: int, label: str) -> None:
    if not isinstance(parameters, dict):
        raise ComparisonError("model %r is not a parameter mapping" % label)
    for field in ("scaler_mean", "scaler_scale", "coefficients", "intercept"):
        if field not in parameters:
            raise ComparisonError("model %r is missing %s" % (label, field))
    if len(parameters["coefficients"]) != width or \
            len(parameters["scaler_mean"]) != width or \
            len(parameters["scaler_scale"]) != width:
        raise ComparisonError("model %r is not the %d-wide F10 candidate" % (label, width))
    stored = parameters.get("feature_names")
    if stored is not None and list(stored) != list(names):
        raise ComparisonError("model %r was fitted on different feature names" % label)


def score_sequence(seq, references: Dict[str, Dict], models: Dict) -> Dict:
    """Score ONE sequence against every class. There is no target argument, by contract.

    references: key -> {'label': str, 'seq': FeatureSequence}
    models:     {'ml_only': parameters(45), 'hybrid': parameters(48)}

    Returns the three method results, each either RANKED with raw per-class scores, top1
    and margin, or a retained TECHNICAL_FAILURE. dtw_only needs no model: it is the
    unchanged geometric path cost, not a newly fitted method. Inputs are not mutated and
    nothing here fits, enrolls, accepts or promotes anything."""
    if not isinstance(references, dict) or not references:
        raise ComparisonError("no reference was supplied")
    if not isinstance(models, dict):
        raise ComparisonError("models must be a mapping with 'ml_only' and 'hybrid'")
    _check_parameters(models.get("ml_only"), f9.FIXED_FEATURE_NAMES,
                      f9.FIXED_FEATURE_COUNT, "ml_only")
    _check_parameters(models.get("hybrid"), f9.ALIGNED_FEATURE_NAMES,
                      f9.ALIGNED_FEATURE_COUNT, "hybrid")
    f9.validate_sequence(seq, "query")

    sequences = {QUERY_KEY: seq}
    records: Dict[str, Dict] = {}
    for key in sorted(references):
        entry = references[key]
        if str(key) == QUERY_KEY:
            raise ComparisonError("a reference uses the reserved query key")
        if not isinstance(entry, dict) or "label" not in entry or "seq" not in entry:
            raise ComparisonError("reference %r needs a 'label' and a 'seq'" % key)
        f9.validate_sequence(entry["seq"], "reference %s" % key)
        sequences[str(key)] = entry["seq"]
        records[str(key)] = {"label": str(entry["label"])}

    classes = sorted({record["label"] for record in records.values()})
    cache = FingerPairCache(sequences)
    outcome = cmp.score_query(QUERY_KEY, sorted(records), records, classes, cache,
                              models["ml_only"], models["hybrid"])
    return {"schema": OUTPUT_SCHEMA, "feature_schema": f9.SCHEMA_ID,
            "classes": classes, "reference_count": len(records),
            "methods": outcome, "is_accept_decision": False,
            "pair_computations": dict(cache.calls)}


# ---- the TRAIN run ----------------------------------------------------------

def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(index_path: Path, out: Path, fit_function=None) -> Dict:
    """Leave-one-group-out over TRAIN, then the two final TRAIN models. Expected fits for
    the current 11-person dataset: 11 * 2 + 2 = 24, and the actual count is recorded."""
    if (out / "private").exists() or (out / "safe_summary.json").exists():
        # refused before any array is read and before anything is fitted
        raise ComparisonError("the output directory already holds a result; an earlier "
                              "run is never replaced")
    fit_model = fit_function or cmp.fit_pair_model
    fits = {"count": 0}

    def fit(rows, labels, names):
        fits["count"] += 1
        return fit_model(rows, labels, names)

    started = time.time()
    started_utc = _utc()
    # the approved reader: F1 point schema, per-file sha256, npz paths bounded to npz/,
    # TRAIN split only, one signer to one outer group
    records, group_info = cmp.load_records(index_path)

    sequences: Dict[str, fh.FeatureSequence] = {}
    unusable: List[Dict] = []
    for key, record in sorted(records.items()):
        sequence = fh.build_feature_sequence(record["coordinates"], record["mask"])
        if not sequence.usable:
            unusable.append({"key": key, "failure": sequence.failure})
            continue
        sequences[key] = sequence
    all_keys = sorted(records)
    usable_keys = sorted(sequences)
    unusable_by_key = {row["key"]: row["failure"] for row in unusable}
    if len(usable_keys) < 2:
        raise ComparisonError("fewer than two usable sequences")
    # every denominator comes from EVERY validated record, so an unusable input can never
    # shrink a class or a fold
    classes = sorted({records[key]["label"] for key in all_keys})
    signers = sorted({records[key]["signer"] for key in all_keys})
    outer_groups = sorted({records[key]["group"] for key in all_keys})
    if len(outer_groups) < 2:
        raise ComparisonError("leave-one-signer-out needs at least two outer groups")
    cache = FingerPairCache(sequences)

    per_query: List[Dict] = []
    fold_models: List[Dict] = []
    for group in outer_groups:
        fit_keys = [key for key in usable_keys if records[key]["group"] != group]
        query_keys = [key for key in all_keys if records[key]["group"] == group]
        held_out_signers = {records[key]["signer"] for key in query_keys}
        if not fit_keys:
            raise ComparisonError("fold %r has no usable reference outside the held-out "
                                  "group; the run fails rather than losing its queries"
                                  % group)
        # the held-out person is excluded from the REFERENCES as well as the pairs
        for key in fit_keys:
            if records[key]["group"] == group or \
                    records[key]["signer"] in held_out_signers:
                raise ComparisonError("the held-out person reached the reference set")
        pairs = cmp.training_pairs(fit_keys, records)
        if not pairs:
            raise ComparisonError("fold %r produced no cross-signer training pair; no "
                                  "parameter is fabricated" % group)
        for first, second, _ in pairs:
            for end in (first, second):
                if records[end]["group"] == group or \
                        records[end]["signer"] in held_out_signers:
                    raise ComparisonError("the held-out person reached a training pair")
        fixed_rows, fixed_labels, fixed_used, fixed_failed = \
            cmp.build_training_matrix(pairs, cache, "fixed")
        aligned_rows, aligned_labels, aligned_used, aligned_failed = \
            cmp.build_training_matrix(pairs, cache, "aligned")
        if fixed_rows.shape[1] != f9.FIXED_FEATURE_COUNT or \
                aligned_rows.shape[1] != f9.ALIGNED_FEATURE_COUNT:
            raise ComparisonError("a fold matrix is not the F10 45/48 candidate width")
        ml_parameters = fit(fixed_rows, fixed_labels, f9.FIXED_FEATURE_NAMES)
        hybrid_parameters = fit(aligned_rows, aligned_labels, f9.ALIGNED_FEATURE_NAMES)
        fold_models.append({
            "held_out_group": group, "held_out_signers": sorted(held_out_signers),
            "reference_keys": list(fit_keys),
            "fit_signers": sorted({records[key]["signer"] for key in fit_keys}),
            "fit_groups": sorted({records[key]["group"] for key in fit_keys}),
            "held_out_in_training_rows": False,
            "held_out_in_reference_keys": False,
            "query_keys": list(query_keys),
            "unusable_query_keys": [key for key in query_keys if key in unusable_by_key],
            "ml_only": ml_parameters, "hybrid": hybrid_parameters,
            "training_pair_ledger": {"ml_only": fixed_used, "hybrid": aligned_used},
            "failed_training_pairs": {"ml_only": fixed_failed,
                                      "hybrid": aligned_failed}})

        for key in query_keys:
            usable_input = key not in unusable_by_key
            if usable_input:
                outcome = cmp.score_query(key, fit_keys, records, classes, cache,
                                          ml_parameters, hybrid_parameters)
            else:
                # the input keeps its place in the denominator with its concrete reason
                outcome = {method: {"status": "TECHNICAL_FAILURE",
                                    "reason": unusable_by_key[key],
                                    "failed_pairs": 0} for method in METHODS}
            row = {"fold_group": group, "fold_signer": records[key]["signer"],
                   "key": key, "label": records[key]["label"],
                   "group": records[key]["group"], "usable_input": usable_input,
                   "allowed_reference_count": len(fit_keys),
                   "allowed_reference_keys": list(fit_keys), "methods": {}}
            for method in METHODS:
                result = dict(outcome[method])
                result["correct"] = (result.get("top1") == records[key]["label"]) \
                    if result["status"] == "RANKED" else None
                row["methods"][method] = result
            per_query.append(row)

    # the two final TRAIN models are fitted only AFTER every outer result is recorded
    all_pairs = cmp.training_pairs(usable_keys, records)
    final_fixed, final_fixed_labels, final_fixed_used, final_fixed_failed = \
        cmp.build_training_matrix(all_pairs, cache, "fixed")
    final_aligned, final_aligned_labels, final_aligned_used, final_aligned_failed = \
        cmp.build_training_matrix(all_pairs, cache, "aligned")
    final = {"ml_only": fit(final_fixed, final_fixed_labels, f9.FIXED_FEATURE_NAMES),
             "hybrid": fit(final_aligned, final_aligned_labels,
                           f9.ALIGNED_FEATURE_NAMES),
             "fitted_on": "ALL authorized TRAIN cross-signer pairs",
             "used_for_outer_predictions": False,
             "training_pair_ledger": {"ml_only": final_fixed_used,
                                      "hybrid": final_aligned_used},
             "failed_training_pairs": {"ml_only": final_fixed_failed,
                                       "hybrid": final_aligned_failed}}

    expected_fits = 2 * len(outer_groups) + 2
    if fits["count"] != expected_fits:
        raise ComparisonError("expected %d fits, performed %d"
                              % (expected_fits, fits["count"]))

    private = out / "private"
    private.mkdir(parents=True, exist_ok=False)
    cmp.write_json(private / "per_query.json",
                   {"handling": "LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED",
                    "schema": OUTPUT_SCHEMA,
                    "group_of_signer": group_info["group_of_signer"],
                    "queries": per_query, "fold_models": fold_models,
                    "final_models": final})
    reference_arrays = {"schema": np.array(f9.SCHEMA_ID),
                        "reference_keys": np.array(usable_keys)}
    for key in usable_keys:
        reference_arrays["points_" + key] = sequences[key].points
        reference_arrays["mask_" + key] = sequences[key].mask
    # arrays only; nothing is pickled, so the file loads with allow_pickle=False
    np.savez(str(private / "final_references.npz"), **reference_arrays)
    cmp.write_json(private / "final_models.json",
                   {"handling": "LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED",
                    "feature_schema": {"pair_schema": f9.SCHEMA_ID,
                                       "sequence_schema": fh.SCHEMA_ID,
                                       "point_count": fh.FEATURE_POINT_COUNT,
                                       "groups": [list(g) for g in fh.GROUPS],
                                       "finger_ranges": [list(r) for r
                                                         in f9.FINGER_POINT_RANGES],
                                       "fixed_features": list(f9.FIXED_FEATURE_NAMES),
                                       "aligned_features": list(f9.ALIGNED_FEATURE_NAMES)},
                    "models": final, "reference_keys": usable_keys,
                    "threshold_policy": dict(THRESHOLD_POLICY),
                    "not_bundled_into_unity": True,
                    "not_promoted_to_the_game": True})

    summary = summarize(per_query, classes, signers, outer_groups, unusable, usable_keys,
                        final, cache, fits["count"], expected_fits, started_utc,
                        time.time() - started)
    cmp.write_json(out / "safe_summary.json", summary)
    return summary


def summarize(per_query, classes, signers, outer_groups, unusable, usable_keys, final,
              cache, actual_fits, expected_fits, started_utc, duration) -> Dict:
    """Safe aggregate counts: no key, signer, group, coefficient or array."""
    counts = {method: {"ranked": 0, "top1_correct": 0, "top1_incorrect": 0,
                       "technical_failures": 0} for method in METHODS}
    for row in per_query:
        for method in METHODS:
            result = row["methods"][method]
            if result["status"] != "RANKED":
                counts[method]["technical_failures"] += 1
                continue
            counts[method]["ranked"] += 1
            counts[method]["top1_correct" if result["correct"]
                           else "top1_incorrect"] += 1
    for method in METHODS:
        row = counts[method]
        if row["top1_correct"] + row["top1_incorrect"] + row["technical_failures"] \
                != len(per_query):
            raise ComparisonError("method %r does not account for every input exactly "
                                  "once" % method)

    transitions, regressions = {}, 0
    for other in ("dtw_only", "ml_only"):
        table = {"other_correct_to_hybrid_incorrect": 0,
                 "other_correct_to_hybrid_technical": 0,
                 "other_incorrect_to_hybrid_correct": 0,
                 "other_technical_to_hybrid_correct": 0,
                 "both_correct": 0, "both_incorrect_or_technical": 0}
        for row in per_query:
            single, hybrid = row["methods"][other], row["methods"]["hybrid"]
            single_ok = single["status"] == "RANKED" and single["correct"]
            hybrid_ok = hybrid["status"] == "RANKED" and hybrid["correct"]
            if single_ok and hybrid_ok:
                table["both_correct"] += 1
            elif single_ok and hybrid["status"] != "RANKED":
                table["other_correct_to_hybrid_technical"] += 1
            elif single_ok:
                table["other_correct_to_hybrid_incorrect"] += 1
            elif hybrid_ok and single["status"] != "RANKED":
                table["other_technical_to_hybrid_correct"] += 1
            elif hybrid_ok:
                table["other_incorrect_to_hybrid_correct"] += 1
            else:
                table["both_incorrect_or_technical"] += 1
        regressions += table["other_correct_to_hybrid_incorrect"] + \
            table["other_correct_to_hybrid_technical"]
        transitions["hybrid_vs_" + other] = table

    return {
        "schema": OUTPUT_SCHEMA,
        "conclusion": CONCLUSION,
        "material": "KNOWN TRAIN DEVELOPMENT MATERIAL; NOT INDEPENDENT FINAL PERFORMANCE",
        "candidate": "F9 finger-preserving pair features (45 fixed / 48 aligned)",
        "started_utc": started_utc,
        "finished_utc": _utc(),
        "ranking_nonregression": "PASS" if regressions == 0 else "FAIL",
        "ranking_nonregression_note":
            "FAIL means the hybrid lost at least one case that a single method ranked "
            "correctly. A loss is never offset by an average score.",
        "counts": {"input_records": len(per_query),
                   "usable_sequences": len(usable_keys), "classes": len(classes),
                   "signers": len(signers), "outer_groups": len(outer_groups),
                   "folds": len(outer_groups), "queries": len(per_query),
                   "unusable_sequences": len(unusable),
                   "unusable_reasons": sorted({row["failure"] for row in unusable}),
                   "class_query_counts": {label: sum(1 for row in per_query
                                                     if row["label"] == label)
                                          for label in classes},
                   "pair_computations": dict(cache.calls),
                   "fits_expected": expected_fits, "fits_actual": actual_fits},
        "per_method": counts,
        "transitions_vs_hybrid": transitions,
        "feature_schema": {"pair_schema": f9.SCHEMA_ID,
                           "fixed_width": f9.FIXED_FEATURE_COUNT,
                           "aligned_width": f9.ALIGNED_FEATURE_COUNT,
                           "baseline_prefix": [f9.BASELINE_FIXED_PREFIX,
                                               f9.BASELINE_ALIGNED_PREFIX]},
        "final_models": {
            "separate_from_outer_predictions": True,
            "used_for_outer_predictions": False,
            "estimator": final["ml_only"].get("estimator"),
            "config": final["ml_only"].get("config"),
            "sklearn_version": final["ml_only"].get("sklearn_version"),
            "fit_warnings": {"ml_only": final["ml_only"].get("warnings"),
                             "hybrid": final["hybrid"].get("warnings")},
            "training_rows": {"ml_only": final["ml_only"].get("rows"),
                              "hybrid": final["hybrid"].get("rows")}},
        "threshold_policy": dict(THRESHOLD_POLICY),
        "duration_seconds": round(float(duration), 3),
        "limits": [
            "TRAIN development on material already used for development; NOT independent "
            "final accuracy and NOT a promotion",
            "closed-set ranking and technical failures only; no threshold was applied, no "
            "ACCEPT, no enrollment, no Unity build and no game authority",
            "dtw_only is the unchanged geometric path cost, not a newly fitted method",
            "the 22 TRAIN sequences from 11 people and the one-user pilot captures are "
            "KNOWN regression material, never a fresh final set for this candidate",
            "no expert-certified multi-person quality label exists, so verified KSL "
            "execution quality is NOT learned or claimed and word labels were never "
            "turned into quality labels",
            "wrong performance, idle input, out-of-vocabulary false acceptance and any "
            "personal before/after remain UNVERIFIED",
            "2D only; no 3D bend angle, no face or other non-manual channel",
            "the per-hand size scalar is per-capture geometry, not a learned personal "
            "profile"],
    }


# ---- CLI --------------------------------------------------------------------

def _pins(index_path: Path) -> Dict:
    modules = (("full_hand_pair_features_v1", fh), ("full_hand_comparison_v1", cmp),
               ("finger_pair_features_v2", f9),
               ("finger_candidate_experiment_v2", sys.modules[__name__]))
    return {"schema": OUTPUT_SCHEMA, "conclusion": CONCLUSION,
            "config": dict(cmp.LOGISTIC_CONFIG),
            "pair_schema": f9.SCHEMA_ID, "split": cmp.TRAIN_SPLIT,
            "started_utc": _utc(),
            "code_sha256": {name: cmp.sha256_path(Path(module.__file__))
                            for name, module in modules},
            "train_index_sha256": cmp.sha256_path(index_path)
                                  if index_path.is_file() else None}


def main(argv=None, fit_function=None) -> int:
    parser = argparse.ArgumentParser(
        description="F10 finger candidate, TRAIN development only")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    index_path = Path(args.train_index).resolve()
    out = Path(args.out)
    try:
        # refused BEFORE any array is read and before anything is fitted, so an earlier
        # result can never be replaced
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("status=FAILED reason=OUTPUT_DIRECTORY_EXISTS")
        return 1
    pins = _pins(index_path)
    run_path = out / "run.json"
    cmp.write_json(run_path, dict(pins, status="STARTED"))
    try:
        summary = run(index_path, out, fit_function=fit_function)
    except BaseException:
        failure = out / "first_failure.json"
        if not failure.exists():          # the FIRST failure is kept, never replaced
            cmp.write_json(failure, dict(pins, status="FAILED", finished_utc=_utc(),
                                         traceback=traceback.format_exc()))
        cmp.write_json(run_path, dict(pins, status="FAILED", finished_utc=_utc(),
                                      first_failure_file="first_failure.json"))
        print("status=FAILED")
        return 1
    cmp.write_json(run_path, dict(pins, status="COMPLETED", finished_utc=_utc(),
                                  counts=summary["counts"]))
    # safe counts only: no key, signer, group, coefficient or array on stdout
    print("status=COMPLETED conclusion=%s nonregression=%s queries=%d classes=%d "
          "signers=%d fits=%d" % (summary["conclusion"],
                                  summary["ranking_nonregression"],
                                  summary["counts"]["queries"],
                                  summary["counts"]["classes"],
                                  summary["counts"]["signers"],
                                  summary["counts"]["fits_actual"]))
    for method in METHODS:
        row = summary["per_method"][method]
        print("  %-9s correct=%d incorrect=%d technical=%d"
              % (method, row["top1_correct"], row["top1_incorrect"],
                 row["technical_failures"]))
    print("  TRAIN development diagnostic only; not final performance, not a promotion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
