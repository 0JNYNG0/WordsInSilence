#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F3 predeclared TRAIN development comparison of three methods on the 86-point pairs.

    1 DTW-only     nearest reference by the grouped path cost (class minimum)
    2 ML-only      pair compatibility on the 15 fixed-time descriptors (no alignment)
    3 Hybrid       pair compatibility on the 15 aligned descriptors + 3 path statistics

DEVELOPMENT ONLY. These people and captures were already used to develop the system, so
leave-one-signer-out here is never independent final accuracy. It measures CLOSED-SET
RANKING and technical failures. Wrong performance, idle input, out-of-vocabulary false
acceptance and any personal before/after remain UNVERIFIED and are not inferred from
correct words. No acceptance threshold, enrollment, promotion or game authority exists
here, and a rank is never an ACCEPT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_hand_pair_features_v1 as fh          # noqa: E402

OUTPUT_SCHEMA = "wis_full_hand_comparison_v1"
TRAIN_SPLIT = "train"
METHODS = ("dtw_only", "ml_only", "hybrid")
LOGISTIC_CONFIG = {"penalty": "l2", "C": 1.0, "class_weight": "balanced",
                   "solver": "lbfgs", "max_iter": 2000, "random_state": 0}
CONCLUSION = "RANKING_ONLY_NO_PROMOTION"


class ComparisonError(RuntimeError):
    """A refusal. Nothing partial is claimed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True,
                               default=str), encoding="utf-8")


# ---- learner ---------------------------------------------------------------

def fit_scaler(rows: np.ndarray):
    """Per-feature nan-mean/std from the supplied TRAIN rows only."""
    values = np.asarray(rows, dtype=np.float64)
    width = values.shape[1]
    mean = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    available = np.zeros(width, dtype=bool)
    for column in range(width):
        finite = values[:, column][np.isfinite(values[:, column])]
        if finite.size == 0:
            continue                      # empty feature: mean 0, scale 1, unavailable
        available[column] = True
        mean[column] = float(finite.mean())
        deviation = float(finite.std(ddof=0))
        scale[column] = deviation if deviation > 0.0 else 1.0
    return mean, scale, available


def standardize(rows, mean, scale) -> np.ndarray:
    """A missing descriptor becomes 0, i.e. the training mean, never a perfect match."""
    values = (np.asarray(rows, dtype=np.float64) - mean) / scale
    return np.where(np.isfinite(values), values, 0.0)


def fit_pair_model(rows: np.ndarray, labels: Sequence[int], names: Sequence[str]) -> Dict:
    from sklearn.linear_model import LogisticRegression
    import sklearn

    features = np.asarray(rows, dtype=np.float64)
    targets = np.asarray(list(labels), dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != targets.size or features.shape[0] == 0:
        raise ComparisonError("malformed training rows")
    if set(np.unique(targets)) != {0, 1}:
        raise ComparisonError("a fold produced only one pair class; a compatibility model "
                              "cannot be fit from it")
    mean, scale, available = fit_scaler(features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(**LOGISTIC_CONFIG).fit(
            standardize(features, mean, scale), targets)
    if list(model.classes_) != [0, 1]:
        raise ComparisonError("unexpected pair class order from the fit")
    return {"feature_names": list(names), "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(), "feature_available": available.tolist(),
            "coefficients": np.asarray(model.coef_, dtype=np.float64).reshape(-1).tolist(),
            "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
            "rows": int(features.shape[0]), "positive_rows": int(targets.sum()),
            "estimator": "sklearn.linear_model.LogisticRegression",
            "config": dict(LOGISTIC_CONFIG), "sklearn_version": sklearn.__version__,
            "warnings": sorted({w.category.__name__ for w in caught})}


def compatibility(parameters: Dict, features) -> float:
    """Pure NumPy reproduction of the fitted logistic. Saturating, never raising."""
    z = standardize(np.asarray(features, dtype=np.float64).reshape(1, -1),
                    np.asarray(parameters["scaler_mean"], dtype=np.float64),
                    np.asarray(parameters["scaler_scale"], dtype=np.float64))
    product = z @ np.asarray(parameters["coefficients"], dtype=np.float64)
    logit = float(np.asarray(product).reshape(-1)[0]) + float(parameters["intercept"])
    if not np.isfinite(logit):
        raise ComparisonError("a non-finite logit is a technical failure")
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exponential = float(np.exp(logit))
    return exponential / (1.0 + exponential)


# ---- inputs ----------------------------------------------------------------

def load_records(index_path: Path):
    document = json.loads(index_path.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ComparisonError("the input index carries no records")
    if str(document.get("point_schema", fh.F1_SCHEMA_ID)) != fh.F1_SCHEMA_ID:
        raise ComparisonError("the input index is not the expected F1 point schema")
    root = index_path.parent.resolve()
    npz_root = (root / "npz").resolve()
    loaded: Dict[str, Dict] = {}
    groups: Dict[str, str] = {}
    for record in records:
        for field in ("key", "label", "signer", "group_signer", "split", "length",
                      "output_file", "output_sha256", "coordinates_shape", "mask_shape"):
            if field not in record:
                raise ComparisonError("a record is missing the field " + field)
        key = str(record["key"])
        if key in loaded:
            raise ComparisonError("the input index repeats a record key")
        if str(record["split"]) != TRAIN_SPLIT:
            raise ComparisonError("a non-train record reached this comparison; no "
                                  "validation, test or historical model input is read")
        signer = str(record["signer"])
        group = str(record["group_signer"])
        if groups.setdefault(signer, group) != group:
            raise ComparisonError("one signer maps to two outer groups; a person and "
                                  "every view or take of theirs must stay in one group")
        path = (npz_root / str(record["output_file"])).resolve()
        if npz_root not in path.parents or not path.is_file():
            raise ComparisonError("an array file is missing or outside the npz directory")
        if sha256_path(path) != str(record["output_sha256"]):
            raise ComparisonError("an array file does not match its declared sha256")
        with np.load(str(path), allow_pickle=False) as saved:
            if str(saved["schema"]) != fh.F1_SCHEMA_ID:
                raise ComparisonError("an array file is not the expected F1 schema")
            coordinates = np.array(saved["coordinates"], dtype=np.float64)
            mask = np.array(saved["mask"], dtype=bool)
        if list(coordinates.shape) != [int(v) for v in record["coordinates_shape"]] or \
                list(mask.shape) != [int(v) for v in record["mask_shape"]] or \
                coordinates.shape[0] != int(record["length"]):
            raise ComparisonError("an array shape disagrees with the index")
        coordinates.flags.writeable = False
        mask.flags.writeable = False
        loaded[key] = {"key": key, "label": str(record["label"]), "signer": signer,
                       "group": group, "coordinates": coordinates, "mask": mask,
                       "source_sha256": str(record["output_sha256"])}
    if len(set(groups.values())) != len(groups):
        raise ComparisonError("two signers share one outer group; the fold unit would be "
                              "ambiguous and is refused rather than guessed")
    return loaded, {"group_of_signer": groups}


# ---- pair access, cached because it fits nothing ---------------------------

class PairCache:
    def __init__(self, sequences: Dict[str, fh.FeatureSequence]):
        self.sequences = sequences
        self.fixed: Dict[Tuple[str, str], fh.PairResult] = {}
        self.aligned: Dict[Tuple[str, str], fh.PairResult] = {}
        self.calls = {"fixed": 0, "aligned": 0}

    def fixed_pair(self, a: str, b: str) -> fh.PairResult:
        if (a, b) not in self.fixed:
            self.calls["fixed"] += 1
            self.fixed[(a, b)] = fh.fixed_time_features(self.sequences[a],
                                                        self.sequences[b])
        return self.fixed[(a, b)]

    def aligned_pair(self, a: str, b: str) -> fh.PairResult:
        if (a, b) not in self.aligned:
            self.calls["aligned"] += 1
            self.aligned[(a, b)] = fh.aligned_pair_features(self.sequences[a],
                                                            self.sequences[b])
        return self.aligned[(a, b)]


def training_pairs(keys: Sequence[str], records: Dict[str, Dict]):
    """Unordered, canonically sorted, ALWAYS cross-signer."""
    rows = []
    for first, second in combinations(sorted(keys), 2):
        if records[first]["signer"] == records[second]["signer"]:
            continue
        rows.append((first, second,
                     1 if records[first]["label"] == records[second]["label"] else 0))
    return rows


def build_training_matrix(pairs, cache: PairCache, mode: str):
    rows, labels, used, failed = [], [], [], 0
    for first, second, target in pairs:
        result = cache.fixed_pair(first, second) if mode == "fixed" \
            else cache.aligned_pair(first, second)
        if not result.ok:
            failed += 1
            continue
        rows.append(result.features)
        labels.append(target)
        used.append({"pair": [first, second], "label": target})
    if not rows:
        raise ComparisonError("no usable training pair in a fold")
    return np.asarray(rows, dtype=np.float64), labels, used, failed


def rank_classes(scores: Dict[str, float], higher_is_better: bool):
    """Deterministic: sorted label order resolves an exact tie."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1] if higher_is_better
                                                       else item[1], item[0]))
    best_label, best_value = ordered[0]
    runner = ordered[1][1] if len(ordered) > 1 else None
    if runner is None:
        margin = float("nan")
    else:
        margin = best_value - runner if higher_is_better else runner - best_value
    return best_label, float(best_value), float(margin)


def score_query(query: str, reference_keys: Sequence[str], records, classes,
                cache: PairCache, ml_parameters, hybrid_parameters):
    """All classes through all allowed references; class maximum compatibility."""
    by_class: Dict[str, List[str]] = {label: [] for label in classes}
    for key in reference_keys:
        by_class[records[key]["label"]].append(key)
    absent_classes = [label for label in classes if not by_class[label]]
    scores = {method: {} for method in METHODS}
    failures = {method: 0 for method in METHODS}
    without_evidence = {method: [] for method in METHODS}
    for label in classes:
        costs, ml_values, hybrid_values = [], [], []
        for key in sorted(by_class[label]):
            aligned = cache.aligned_pair(query, key)
            if aligned.ok:
                hybrid_values.append(compatibility(hybrid_parameters, aligned.features))
                costs.append(aligned.dtw_cost)
            else:
                failures["hybrid"] += 1
                failures["dtw_only"] += 1
            fixed = cache.fixed_pair(query, key)
            if fixed.ok:
                ml_values.append(compatibility(ml_parameters, fixed.features))
            else:
                failures["ml_only"] += 1
        for method, values, better in (("dtw_only", costs, min),
                                       ("ml_only", ml_values, max),
                                       ("hybrid", hybrid_values, max)):
            if values:
                scores[method][label] = float(better(values))
            else:
                without_evidence[method].append(label)

    outcome = {}
    for method, higher in (("dtw_only", False), ("ml_only", True), ("hybrid", True)):
        blocked = sorted(set(without_evidence[method]) | set(absent_classes))
        if blocked:
            outcome[method] = {
                "status": "TECHNICAL_FAILURE",
                "reason": "NO_REFERENCE_FOR_CLASS" if absent_classes
                          else "NO_USABLE_REFERENCE_PAIR",
                "classes_without_evidence": blocked,
                "failed_pairs": failures[method]}
            continue
        top1, value, margin = rank_classes(scores[method], higher)
        outcome[method] = {"status": "RANKED", "top1": top1, "score": value,
                           "margin": margin, "raw": scores[method],
                           "failed_pairs": failures[method]}
    return outcome


# ---- the run ---------------------------------------------------------------

def run(index_path: Path, out: Path) -> Dict:
    started = time.time()
    records, group_info = load_records(index_path)
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
    # the expected vocabulary, people and groups come from EVERY validated input record,
    # so an unusable input can never shrink a denominator or hide a whole class
    classes = sorted({records[key]["label"] for key in all_keys})
    signers = sorted({records[key]["signer"] for key in all_keys})
    # the outer unit is the GROUP, so every view and take of one person is held out together
    outer_groups = sorted({records[key]["group"] for key in all_keys})
    if len(outer_groups) < 2:
        raise ComparisonError("leave-one-signer-out needs at least two outer groups")
    cache = PairCache(sequences)

    per_query: List[Dict] = []
    fold_models: List[Dict] = []
    for group in outer_groups:
        fit_keys = [key for key in usable_keys if records[key]["group"] != group]
        query_keys = [key for key in all_keys if records[key]["group"] == group]
        held_out_signers = {records[key]["signer"] for key in query_keys}
        if not fit_keys:
            # never continue past an empty-reference fold: its queries would vanish
            raise ComparisonError("fold %r has no usable reference outside the held-out "
                                  "group; the run fails rather than losing its queries"
                                  % group)
        pairs = training_pairs(fit_keys, records)
        if not pairs:
            raise ComparisonError("fold %r produced no cross-signer training pair; no "
                                  "parameter is fabricated" % group)
        for first, second, _ in pairs:
            for end in (first, second):
                if records[end]["group"] == group or \
                        records[end]["signer"] in held_out_signers:
                    raise ComparisonError("the held-out person reached a training pair")
        fixed_rows, fixed_labels, fixed_used, fixed_failed = \
            build_training_matrix(pairs, cache, "fixed")
        aligned_rows, aligned_labels, aligned_used, aligned_failed = \
            build_training_matrix(pairs, cache, "aligned")
        ml_parameters = fit_pair_model(fixed_rows, fixed_labels, fh.FIXED_FEATURE_NAMES)
        hybrid_parameters = fit_pair_model(aligned_rows, aligned_labels,
                                           fh.ALIGNED_FEATURE_NAMES)
        fold_models.append({
            "held_out_group": group, "held_out_signers": sorted(held_out_signers),
            "reference_keys": list(fit_keys),
            "fit_signers": sorted({records[key]["signer"] for key in fit_keys}),
            "fit_groups": sorted({records[key]["group"] for key in fit_keys}),
            "held_out_in_training_rows": False,
            "query_keys": list(query_keys),
            "unusable_query_keys": [key for key in query_keys if key in unusable_by_key],
            "ml_only": ml_parameters, "hybrid": hybrid_parameters,
            "training_pair_ledger": {"ml_only": fixed_used, "hybrid": aligned_used},
            "failed_training_pairs": {"ml_only": fixed_failed, "hybrid": aligned_failed}})

        for key in query_keys:
            usable_input = key not in unusable_by_key
            if usable_input:
                outcome = score_query(key, fit_keys, records, classes, cache,
                                      ml_parameters, hybrid_parameters)
            else:
                # the input stays in the denominator with its concrete reason
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

    # the final candidate models are fitted only AFTER every outer result is recorded
    all_pairs = training_pairs(usable_keys, records)
    final_fixed, final_fixed_labels, final_fixed_used, _ = \
        build_training_matrix(all_pairs, cache, "fixed")
    final_aligned, final_aligned_labels, final_aligned_used, _ = \
        build_training_matrix(all_pairs, cache, "aligned")
    final = {"ml_only": fit_pair_model(final_fixed, final_fixed_labels,
                                       fh.FIXED_FEATURE_NAMES),
             "hybrid": fit_pair_model(final_aligned, final_aligned_labels,
                                      fh.ALIGNED_FEATURE_NAMES),
             "fitted_on": "ALL authorized TRAIN cross-signer pairs",
             "used_for_outer_predictions": False,
             "training_pair_ledger": {"ml_only": final_fixed_used,
                                      "hybrid": final_aligned_used}}

    private = out / "private"
    private.mkdir(parents=True, exist_ok=False)
    write_json(private / "per_query.json",
               {"handling": "LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED",
                "group_of_signer": group_info["group_of_signer"],
                "queries": per_query, "fold_models": fold_models,
                "final_models": final})
    reference_arrays = {}
    for key in usable_keys:
        reference_arrays["points_" + key] = sequences[key].points
        reference_arrays["mask_" + key] = sequences[key].mask
    np.savez(str(private / "final_references.npz"), **reference_arrays)
    write_json(private / "final_models.json",
               {"handling": "LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED",
                "feature_schema": {"pair_schema": fh.SCHEMA_ID,
                                   "point_count": fh.FEATURE_POINT_COUNT,
                                   "groups": [list(group) for group in fh.GROUPS],
                                   "fixed_features": list(fh.FIXED_FEATURE_NAMES),
                                   "aligned_features": list(fh.ALIGNED_FEATURE_NAMES)},
                "models": final, "reference_keys": usable_keys,
                "not_bundled_into_unity": True})

    summary = summarize(per_query, classes, signers, outer_groups, unusable,
                        usable_keys, final, cache, time.time() - started)
    write_json(out / "safe_summary.json", summary)
    return summary


def summarize(per_query, classes, signers, outer_groups, unusable, usable_keys, final,
              cache, duration) -> Dict:
    counts = {method: {"ranked": 0, "top1_correct": 0, "top1_incorrect": 0,
                       "technical_failures": 0} for method in METHODS}
    for row in per_query:
        for method in METHODS:
            result = row["methods"][method]
            if result["status"] != "RANKED":
                counts[method]["technical_failures"] += 1
                continue
            counts[method]["ranked"] += 1
            counts[method]["top1_correct" if result["correct"] else "top1_incorrect"] += 1
    for method in METHODS:
        row = counts[method]
        if row["top1_correct"] + row["top1_incorrect"] + row["technical_failures"] \
                != len(per_query):
            raise ComparisonError("method %r does not account for every input exactly "
                                  "once" % method)

    transitions = {}
    regressions = 0
    for other in ("dtw_only", "ml_only"):
        table = {"other_correct_to_hybrid_incorrect": 0,
                 "other_correct_to_hybrid_technical": 0,
                 "other_incorrect_to_hybrid_correct": 0,
                 "other_technical_to_hybrid_correct": 0,
                 "both_correct": 0, "both_incorrect_or_technical": 0}
        for row in per_query:
            single = row["methods"][other]
            hybrid = row["methods"]["hybrid"]
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
                   "pair_computations": dict(cache.calls)},
        "per_method": counts,
        "transitions_vs_hybrid": transitions,
        "final_models": {
            "separate_from_outer_predictions": True,
            "used_for_outer_predictions": False,
            "estimator": final["ml_only"]["estimator"],
            "config": final["ml_only"]["config"],
            "sklearn_version": final["ml_only"]["sklearn_version"],
            "fit_warnings": {"ml_only": final["ml_only"]["warnings"],
                             "hybrid": final["hybrid"]["warnings"]},
            "training_rows": {"ml_only": final["ml_only"]["rows"],
                              "hybrid": final["hybrid"]["rows"]}},
        "duration_seconds": round(float(duration), 3),
        "limits": [
            "development comparison on data already used for development; NOT independent "
            "final accuracy",
            "closed-set ranking and technical failures only; no acceptance threshold, no "
            "ACCEPT, no promotion, no enrollment and no game authority",
            "wrong performance, idle input, out-of-vocabulary false acceptance and any "
            "personal before/after remain UNVERIFIED and were not inferred from correct "
            "words",
            "this measures a NEW 46-point representation, not a fusion-only effect against "
            "the historical 16-point models",
            "2D only; no 3D bend angle, no face or other non-manual channel",
            "the per-hand size scalar is per-capture geometry, not a learned personal "
            "profile"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F3 TRAIN development comparison")
    parser.add_argument("--input-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    index_path = Path(args.input_index).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)        # a pre-existing --out is a failure
    run_path = out / "run.json"
    pins = {"schema": OUTPUT_SCHEMA, "conclusion": CONCLUSION,
            "config": dict(LOGISTIC_CONFIG),
            "pair_schema": fh.SCHEMA_ID,
            "code_sha256": {name: sha256_path(Path(module.__file__)) for name, module
                            in (("full_hand_pair_features_v1", fh),
                                ("full_hand_comparison_v1", sys.modules[__name__]))},
            "input_index_sha256": sha256_path(index_path) if index_path.is_file() else None}
    write_json(run_path, dict(pins, status="STARTED"))
    try:
        summary = run(index_path, out)
    except BaseException:
        write_json(run_path, dict(pins, status="FAILED",
                                  traceback=traceback.format_exc()))
        print("status=FAILED")
        return 1
    write_json(run_path, dict(pins, status="COMPLETED", counts=summary["counts"]))
    # safe counts only: no key, signer, group, coefficient or array on stdout
    print("status=COMPLETED conclusion=%s nonregression=%s queries=%d classes=%d "
          "signers=%d" % (summary["conclusion"], summary["ranking_nonregression"],
                          summary["counts"]["queries"], summary["counts"]["classes"],
                          summary["counts"]["signers"]))
    for method in METHODS:
        row = summary["per_method"][method]
        print("  %-9s correct=%d incorrect=%d technical=%d"
              % (method, row["top1_correct"], row["top1_incorrect"],
                 row["technical_failures"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
