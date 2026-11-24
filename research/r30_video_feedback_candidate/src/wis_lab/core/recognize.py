"""predict(observation, reference_set, frozen_config).

Never sees the requested or the ground-truth word. Task matching happens elsewhere, against
this output. R1 produces a ranking and diagnostics; execution_assessment stays UNCONFIGURED
because no pass line has been calibrated on real correct/incorrect recordings yet.
"""
from __future__ import annotations

import numpy as np

from . import features as F
from .dtw import dtw, group_residuals as dtw_group_residuals
from .joints import joint_deviation
from .trajectory import location_path, source_observation

SCHEMA_VERSION = "wis_result_v1"


def build_reference(obs, clip_id: str, word_id: str, budget: int) -> dict:
    seq = F.sequence_features(obs)
    sample = F.uniform_time_sample(seq, budget)
    sample["clip_id"] = clip_id
    sample["word_id"] = word_id
    sample["span"] = seq["span"]
    sample["shoulder_width_px"] = seq["shoulder_width_px"]
    # the ORIGINAL per-row hand presence, kept so a dropout between two sampled rows can still be
    # seen. An added key only: features, present and timestamp_us are exactly as before, and no
    # recogniser reads it.
    sample["source_observation"] = source_observation(seq)
    return sample


def predict(observation, reference_set: list[dict], frozen_config: dict) -> dict:
    """Rank every prepared reference. reference_set order does not affect the outcome."""
    budget = int(frozen_config["sampling_A"]["budget"])
    band = float(frozen_config["dtw"]["band"].split("ratio")[-1].strip(" ,:")) \
        if isinstance(frozen_config["dtw"].get("band"), str) else 0.2
    min_frames = int(frozen_config["dtw"].get("min_sampled_frames", 8))

    out = {
        "schema_version": SCHEMA_VERSION,
        "observation_status": "OK",
        "recognition": "UNCONFIGURED",
        "execution_assessment": "UNCONFIGURED",
        "accepted": None,
        "ranking": [],
        "quality_summary": None,
        "insufficient_reason": None,
    }

    try:
        seq = F.sequence_features(observation)
        query = F.uniform_time_sample(seq, budget)
    except F.FeatureError as exc:
        out["observation_status"] = "INSUFFICIENT"
        out["insufficient_reason"] = str(exc)
        out["quality_summary"] = observation.quality_summary()
        return out

    q = observation.quality_summary()
    q["sampled_frames"] = int(len(query["features"]))
    q["active_span"] = list(seq["span"])
    q["shoulder_width_px"] = round(seq["shoulder_width_px"], 2)
    out["quality_summary"] = q

    if len(query["features"]) < min_frames:
        out["observation_status"] = "INSUFFICIENT"
        out["insufficient_reason"] = (
            f"only {len(query['features'])} usable sampled frames, minimum is {min_frames}")
        return out

    rows = []
    for ref in reference_set:
        cost_cfg = frozen_config.get("cost", {})
        r = dtw(query, ref, band_ratio=band,
                one_sided_penalty=cost_cfg.get("one_sided_penalty"),
                one_sided_mode=cost_cfg.get("one_sided_mode", "flat"),
                group_weights=cost_cfg.get("group_weights"),
                hand_participation=cost_cfg.get("hand_participation", False))
        rows.append({
            "clip_id": ref["clip_id"],
            "word_id": ref["word_id"],
            "status": r["status"],
            "distance": (round(r["distance"], 5) if r["distance"] is not None else None),
            "path_length": r["path_length"],
            "mean_evidence_fraction": r.get("mean_evidence_fraction"),
            "reason": r.get("reason"),
        })

    scored = [r for r in rows if r["distance"] is not None]
    scored.sort(key=lambda r: r["distance"])
    unscored = [r for r in rows if r["distance"] is None]
    for i, r in enumerate(scored):
        r["rank"] = i + 1
    out["ranking"] = scored + unscored

    if not scored:
        out["observation_status"] = "INSUFFICIENT"
        out["insufficient_reason"] = "no reference produced an admissible warping path"
        return out

    best = scored[0]
    out["top1_word_id"] = best["word_id"]
    out["top3_word_ids"] = [r["word_id"] for r in scored[:3]]
    out["top1_distance"] = best["distance"]
    out["margin_to_second"] = (round(scored[1]["distance"] - best["distance"], 5)
                               if len(scored) > 1 else None)
    out["recognition"] = "UNCONFIGURED"
    out["recognition_note"] = (
        "ranking only. KNOWN / UNKNOWN / AMBIGUOUS needs a distance threshold calibrated on real "
        "correct and incorrect attempts, which does not exist yet.")
    out["axis_checks"] = axis_checks(query, best["word_id"], reference_set, frozen_config)
    out["axis_deviation"] = axis_deviation(query, best["word_id"], reference_set, frozen_config)
    out["handshape_check"] = out["axis_checks"].get("handshape") or handshape_check(
        query, best["word_id"], reference_set, frozen_config)
    out["execution_assessment_note"] = (
        "UNCONFIGURED: the ranking above has no pass line and this distance is not a "
        "sign-language accuracy score. handshape_check is a separate, calibrated verdict on ONE "
        "axis of the performance, not on the performance as a whole.")
    return out


# Which feature groups belong to which axis. Handshape uses the flexion angles because the
# coordinate groups correlate +0.958 with palm orientation; location uses body- and face-relative
# hand position, which is what a "right shape, wrong place" error actually violates.
AXIS_GROUPS = {
    "handshape": ("bend_L", "bend_R"),
    "location": ("handpos_L", "handpos_R", "facerel_L", "facerel_R"),
    "movement": ("arm_L", "arm_R", "motion_L", "motion_R"),
}

FEEDBACK_ROLE_POLICY = "exclude_both_resting_hands_v1"


def _role_feedback_enabled(config: dict) -> bool:
    return (config.get("feedback") or {}).get("exclude_both_resting_hands") is True


def _feedback_support(groups, names, query, reference, config):
    """Eligibility is separate from observation and from the lexical DTW cost weights.

    Only paired-observation resting evidence on BOTH sides excludes a hand. Legacy
    participation=False also means insufficient observation and is NOT evidence of rest.
    Absent/unknown evidence and a hand active in either sequence keep it eligible.
    """
    if not _role_feedback_enabled(config):
        return _axis_support(groups, names)
    qa = query.get("feedback_hand_roles")
    ra = reference.get("feedback_hand_roles")
    resting = [side for side in ("L", "R")
               if F.observed_resting_hand(qa, side) and F.observed_resting_hand(ra, side)]
    excluded = [g for g in names if g.rsplit("_", 1)[-1] in resting]
    eligible = [g for g in names if g not in excluded]
    support = _axis_support(groups, eligible)
    raw = _axis_support(groups, names)
    for g in names:
        if g in excluded:
            detail = dict(raw["per_group"][g])
            detail["aggregate_weight"] = 0.0
            support["per_group"][g] = detail
        detail = support["per_group"][g]
        detail["eligible"] = g not in excluded
        detail["included_in_aggregate"] = g in support["measured_groups"]
        detail["exclusion_reason"] = "both_sequences_resting" if g in excluded else None
    support.update({"feedback_policy": FEEDBACK_ROLE_POLICY,
                    "all_groups": list(names), "eligible_groups": eligible,
                    "excluded_groups": excluded,
                    "both_resting_sides": resting,
                    "role_evidence": {"query": qa, "reference": ra},
                    "role_caveat": "Observed vertical-wrist heuristic, not validated semantic hand relevance."})
    support["aggregation"]["rule"] = "equal mean of ELIGIBLE measured groups' mean distances"
    support["aggregation"]["excluded_groups_contribute"] = (
        "nothing: raw measurements are retained but their aggregate weight is zero; "
        "role exclusion is not missing observation or a zero distance")
    support["aggregation"]["validated"] = (
        "no. Role-filtered diagnostic candidate; execution quality has not been validated.")
    return support


def _axis_support(groups: dict, names) -> dict:
    """How much of an axis was actually measured on both sides, in groups AND in time.

    Two different kinds of incompleteness, reported separately because they mean different things:

      group coverage  did every group the axis needs produce a distance at all
      time coverage   of the steps on the warping path, how many were actually compared

    Counting a group as measured because it produced a distance ONCE was not enough. Independent
    review fed both hands at compared_steps 1 of 8 and got fraction 1.0 with partial False: most
    of the attempt was unobserved and the axis still read as fully confirmed. Any incompleteness
    on either axis now marks the result partial, with the numbers attached.

    What each measured group contributed is reported too. The axis distance is the EQUAL mean of
    the measured groups' own means, so a group measured once and a group measured twenty-seven
    times carry the same 1/n weight. This is the aggregation the project already uses, and
    whether it is suitable for judging an execution has NOT been validated - no statement is made
    here either way. What is certain is that a single number hides the difference: a right hand
    seen 27 times at 0.15999 and a left hand seen once at 0.36627 average to 0.26313, and the
    reader cannot tell that the two halves rest on very different amounts of evidence. The
    per-group block now says it outright.

    No pass line is created here. There is no statistic yet that says how much comparison is
    enough, so partial is reported as partial rather than being scored, and no coverage threshold
    has been validated.
    """
    measured = [g for g in names if groups.get(g) and groups[g]["mean_distance"] is not None]
    missing = [g for g in names if g not in measured]
    detail, times = {}, {}
    for g in names:
        d = groups.get(g) or {}
        steps = sum(int(d.get(k, 0)) for k in
                    ("compared_steps", "one_sided_steps", "unmeasurable_steps",
                     "unobserved_steps"))
        compared = int(d.get("compared_steps", 0))
        frac = round(compared / steps, 4) if steps else 0.0
        if g in measured:
            times[g] = frac
        # A group that was not measured has NO distance. It is not a zero, which would read as a
        # perfect match, and it carries no weight in the mean rather than a small one.
        detail[g] = {"measured": g in measured, "compared_steps": compared,
                     "path_steps": steps, "time_fraction": frac,
                     "mean_distance": (d.get("mean_distance") if g in measured else None),
                     # the exact coefficient, not a rounded one: three groups at round(1/3, 6)
                     # sum to 0.999999 and rebuild 0.1999998 from 0.1/0.2/0.3. Display rounds;
                     # the payload must stay reconstructable.
                     "aggregate_weight": (1.0 / len(measured) if g in measured else 0.0),
                     "one_sided_steps": int(d.get("one_sided_steps", 0)),
                     "unmeasurable_steps": int(d.get("unmeasurable_steps", 0)),
                     "unobserved_steps": int(d.get("unobserved_steps", 0))}
    group_fraction = round(len(measured) / len(names), 4) if names else 0.0
    time_min = round(min(times.values()), 4) if times else 0.0
    time_mean = round(sum(times.values()) / len(times), 4) if times else 0.0
    partial = bool(measured) and (bool(missing) or any(v < 1.0 for v in times.values()))
    return {"expected_groups": list(names), "measured_groups": measured,
            "missing_groups": missing,
            "group_fraction": group_fraction,
            "fraction": group_fraction,          # kept: previous name for the group coverage
            "time_fraction_min": time_min, "time_fraction_mean": time_mean,
            "time_fraction_per_group": times,
            "partial": partial,
            "partial_reason": ([] if not partial else
                               (["missing_groups"] if missing else [])
                               + (["partial_time_coverage"]
                                  if any(v < 1.0 for v in times.values()) else [])),
            "no_pass_line": "partial is reported with its numbers; no threshold decides whether "
                            "the comparison was sufficient",
            "aggregation": {
                "rule": "equal mean of the MEASURED groups' own mean distances",
                "weight_per_measured_group": (1.0 / len(measured) if measured else 0.0),
                "measured_group_count": len(measured),
                "missing_groups_contribute": "nothing: no distance and no weight. A missing group "
                                             "is not a zero difference.",
                "is_not": "the DTW path or ranking group weights. Those belong to the cost "
                          "configuration and are unrelated to this per-axis mean.",
                "evidence_is_uneven": "a group measured once and a group measured many times "
                                      "carry the same weight here. The per-group compared_steps "
                                      "say how much each one actually rests on.",
                "weight_precision": "the exact 1/n, so sum(mean_distance * aggregate_weight) "
                                    "reproduces this axis's mean to floating precision. The "
                                    "published distance field is rounded for display; the "
                                    "reconstruction matches the unrounded mean.",
                "validated": "no. This is the aggregation already in use; whether it is suitable "
                             "for judging an execution has not been established.",
            },
            "per_group": detail}


def axis_check(query: dict, word_id: str, reference_set: list[dict], frozen_config: dict,
               axis: str) -> dict:
    """Is this axis inside the range measured for this word?

    Kept apart from the ranking on purpose. "Which word" is a relative question answered by
    comparing candidates; "is this axis right" is an absolute one that a ranking cannot express -
    a wrong handshape still ranks first when no other reference is closer.

    Upper bound only: being farther than every correct attempt is the error signal; being closer
    is not, and a perfect match scores 0.
    """
    if _role_feedback_enabled(frozen_config):
        return {"status": "NO_BAND", "axis": axis, "word_id": word_id,
                "feedback_policy": FEEDBACK_ROLE_POLICY,
                "calibration_status": "UNCALIBRATED_FEEDBACK_POLICY",
                "reason": "existing bands were measured under a different aggregation policy; "
                          "they cannot judge the role-filtered diagnostic"}
    cfg = (frozen_config.get("handshape_check") or {})
    band = ((cfg.get("axes") or {}).get(axis) or {}).get("bands", {}).get(word_id)
    if not band:
        return {"status": "NO_BAND", "axis": axis, "word_id": word_id,
                "reason": f"no measured {axis} band for this word; it cannot be judged"}
    ref = next((r for r in reference_set if r["word_id"] == word_id), None)
    if ref is None:
        return {"status": "NO_BAND", "axis": axis, "word_id": word_id,
                "reason": "word not in the candidate set"}

    cost_cfg = {k: v for k, v in frozen_config.get("cost", {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    residuals = dtw_group_residuals(query, ref, **cost_cfg)
    groups = residuals.get("groups", {})
    support = _axis_support(groups, AXIS_GROUPS[axis])
    vals = [groups[g]["mean_distance"] for g in support["measured_groups"]]
    if not vals:
        return {"status": "UNMEASURABLE", "axis": axis, "word_id": word_id,
                "support": support, "runtime_version": residuals.get("runtime_version"),
                "reason": "no frame carried the joints this axis needs"}
    d = float(np.mean(vals))
    # Headroom: the band comes from attempts recorded in ONE sitting, which are correlated and so
    # understate how much a correct performance really varies. The multiplier is the 99th
    # percentile of the leave-one-out ratio on the development set alone - no frozen recording.
    headroom = float(((cfg.get("axes") or {}).get(axis) or {}).get("headroom", 1.0))
    limit = band["hi"] * headroom
    inside = d <= limit
    # A partial axis gets its own status so a display cannot paint it as a clean result: the mean
    # was taken over fewer groups than the axis has, which moves it for a reason that has nothing
    # to do with the performance.
    base = "IN_BAND" if inside else "OUT_OF_BAND"
    status = base + "_PARTIAL_SUPPORT" if support["partial"] else base
    return {
        "status": status, "band_verdict": base,
        "axis": axis, "word_id": word_id,
        "support": support, "runtime_version": residuals.get("runtime_version"),
        "distance": round(d, 5),
        "band_max": round(limit, 5), "band_observed": [band["lo"], band["hi"]],
        "headroom": headroom,
        "band_n": band["n"],
        "note": (f"{axis} is no farther than any correct attempt measured for this word"
                 if inside else
                 f"{axis} is farther than every correct attempt measured for this word"),
        "caveat": ("band from one participant; 2D projection distorts depth"
                   + ("; only part of this axis was observed on both sides, so the distance is a "
                      "partial comparison" if support["partial"] else "")),
    }


def axis_checks(query: dict, word_id: str, reference_set: list[dict],
                frozen_config: dict) -> dict:
    axes = ((frozen_config.get("handshape_check") or {}).get("axes") or {})
    return {a: axis_check(query, word_id, reference_set, frozen_config, a)
            for a in AXIS_GROUPS if a in axes}


def axis_deviation(query: dict, word_id: str, reference_set: list[dict],
                   frozen_config: dict) -> dict:
    """Which axis deviates most, in units every axis shares.

    The old descriptive scale is computed from reference clips rather than from this user's
    calibration. That does NOT establish body-proportion invariance, new-person accuracy, or
    execution correctness. The role-filtered candidate reports unscaled per-axis measurements
    without this cross-axis ranking; its old personal bands are not reused.

    The unit is the median distance between DIFFERENT words on that axis, measured over the
    reference set alone. A deviation of 1.0 means "as different as two different words typically
    are on this axis".
    """
    scales = ((frozen_config.get("handshape_check") or {}).get("axis_scales") or {})
    ref = next((r for r in reference_set if r["word_id"] == word_id), None)
    role_filtered = _role_feedback_enabled(frozen_config)
    observed_movement = role_filtered and (frozen_config.get("feedback") or {}).get("observed_movement_structure") is True
    if ref is None or (not scales and not role_filtered):
        return {"status": "UNAVAILABLE", "reason": "no axis scales or word not in the set"}

    cost_cfg = {k: v for k, v in frozen_config.get("cost", {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    residuals = dtw_group_residuals(query, ref, **cost_cfg)
    groups = residuals.get("groups", {})
    out = {}
    for axis, names in AXIS_GROUPS.items():
        if observed_movement and axis == "movement":
            continue  # a partly imputed mixed-group mean is not the new movement feedback
        support = _feedback_support(groups, names, query, ref, frozen_config)
        vals = [groups[g]["mean_distance"] for g in support["measured_groups"]]
        scale = scales.get(axis)
        if not vals or (not scale and not role_filtered):
            continue
        d = float(np.mean(vals))
        out[axis] = {"distance": round(d, 5),
                     "deviation": None if role_filtered else round(d / scale, 4),
                     "support": support, "partial_support": support["partial"]}
    if role_filtered:
        result = {
            "status": "OK" if out else "UNAVAILABLE", "word_id": word_id, "axes": out,
            "order": [], "worst_axis": None, "worst_deviation": None,
            "feedback_policy": FEEDBACK_ROLE_POLICY,
            "normalization_status": "UNCALIBRATED_FEEDBACK_POLICY",
            "runtime_version": residuals.get("runtime_version"),
            "partial_axes": [a for a in out if out[a]["partial_support"]],
            "unavailable_axes": [a for a in AXIS_GROUPS if a not in out],
            "support_by_axis": {a: _feedback_support(groups, names, query, ref, frozen_config)
                                for a, names in AXIS_GROUPS.items() if not (observed_movement and a == "movement")},
            "unit": "raw within-axis feature distance; not comparable across axes",
            "note": "diagnostic candidate only. Existing inter-word scales and personal bands "
                    "were measured with the old aggregation and are not reused. No execution "
                    "verdict or cross-axis ordering is available.",
        }
        if observed_movement:
            from .movement_feedback import compare_sources
            result["movement_observation"] = compare_sources(query.get("movement_observation"), ref.get("movement_observation"))
            result["unavailable_axes"] = [a for a in result["unavailable_axes"] if a != "movement"]
            result["movement_representation"] = "observed components and direction changes, no aggregate distance"
        return result
    if not out:
        return {"status": "UNAVAILABLE", "reason": "no axis could be measured"}
    order = sorted(out, key=lambda a: -out[a]["deviation"])
    return {
        "status": "OK", "word_id": word_id, "axes": out, "order": order,
        "runtime_version": residuals.get("runtime_version"),
        "partial_axes": [a for a in out if out[a]["partial_support"]],
        "worst_axis": order[0], "worst_deviation": out[order[0]]["deviation"],
        "unit": "median distance between different words on that axis, from the reference set",
        "note": "a difference report, not a verdict. It says which axis is furthest from the "
                "reference, never whether the performance was correct.",
    }


def handshape_check(query: dict, word_id: str, reference_set: list[dict],
                    frozen_config: dict) -> dict:
    """Backwards-compatible view of the handshape axis."""
    return axis_check(query, word_id, reference_set, frozen_config, "handshape")


def task_match(result: dict, requested_word_id: str, observation=None,
               reference_set: list[dict] | None = None, frozen_config: dict | None = None) -> dict:
    """Separate from predict: compares the ranking to what the exercise asked for.

    predict() must never see the requested word, so the axis checks AGAINST THAT WORD belong
    here. Without them the check runs on whatever the ranking guessed, which goes silent the
    moment the ranking picks an uncovered word.
    """
    if result["observation_status"] != "OK" or not result["ranking"]:
        return {"task_match": None, "reason": result["observation_status"]}
    scored = [r for r in result["ranking"] if r.get("distance") is not None]
    rank = next((r["rank"] for r in scored if r["word_id"] == requested_word_id), None)
    out = {
        "requested_word_id": requested_word_id,
        "task_match": bool(scored and scored[0]["word_id"] == requested_word_id),
        "requested_rank": rank,
        "requested_distance": next(
            (r["distance"] for r in scored if r["word_id"] == requested_word_id), None),
        "note": "a correctly performed different word is a task mismatch, not a recognition error",
    }
    if observation is not None and reference_set and frozen_config:
        try:
            seq = F.sequence_features(observation)
            q = F.uniform_time_sample(seq, int(frozen_config["sampling_A"]["budget"]))
            out["axis_checks_vs_requested"] = axis_checks(
                q, requested_word_id, reference_set, frozen_config)
            out["axis_deviation_vs_requested"] = axis_deviation(
                q, requested_word_id, reference_set, frozen_config)
            ref = next((r for r in reference_set if r["word_id"] == requested_word_id), None)
            if ref is not None:
                out["joint_deviation_vs_requested"] = joint_deviation(q, ref, frozen_config)
                # where the hands travelled, on the same path. The original sequence goes with it
                # so a gap that fell between two sampled rows is not drawn over.
                out["location_path_vs_requested"] = location_path(
                    q, ref, frozen_config, query_source=source_observation(seq))
        except F.FeatureError as exc:
            out["axis_checks_vs_requested"] = {"error": str(exc)}
    return out
