"""Source-conditioned comparison after frozen neural observation and prototype DTW.

This module does not observe RGB, infer a new feature, read reference videos,
select a model, or turn partial evidence into a whole-sign judgement.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import time

U = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_state(point, thresholds):
    """Model evidence direction, never semantic correctness or finger angles."""
    p = point.get("probability")
    if not point.get("image_available") or p is None:
        return "UNOBSERVED", None
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid point probability")
    g = point.get("joint_support")
    if g is None or not math.isfinite(g) or not 0 <= g <= 1:
        raise ValueError("Invalid joint support")
    direction = ("compatible" if p >= thresholds["compatible_min"] else
                 "counter" if p <= thresholds["counter_max"] else None)
    if direction is None:
        return "UNCERTAIN", None
    if g < thresholds["joint_support_min"]:
        return "IMAGE_TIME_ESTIMATE", direction
    return ("MODEL_COMPATIBLE" if direction == "compatible" else "MODEL_COUNTER_EVIDENCE"), direction


def summarize(points):
    counts = Counter(p["observation"] for p in points)
    positive = counts["MODEL_COMPATIBLE"] > 0
    negative = counts["MODEL_COUNTER_EVIDENCE"] > 0
    if not points or counts["UNOBSERVED"] == len(points):
        state = "INSUFFICIENT_OBSERVATION"
    elif positive and negative:
        state = "MIXED_OBSERVATIONS"
    elif positive:
        state = "COMPATIBLE_EVIDENCE_PRESENT"
    elif negative:
        state = "COUNTER_EVIDENCE_TIME_UNCONFIRMED"
    else:
        state = "UNCERTAIN_OR_ESTIMATED_ONLY"
    return dict(status=state, total_points=len(points), counts=dict(counts),
        observed_model_points=counts["MODEL_COMPATIBLE"] + counts["MODEL_COUNTER_EVIDENCE"],
        coverage=(counts["MODEL_COMPATIBLE"] + counts["MODEL_COUNTER_EVIDENCE"]) / len(points) if points else 0.,
        denominator_includes_missing=True, whole_window_confirmed=False,
        has_unresolved_points=any(p["observation"] not in {"MODEL_COMPATIBLE", "MODEL_COUNTER_EVIDENCE"} for p in points),
        short_counterpoints_preserved=True)


def message(item):
    text = "‘" + item["source_text"] + "’"
    status = item["summary"]["status"]
    if status == "COMPATIBLE_EVIDENCE_PRESENT":
        return text + "과 관련된 특징이 보인다는 분석이 있습니다. 확인한 시간과 불확실한 시간을 나누어 표시합니다."
    if status in {"MIXED_OBSERVATIONS", "COUNTER_EVIDENCE_TIME_UNCONFIRMED"}:
        return text + "과 대조할 특징이 다르게 관측된 시점이 있습니다. 동작 전환이나 시간 연결의 차이일 수 있어 교정으로 확정하지 않습니다."
    if status == "RELATED_MOVEMENT_EVIDENCE_PRESENT":
        return text + "과 관련된 이동 근거가 있습니다. 이동의 전체 수행이나 정확한 끝까지 확인했다는 뜻은 아닙니다."
    return text + "의 해당 특징은 현재 근거로 충분히 비교하지 못했습니다. 잘 맞았다고 처리하지 않습니다."


class GuideComparator:
    def __init__(self, contract_path=None):
        self.path = Path(contract_path or U / "COMPARISON_CONTRACT.json")
        self.contract = read(self.path)
        self.contract_sha = sha(self.path)
        for path, expected in self.contract["pins"].items():
            if sha(path) != expected:
                raise ValueError("Changed frozen comparison input: " + path)
        self.constraints = self.contract["constraints"]

    def attach(self, result, frame_ids, timestamps_us):
        started = time.perf_counter()
        if "guide_feature_comparison" in result:
            raise ValueError("Already compared: use a distinct unmodified base result")
        frames = [int(x) for x in frame_ids]
        times = [int(x) for x in timestamps_us]
        if len(frames) != len(times) or not frames or any(b <= a for a, b in zip(frames, frames[1:])) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Invalid original frame/time axis")
        position = {f: i for i, f in enumerate(frames)}
        observations = result.get("source_components", [])
        items = []
        for constraint in self.constraints:
            if constraint["word_id"] != result["word_id"]:
                continue
            candidates = [c for c in observations if c["component_id"] == constraint["component_id"] and
                any(s["id"] == constraint["clause_id"] for s in c["source_clauses"])]
            if len(candidates) > 1:
                raise ValueError("Ambiguous duplicated component")
            item = dict(clause_id=constraint["clause_id"], component_id=constraint["component_id"],
                source_text=constraint["text"], source_url=constraint["url"], source_entry=constraint["entry_id"],
                source=deepcopy(constraint["source"]), condition_relation=constraint["condition_relation"],
                expected_component_value=constraint["expected_component_value"],
                expectation_is_observation=False, expectation_is_training_label=False,
                applicability="SOURCE_VARIANT_AND_LOCAL_PHASE_NOT_CONFIRMED",
                time_binding="UNBOUND", frame_ids=[], native_points=[],
                summary=dict(status="NO_BOUND_RUNTIME_OBSERVER", total_points=0),
                correction_enabled=False, whole_clause_correctness=None, whole_sign_correctness=None,
                unsupported=deepcopy(constraint["unsupported"]))
            if candidates:
                component = candidates[0]
                clause = next(s for s in component["source_clauses"] if s["id"] == constraint["clause_id"])
                if clause["text"] != constraint["text"] or clause["source"] != constraint["source"]:
                    raise ValueError("Source text or variant provenance differs")
                ff = component["frame_ids"]
                if not ff or any(f not in position for f in ff) or any(b <= a for a, b in zip(ff, ff[1:])):
                    raise ValueError("Invalid component axis")
                expected = frames[position[ff[0]]:position[ff[-1]] + 1]
                if ff != expected:
                    raise ValueError("Component omitted native frames")
                item.update(frame_ids=ff.copy(), start_seconds=times[position[ff[0]]] / 1e6,
                    end_seconds=times[position[ff[-1]]] / 1e6,
                    time_binding="PROPOSED_DTW", previous_component_status=component["status"])
                if constraint["runtime_observer"] == "point":
                    pp = component.get("point_evidence", [])
                    if [p["frame"] for p in pp] != ff:
                        raise ValueError("Point evidence and native extent differ")
                    points = []
                    for p in pp:
                        if abs(p["seconds"] - times[position[p["frame"]]] / 1e6) > 1e-9:
                            raise ValueError("Point time changed")
                        st, direction = point_state(p, self.contract["point_thresholds"])
                        points.append(dict(frame=p["frame"], seconds=p["seconds"],
                            observation=st, model_direction=direction, probability=p["probability"],
                            joint_support=p["joint_support"], image_available=p["image_available"],
                            evidence_origin="FROZEN_NEURAL_OBSERVER_NOT_MANUAL_TRUTH",
                            time_binding="PROPOSED_DTW", guide_violation=None,
                            original_status=p["status"]))
                    item.update(native_points=points, summary=summarize(points))
                elif constraint["runtime_observer"] == "interval":
                    valid = component["status"] == "RELATED_MOVEMENT_OBSERVED" and component.get("guide_time_binding") == "SUPPORTED"
                    item["summary"] = dict(status="RELATED_MOVEMENT_EVIDENCE_PRESENT" if valid else "INSUFFICIENT_MOVEMENT_OR_TIME_EVIDENCE",
                        total_native_frames=len(ff), exact_observer_scope=component.get("evidence_mode"),
                        prior_guide_time_binding=component.get("guide_time_binding"),
                        probability=component.get("probability"), image_coverage=component.get("image_coverage"),
                        complete_action_verified=False, negative_inferred_from_low_probability=False)
            item["message_ko"] = message(item)
            items.append(item)
        out = deepcopy(result)
        out["guide_feature_comparison"] = dict(version=self.contract["version"],
            contract_sha256=self.contract_sha, items=items,
            selected_word_used_for_source_lookup_only=True,
            observed_values_unchanged=True, native_axis_preserved=True,
            individual_reference_comparisons_added=0, new_model_inference=0, new_training=0,
            educational_corrections_enabled=False, accuracy_gain_established=False,
            comparison_seconds=time.perf_counter() - started)
        for key, value in result.items():
            assert out[key] == value, key
        return out
