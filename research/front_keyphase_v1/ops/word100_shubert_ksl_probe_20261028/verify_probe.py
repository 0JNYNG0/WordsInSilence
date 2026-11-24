"""Independent saved-feature arithmetic audit; never calls a learned model.

Reads only the frozen public probe and its pinned public sources. It does not
provide the blinded RGB reviewers with matching results. Educational ground
truth and semantic correctness are explicitly outside this arithmetic audit.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import random
import sys
import numpy as np

U = Path(__file__).resolve().parent
P = U.parent / "word100_shubert_feasibility_20261028"
G = U.parent / "word100_guide_video_feedback_20261028"
sys.path.insert(0, str(G))
from motion_observer import read_observation, extract_features


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def distance_cosine(a, b):
    # Double precision and explicit normalized vectors independently check the
    # original float32 matrix formula; tolerate only floating rounding.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    assert an > 0 and bn > 0
    return max(0., min(2., 1. - float(np.dot(a / an, b / bn))))


def geometry_cost(q, r):
    groups = [cols for name, cols in q["groups"].items()
              if not name.endswith("_speed") and not name.endswith("_diagnostic")]
    matrix = np.empty((len(q["x"]), len(r["x"])))
    for i in range(len(q["x"])):
        for j in range(len(r["x"])):
            scores = []
            for cols in groups:
                expected = [c for c in cols if r["mask"][j, c]]
                # Reference count is independent of missing query observations.
                errors = [min(abs(float(q["x"][i, c] - r["x"][j, c])), 2.)
                          if q["mask"][i, c] else 1. for c in expected]
                scores.append(sum(errors) / len(expected) if expected else 0.)
            matrix[i, j] = sum(scores) / len(scores)
    return matrix


def independent_dtw(cost):
    # Coordinate-key dictionary recurrence, distinct from implementation's
    # padded array and predecessor integer trace. Priority diag/up/left.
    n, m = cost.shape
    states = {(-1, -1): (0., [])}
    for i in range(n):
        for j in range(m):
            predecessors = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
            choices = [(states[p][0], rank, p) for rank, p in enumerate(predecessors) if p in states]
            score, _, pred = min(choices)
            states[i, j] = (score + float(cost[i, j]), states[pred][1] + [[i, j]])
    total, path = states[n - 1, m - 1]
    mapping = [int(np.median([j for ii, j in path if ii == i])) for i in range(n)]
    return path, mapping, total / len(path)


def verify_native_geometry(a, f):
    """Recalculate every finger angle, all position fields and velocities.

    The old feature extractor is pinned; this audits its mathematical output
    from original coordinates, rather than comparing it with itself.
    """
    xy = a["xy_px"].astype(np.float64)
    valid = a["observed"].astype(bool) & a["usable"].astype(bool) & np.isfinite(xy).all(axis=-1)
    scale = np.sqrt(((xy[:, 11] - xy[:, 12]) ** 2).sum(axis=-1))
    body = valid[:, 11] & valid[:, 12] & (scale > 1e-8) & np.isfinite(scale)
    center = (xy[:, 11] + xy[:, 12]) / 2
    normalized = (xy - center[:, None]) / scale[:, None, None]
    check_count = 0
    max_delta = 0.

    def check(name, value, mask):
        nonlocal check_count, max_delta
        col = f["feature_names"].index(name)
        mask = mask & np.isfinite(value)
        assert np.array_equal(mask, f["mask"][:, col]), name
        assert np.allclose(value[mask], f["x"][mask, col], atol=1e-11, rtol=1e-10), name
        if mask.any():
            max_delta = max(max_delta, float(np.max(np.abs(value[mask] - f["x"][mask, col]))))
        check_count += 1

    # Triplets independently state the same documented anatomical convention.
    triplets = [("thumb_mcp", (1, 2, 3)), ("thumb_ip", (2, 3, 4))]
    for finger, base in (("index", 5), ("middle", 9), ("ring", 13), ("little", 17)):
        triplets.extend([(finger + "_mcp", (0, base, base + 1)),
                         (finger + "_pip", (base, base + 1, base + 2)),
                         (finger + "_dip", (base + 1, base + 2, base + 3))])
    for side, offset, arm in (("L", 33, (11, 13, 15)), ("R", 54, (12, 14, 16))):
        for label, ids in [(side + "_elbow_angle", arm)] + [
                (side + "_" + name + "_angle", tuple(offset + i for i in ids)) for name, ids in triplets]:
            aa, bb, cc = ids
            first, second = xy[:, aa] - xy[:, bb], xy[:, cc] - xy[:, bb]
            denominator = np.sqrt((first ** 2).sum(1)) * np.sqrt((second ** 2).sum(1))
            good = valid[:, list(ids)].all(1) & np.isfinite(denominator) & (denominator > 1e-12)
            value = np.full(len(xy), np.nan)
            cosine = (first[good] * second[good]).sum(1) / denominator[good]
            value[good] = np.arccos(np.clip(cosine, -1., 1.)) / math.pi
            check(label, value, good)
        for label, ids in (("hand_wrist", [offset]), ("pose_wrist", [arm[2]]),
                           ("elbow", [arm[1]]), ("index_tip", [offset + 8]),
                           ("palm_center", [offset + k for k in (0, 5, 9, 13, 17)])):
            values = normalized[:, ids].mean(1)
            good = body & valid[:, ids].all(1)
            for dimension, axis in enumerate("xy"):
                check(f"{side}_{label}_{axis}", values[:, dimension], good)
        interval = np.diff(a["timestamp_us"]) / 1e6
        good = np.r_[False, body[1:] & body[:-1] & valid[1:, offset] & valid[:-1, offset]
                     & (np.diff(a["frame_id"]) == 1) & (interval > 0) & (interval <= .15)]
        velocity = np.full((len(xy), 2), np.nan)
        velocity[1:] = np.diff(normalized[:, offset], axis=0) / interval[:, None]
        for dimension, axis in enumerate("xy"):
            check(f"{side}_wrist_velocity_{axis}", velocity[:, dimension], good)
    return {"independently_recomputed_feature_columns": check_count, "max_abs_delta": max_delta}


def verify_evaluation(correspondence, frozen, samples, metadata):
    evaluation, saved_consensus = read(U / "EVALUATION.json"), read(U / "CONSENSUS.json")
    rubric = read(U / "RUBRIC.json")
    fields = rubric["fields"]
    assert evaluation["frozen_sha256"] == sha(U / "FROZEN.json")
    assert evaluation["correspondence_sha256"] == sha(U / "CORRESPONDENCE.json")
    assert evaluation["consensus_sha256"] == sha(U / "CONSENSUS.json")
    assert evaluation["script_sha256"] == sha(U / "evaluate_probe.py")
    reviews = []
    for file in ("REVIEW_A.json", "REVIEW_B.json"):
        assert saved_consensus["review_pins"][file] == sha(U / file)
        r = read(U / file)
        assert len(r["rows"]) == 108
        assert len(r["images_viewed"]) == len(set(r["images_viewed"])) == 36
        # Reviewers use relative or absolute names; all must identify the exact
        # fixed public contact-sheet inventory, without unexplained additions.
        expected_sheets = {Path(p).name for p in read(U / "SAMPLES.json")["sheets"]}
        assert {Path(p).name for p in r["images_viewed"]} == expected_sheets
        by_id = {x["sample_id"]: x["values"] for x in r["rows"]}
        assert set(by_id) == set(samples)
        for values in by_id.values():
            assert len(values) == 10
            assert all(v in (0, 1, None) for v in values[:7])
            assert all(v in ("up", "down", "still", None) for v in values[7:9])
            assert values[9] in ("closer", "farther", "still", None)
        reviews.append(by_id)
    consensus, review_counts = {}, Counter()
    for sid in samples:
        consensus[sid] = []
        for fi in range(10):
            left, right = reviews[0][sid][fi], reviews[1][sid][fi]
            if left is None or right is None:
                value, reason = None, "one_or_both_unknown"
            elif left != right:
                value, reason = None, "known_disagreement"
            else:
                value, reason = left, "agreed_known"
            consensus[sid].append(value)
            review_counts[reason] += 1
    assert consensus == saved_consensus["values"]
    assert dict(review_counts) == saved_consensus["counts"] == evaluation["review_counts"]
    assert sum(review_counts.values()) == evaluation["total_review_cells"] == 1080
    assert saved_consensus["fields"] == fields
    sources = {}
    for row in frozen["rows"]:
        with np.load(P / "inputs" / row["alias"] / "provenance.npz", allow_pickle=False) as a:
            sources[row["alias"]] = np.array(a["source_kind"])

    def has_current(sid, field):
        sample = samples[sid]
        slots = ([1] if field.startswith("R_") else [0]) if field.startswith(("R_", "L_")) else [0, 1]
        times = sample["context_indices"] if field in fields[7:] else [sample["frame_index"]]
        return all(sources[sample["alias"]][t, slot] == 1 for t in times for slot in slots)

    shape_records, shape_exclusions = [], Counter()
    shape_support = {}
    for query in correspondence["shape_queries"]:
        sid = query["query_sample_id"]
        for suffix in ("index_only", "open_four"):
            field = query["side"] + "_" + suffix
            fi = fields.index(field)
            target = consensus[sid][fi]
            if target is None:
                shape_exclusions["query_visual_unknown"] += 1
                continue
            if not (query["query_current"] and query["query_angles_valid"] and query["matches"]):
                shape_exclusions["query_or_reference_sensor_unavailable"] += 1
                continue
            same_class_candidates = [rid for rid in query["candidate_ids"] if consensus[rid][fi] == target and has_current(rid, field)]
            if not same_class_candidates:
                shape_exclusions["target_class_absent_from_other_words"] += 1
                continue
            count = shape_support.setdefault(field, Counter())
            count[str(target)] += 1
            references = {method: match["reference_sample_id"] for method, match in query["matches"].items()}
            outcomes = {}
            for method, rid in references.items():
                value = consensus[rid][fi]
                outcomes[method] = "unknown" if value is None or not has_current(rid, field) else ("agree" if target == value else "different")
            shape_records.append({"query_sample_id": sid, "query": query["query"], "field": field,
                                  "target": target, "annotation_active": metadata[sid]["annotation_active"],
                                  "reference_ids": references, "outcomes": outcomes})
    assert shape_records == evaluation["shape"]["records"]
    assert dict(shape_exclusions) == evaluation["shape_exclusions"]
    assert {key: dict(value) for key, value in shape_support.items()} == evaluation["shape_class_support"]
    assert len(shape_records) + sum(shape_exclusions.values()) == evaluation["hand_query_feature_slots"] == 144

    stage_lookup = {}
    for pair in correspondence["stage_pairs"]:
        stage_lookup.setdefault((pair["query"], pair["reference"]), {})[pair["method"]] = pair
    stage_records, stage_exclusions = [], Counter()
    for (qalias, ralias), methods in stage_lookup.items():
        assert set(methods) == {"geometry", "shubert", "time_fraction"}
        for k, sid in enumerate(methods["geometry"]["query_sample_ids"]):
            for fi, field in enumerate(fields):
                target = consensus[sid][fi]
                if target is None:
                    stage_exclusions["query_visual_unknown"] += 1
                    continue
                if not has_current(sid, field):
                    stage_exclusions["query_current_evidence_unavailable"] += 1
                    continue
                references = {method: pair["reference_sample_ids"][pair["mapping"][k]] for method, pair in methods.items()}
                outcomes = {}
                for method, rid in references.items():
                    value = consensus[rid][fi]
                    outcomes[method] = "unknown" if value is None or not has_current(rid, field) else ("agree" if target == value else "different")
                stage_records.append({"query_sample_id": sid, "query": qalias, "reference": ralias, "field": field,
                                      "target": target, "group": "motion" if field in fields[7:] else "static",
                                      "annotation_active": metadata[sid]["annotation_active"],
                                      "reference_ids": references, "outcomes": outcomes})
    assert stage_records == evaluation["stage"]["records"]
    assert dict(stage_exclusions) == evaluation["stage_exclusions"]
    assert len(stage_records) + sum(stage_exclusions.values()) == evaluation["stage_query_feature_slots"] == 720

    def count_rows(rows, method):
        labels = [row["outcomes"][method] for row in rows]
        agree, different, unknown = (labels.count(name) for name in ("agree", "different", "unknown"))
        known, n = agree + different, len(rows)
        assert known + unknown == n
        return {"total": n, "agree": agree, "different": different, "unknown": unknown, "known": known,
                "known_agreement": agree / known if known else None,
                "agreement_over_all": agree / n if n else None, "coverage": known / n if n else None}

    def summary(rows, methods):
        indices = set(range(len(rows)))
        for method in methods:
            indices &= {i for i, row in enumerate(rows) if row["outcomes"][method] in ("agree", "different")}
        common = [rows[i] for i in sorted(indices)]
        return {"all_eligible": {method: count_rows(rows, method) for method in methods},
                "common_known_count": len(common), "common_known": {method: count_rows(common, method) for method in methods}}

    for kind, rows, methods in (("shape", shape_records, ("geometry", "dino", "shubert")),
                                 ("stage", stage_records, ("geometry", "shubert", "time_fraction"))):
        block = evaluation[kind]
        active = [r for r in rows if r["annotation_active"]]
        assert block["all"] == summary(rows, methods)
        assert block["annotation_active"] == summary(active, methods)
        if kind == "shape":
            for field in fields[:4]:
                assert block["by_feature"][field] == summary([r for r in rows if r["field"] == field], methods)
        else:
            for group in ("static", "motion"):
                assert block["active_" + group] == summary([r for r in active if r["group"] == group], methods)
            for alias in sorted({r["query"] for r in rows}):
                assert block["by_query"][alias] == summary([r for r in active if r["query"] == alias], methods)
    assert evaluation["training_runs"] == evaluation["operating_changes"] == 0
    assert evaluation["private_access"] is False
    # Full denominators remain separately visible even though the main method
    # comparison uses a common known intersection to avoid missingness bias.
    return {"status": "PASS", "evaluation_sha256": sha(U / "EVALUATION.json"),
            "evaluation_script_sha256": sha(U / "evaluate_probe.py"),
            "consensus_sha256": sha(U / "CONSENSUS.json"), "review_counts": dict(review_counts),
            "shape_records": len(shape_records), "shape_excluded": sum(shape_exclusions.values()),
            "shape_total_slots": 144, "stage_records": len(stage_records),
            "stage_excluded": sum(stage_exclusions.values()), "stage_total_slots": 720,
            "shape_common_known_all": evaluation["shape"]["all"]["common_known_count"],
            "shape_common_known_active": evaluation["shape"]["annotation_active"]["common_known_count"],
            "stage_common_known_all": evaluation["stage"]["all"]["common_known_count"],
            "stage_common_known_active": evaluation["stage"]["annotation_active"]["common_known_count"],
            "unknown_retained": True, "same_common_support_compared": True,
            "interpretation": "Arithmetic of two blinded AI drafts only; no semantic or educational accuracy assertion."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    frozen, samples_doc, out = read(U / "FROZEN.json"), read(U / "SAMPLES.json"), read(U / "CORRESPONDENCE.json")
    assert sha(U / "FROZEN.json") == out["frozen_sha256"] == samples_doc["frozen_sha256"]
    assert sha(U / "RUBRIC.json") == samples_doc["rubric_sha256"]
    assert sha(U / "EVALUATION_METRICS.json") == out["metrics_sha256"]
    assert sha(U / "compute_correspondence.py") == out["script_sha256"]
    for path, digest in frozen["source_pins"].items():
        assert sha(path) == digest, path
    assert frozen["training"] is False and out["training_runs"] == out["model_forward_calls"] == 0
    assert out["private_access"] is False
    original_order = list(read(P / "inputs/MANIFEST.json")["rows"])
    random.Random(20261028).shuffle(original_order)
    assert [r["alias"] for r in frozen["rows"]] == [r["alias"] for r in original_order]
    assert len(samples_doc["samples"]) == 108 and len(samples_doc["sheets"]) == 36
    assert len({s["sample_id"] for s in samples_doc["samples"]}) == 108
    for sheet in samples_doc["sheets"]:
        assert (U / sheet).is_file()
    samples = {s["sample_id"]: s for s in samples_doc["samples"]}
    metadata = {s["sample_id"]: s for s in out["sample_metadata"]}
    cache, diagnostics = {}, []
    current_count = Counter()
    for row in frozen["rows"]:
        alias, n = row["alias"], row["frames"]
        assert row["anchors"] == [int(round((n - 1) * k / 10)) for k in range(1, 10)]
        folder, feature_folder = P / "inputs" / alias, P / "features" / alias
        assert sha(folder / "DONE.json") == row["done_sha256"]
        for key, name in (("native", "native_observation.npz"), ("provenance", "provenance.npz")):
            assert sha(folder / name) == out["input_pins"][alias][key]
        for key, name in (("streams", "streams.npz"), ("last_ffn", "last_ffn.npy")):
            assert sha(feature_folder / name) == out["input_pins"][alias][key]
        a = read_observation(folder / "native_observation.npz")
        assert len(a["frame_id"]) == n
        assert np.array_equal(a["frame_id"], np.arange(n))
        assert np.max(np.abs(a["timestamp_us"] - np.rint(np.arange(n) * 1e6 / 30))) <= 1
        full = extract_features(a)
        assert full["x"].shape == (n, 69)
        independent = verify_native_geometry(a, full)
        idx = row["anchors"]
        features = {**full, "x": full["x"][idx], "mask": full["mask"][idx]}
        with np.load(folder / "provenance.npz", allow_pickle=False) as z:
            source_kind = z["source_kind"][idx, :2]
        with np.load(feature_folder / "streams.npz", allow_pickle=False) as z:
            hand = {"L": z["left_hand"][idx], "R": z["right_hand"][idx]}
        shubert = np.load(feature_folder / "last_ffn.npy", allow_pickle=False)[idx]
        ids = [f"{row['blind_clip']}_{k+1:02d}" for k in range(9)]
        for k, sid in enumerate(ids):
            sample, meta = samples[sid], metadata[sid]
            assert sample["alias"] == alias and sample["frame_index"] == idx[k]
            assert sample["context_indices"] == [max(0, min(n-1, idx[k] + delta)) for delta in (-6, 0, 6)]
            assert meta["frame_id"] == idx[k] and meta["timestamp_us"] == int(a["timestamp_us"][idx[k]])
            active = any(round(start * 1e6) <= meta["timestamp_us"] <= round(end * 1e6) for start, end in row["annotation_intervals"])
            assert meta["annotation_active"] == active
            for side, col in (("L", 0), ("R", 1)):
                assert meta[side + "_current"] == (source_kind[k, col] == 1)
                current_count[str(int(source_kind[k, col]))] += 1
        cache[alias] = {"row": row, "g": features, "ids": ids, "hand": hand,
                        "shubert": shubert, "current": source_kind == 1}
        diagnostics.append({"alias": alias, **independent})
    refs = [v for v in cache.values() if v["row"]["recorded_person_id"] in (1, 2)]
    queries = [v for v in cache.values() if v["row"]["recorded_person_id"] == 3]
    assert len(refs) == 8 and len(queries) == 4
    pairs = out["stage_pairs"]
    assert len(pairs) == 24
    stage_checks, max_cost_delta = [], 0.
    for pair in pairs:
        q, r = cache[pair["query"]], cache[pair["reference"]]
        assert q in queries and r in refs
        assert q["row"]["word_id"] == r["row"]["word_id"]
        assert pair["query_sample_ids"] == q["ids"] and pair["reference_sample_ids"] == r["ids"]
        method = pair["method"]
        if method == "time_fraction":
            expected_path, mapping, cost = [[k, k] for k in range(9)], list(range(9)), None
        else:
            matrix = geometry_cost(q["g"], r["g"]) if method == "geometry" else np.array([
                [distance_cosine(a, b) for b in r["shubert"]] for a in q["shubert"]])
            expected_path, mapping, cost = independent_dtw(matrix)
            delta = abs(pair["mean_path_cost"] - cost)
            max_cost_delta = max(max_cost_delta, delta)
            assert delta < 2e-6, (method, delta)
        assert pair["path"] == expected_path and pair["mapping"] == mapping, pair
        assert {i for i, j in pair["path"]} == {j for i, j in pair["path"]} == set(range(9))
        assert all((b[0]-a[0], b[1]-a[1]) in ((1, 1), (1, 0), (0, 1)) for a, b in zip(pair["path"], pair["path"][1:]))
        stage_checks.append({"query": pair["query"], "reference": pair["reference"], "method": method,
                             "path_length": len(expected_path), "mapped_unique_reference_anchors": len(set(mapping)),
                             "all_reference_anchors_in_full_path": True})
    assert len(out["shape_queries"]) == 72
    shape_checks, max_distance_delta = [], 0.
    for entry in out["shape_queries"]:
        q, side = cache[entry["query"]], entry["side"]
        slot = 0 if side == "L" else 1
        k = q["ids"].index(entry["query_sample_id"])
        cols = q["g"]["groups"][side + "_finger_angle"]
        expected_candidates = [(r, j) for r in refs if r["row"]["word_id"] != q["row"]["word_id"]
                               for j in range(9) if r["current"][j, slot] and r["g"]["mask"][j, cols].all()]
        assert entry["candidate_ids"] == [r["ids"][j] for r, j in expected_candidates]
        current, angles = bool(q["current"][k, slot]), bool(q["g"]["mask"][k, cols].all())
        assert entry["query_current"] == current and entry["query_angles_valid"] == angles
        allowed = current and angles and bool(expected_candidates)
        assert bool(entry["matches"]) == allowed
        for method, match in entry["matches"].items():
            values = []
            for r, j in expected_candidates:
                if method == "geometry":
                    value = sum(abs(float(q["g"]["x"][k, c] - r["g"]["x"][j, c])) for c in cols) / len(cols)
                elif method == "dino":
                    value = distance_cosine(q["hand"][side][k], r["hand"][side][j])
                else:
                    value = distance_cosine(q["shubert"][k], r["shubert"][j])
                values.append(value)
            selected = entry["candidate_ids"].index(match["reference_sample_id"])
            delta = abs(values[selected] - match["distance"])
            max_distance_delta = max(max_distance_delta, delta)
            assert delta < 2e-6 and values[selected] <= min(values) + 2e-6
        shape_checks.append({"query_sample_id": entry["query_sample_id"], "side": side,
                             "candidates": len(expected_candidates), "query_eligible": bool(allowed)})
    result = {"status": "PASS_CORRESPONDENCE_ARITHMETIC", "scope": "Independent saved public feature/mask/path/distance audit; no educational or human ground-truth claim",
              "verifier": "separate Codex agent shubert_adapter_verifier", "verifier_sha256": sha(__file__),
              "frozen_sha256": sha(U / "FROZEN.json"), "correspondence_sha256": sha(U / "CORRESPONDENCE.json"),
              "samples": 108, "blind_sheets": 36, "clips": 12, "recorded_reference_ids": [1, 2], "recorded_query_ids": [3],
              "independent_geometry_checks": diagnostics, "crop_source_kind_counts_108anchors_twohands": dict(current_count),
              "stage_checks": stage_checks, "max_independent_stage_cost_delta": max_cost_delta,
              "shape_checks": shape_checks, "max_independent_shape_distance_delta": max_distance_delta,
              "shape_queries_eligible": sum(x["query_eligible"] for x in shape_checks),
              "shape_queries_retained": 72, "model_forward_calls": 0, "training_runs": 0, "private_access": False,
              "evaluation_arithmetic": "PENDING_EVALUATION_JSON",
              "limitations": ["9 anchors per clip are sparse physical-feature correspondences, not dense semantic stage boundaries.",
                              "SHuBERT has whole-clip/body/both-hand context while DINO/angles are per-hand instantaneous observations; this is a deployment-role comparison, not isolated architecture ablation.",
                              "Rest/entry/exit anchors included. Active source-annotation subset must be reported separately.",
                              "Full DTW path retains all reference anchors, but query-to-reference median mapping may select fewer unique reference anchors; no missing-action guarantee follows.",
                              "Blinded Codex agreement is not independent human sign-language correctness.",
                              "Source-kind current denotes detector/crop provenance, not proof that hidden fingers are visible."]}
    if args.evaluation:
        result["evaluation_arithmetic"] = verify_evaluation(out, frozen, samples, metadata)
        result["status"] = "PASS_CORRESPONDENCE_AND_EVALUATION_ARITHMETIC"
    (U / "INDEPENDENT_VERIFY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["공개 KSL 비교의 독립 산술 검산: PASS", "12영상·108시점·72손 질의·24단계 비교를 확인했습니다.",
            "동결 해시/원본 시점/좌우 현재 관측/다른 단어 후보/거리/DTW 경로를 검산했습니다.",
            f"손 비교 가능 질의: {result['shape_queries_eligible']}/72. 제외 질의도 분모에 보존합니다.",
            "학습 또는 모델 재추론은 실행하지 않았습니다. " + ("EVALUATION 합의·제외·교집합·비율도 원자료로 검산했습니다." if args.evaluation else "최종 EVALUATION 산술 검산은 아직 대기 중입니다."),
            "수어 교육 정답이나 손 관측의 실제 정확성을 증명하는 검산은 아닙니다."]
    (U / "INDEPENDENT_VERIFY.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "eligible": result["shape_queries_eligible"],
                      "max_stage_cost_delta": max_cost_delta, "max_shape_distance_delta": max_distance_delta}))


if __name__ == "__main__":
    main()
