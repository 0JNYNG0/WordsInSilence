"""Compile directly reviewed physical-process evidence without changing its scope.

No model execution, label densification, image extraction or source mutation.
Original reviews are embedded unchanged; new fields only bind visible pixels to
source and output clocks. A cut type never supplies a process-evidence label.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import traceback
import statistics
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

U = Path(__file__).resolve().parent
OUT = U / "PROCESS_VIEW_TARGETS.json"
VERIFIED = U / "PROCESS_VIEW_TARGETS_VERIFIED.json"
REPORT = U / "PROCESS_VIEW_TARGETS_AUDIT_KO.txt"
REVIEW_FILES = [f"REVIEW_PARTIAL_{w}.json" for w in ("0739", "1133", "1567", "2523")] + ["REVIEW_SENTINEL_PROCESS.json"]
HASHES: dict[str, str] = {}
CHECKS = collections.Counter()


def require(ok, message, category="invariant"):
    CHECKS[category] += 1
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    path = str(Path(path).resolve())
    if path not in HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(2 ** 20), b""):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def pin(path, expected):
    require(sha(path) == expected, f"Hash mismatch: {path}", "hash")


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def pixel_signature(view):
    fields = ("output_index", "source_native_frame", "source_native_timestamp_us",
              "source_frame_path", "source_frame_sha256", "pixel_operation", "image_available")
    return [tuple(f[k] for k in fields) for f in view["ordered_frame_map"]]


def complete_interval(frame_map, start, end):
    """Return only the WHOLE native interval when every pixel/order survives."""
    by_source = {x["source_native_frame"]: x for x in frame_map}
    frames = list(range(start, end + 1))
    if not all(f in by_source for f in frames):
        return None, "not_all_native_frames_retained"
    selected = [by_source[f] for f in frames]
    if not all(x["image_available"] and x["pixel_operation"] == "copy" for x in selected):
        return None, "contains_missing_or_changed_pixels"
    oi = [x["output_index"] for x in selected]
    if oi != list(range(oi[0], oi[0] + len(oi))):
        return None, "not_contiguous_in_output"
    return selected, None


def binding(f):
    return {"source_frame": f["source_native_frame"], "output_index": f["output_index"],
            "source_native_timestamp_us": f["source_native_timestamp_us"],
            "view_timestamp_us": f["view_timestamp_us"], "source_frame_path": f["source_frame_path"],
            "source_frame_sha256": f["source_frame_sha256"], "image_available": f["image_available"]}


def bind_point(point, view):
    oi = point["output_index"]
    fm = view["ordered_frame_map"]
    require(isinstance(oi, int) and 0 <= oi < len(fm), "Witness output index", "witness")
    f = fm[oi]
    require(point["source_frame"] == f["source_native_frame"], "Witness source/output mismatch", "witness")
    require(f["image_available"] and f["pixel_operation"] == "copy", "Point witness on unavailable pixels", "witness")
    for key, target in (("source_timestamp_us", "source_native_timestamp_us"),
                        ("source_native_timestamp_us", "source_native_timestamp_us"),
                        ("view_timestamp_us", "view_timestamp_us"),
                        ("source_frame_path", "source_frame_path"),
                        ("source_frame_sha256", "source_frame_sha256")):
        if key in point:
            require(point[key] == f[target], "Witness provenance/time mismatch: " + key, "witness")
    return {"original_witness": copy.deepcopy(point), **binding(f)}


def bind_motion(motion, view):
    sf, oi = motion["source_frames"], motion["output_indices"]
    require(len(sf) == len(oi) and len(sf) >= 2, "Motion witness size mismatch", "witness")
    require(len(set(oi)) == len(oi) and oi == sorted(oi), "Motion witness output order", "witness")
    fm = view["ordered_frame_map"]
    for source, output in zip(sf, oi):
        require(0 <= output < len(fm) and fm[output]["source_native_frame"] == source,
                "Motion witness source/output mismatch", "witness")
    label = motion.get("label")
    require(label in (0, 1, None), "Invalid motion label", "witness")
    explicit_label = "label" in motion
    # REVIEW_PARTIAL_1567 records observed motion witnesses, not extra numeric
    # interval targets. Validate their visibility without inventing a 1 label.
    visible_support = label is not None or not explicit_label
    if visible_support:
        require(sf == list(range(sf[0], sf[-1] + 1)), "Known motion crosses native seam/gap", "witness")
        require(oi == list(range(oi[0], oi[-1] + 1)), "Known motion crosses output gap", "witness")
        require(all(fm[x]["image_available"] and fm[x]["pixel_operation"] == "copy" for x in oi),
                "Known motion includes unavailable pixels", "witness")
        require(not any(oi[0] < seam <= oi[-1] for seam in view["seam_output_indices"]),
                "Known motion crosses seam", "witness")
    return {"original_witness": copy.deepcopy(motion), "source_frames": sf, "output_indices": oi,
            "native_timestamps_us": [fm[x]["source_native_timestamp_us"] for x in oi],
            "view_timestamps_us": [fm[x]["view_timestamp_us"] for x in oi], "label": label,
            "label_was_explicit": explicit_label,
            "visible_contiguous_witness_validated": visible_support,
            "numeric_interval_target_created_from_unlabelled_witness": False,
            "supervision_unit": "one reviewed interval; never dense frame targets"}


def native_review(target, view):
    seq = target["sequence_links"][view["sequence_link_index"]]
    by_source = {f["source_native_frame"]: f for f in view["ordered_frame_map"]}
    value = 1 if seq["visible_chain"] == "yes" else None
    interval = seq["motion_interval"]
    known = [m for m in target["motion_intervals"] if interval[0] <= m["start"] < m["end"] <= interval[1] and m["label"] == 1]
    # Several individually reviewed intervals may jointly witness the chain.
    # This checks coverage of adjacent edges; it does not create frame targets.
    witnessed_edges = {edge for m in known for edge in range(m["start"], m["end"])}
    require(witnessed_edges == set(range(interval[0], interval[1])),
            "Native chain interval lacks contiguous explicit motion support", "native")
    motion_witnesses = []
    for m in known:
        sf = list(range(m["start"], m["end"] + 1))
        motion_witnesses.append({"source_frames": sf, "output_indices": [by_source[f]["output_index"] for f in sf],
                                 "label": 1, "reason": m["reason"]})
    return {"view_id": view["view_id"], "query_id": view["query_id"],
            "physical_query": view["physical_query"], "pixel_view_key": view["pixel_view_key"],
            "reviewed_pages": copy.deepcopy(target["reviewed_pages"]),
            "reviewed_output_indices": list(range(len(view["ordered_frame_map"]))),
            "observed_process_evidence": value, "visible_chain": seq["visible_chain"],
            "evidence_reason": seq["reason"],
            "remaining_evidence_by_fragment": [{"original_sequence_link": copy.deepcopy(seq),
                "original_scope_variant": copy.deepcopy(view.get("original_scope_variant"))}],
            "setup_witnesses": [binding(by_source[f]) for f in seq["setup_frames"]],
            "motion_witnesses": motion_witnesses,
            "terminal_witnesses": [binding(by_source[f]) for f in seq["terminal_frames"]],
            "whole_word_correctness": None, "learner_error": None, "required_repetitions": None,
            "review_method": "Original directly reviewed native images; exact source-frame/hash/order binding, not new visual adjudication.",
            "source_sequence_link": copy.deepcopy(seq), "source_annotation": target["source_annotation"],
            "source_annotation_sha256": target["source_annotation_sha256"]}


def main():
    for path in (OUT, VERIFIED, REPORT):
        require(not path.exists(), f"Refusing to overwrite completed output: {path}")
    # Missing/incomplete agent files must stop before any output is written.
    for name in REVIEW_FILES:
        require((U / name).is_file(), f"Review not ready: {name}")
    manifest = read(U / "PARTIAL_VIEWS_MANIFEST.json")
    targets = read(U / "PROCESS_TARGETS.json")
    source_manifest = read(U / "PROCESS_REVIEW_MANIFEST.json")
    target_by = {r["id"]: r for r in targets["rows"]}
    source_by = {r["id"]: r for r in source_manifest["rows"]}
    views = manifest["views"]
    view_by = {v["view_id"]: v for v in views}
    require(len(views) == len(view_by) == 217 and len(target_by) == 45, "Unexpected frozen coverage")
    inputs = [U / "PARTIAL_VIEWS_MANIFEST.json", U / "PROCESS_TARGETS.json", U / "PROCESS_REVIEW_MANIFEST.json"]
    for document in (manifest, targets):
        for path, expected in document["pins"].items():
            pin(path, expected)
    reviews = {}
    for name in REVIEW_FILES:
        path = U / name
        doc = read(path)
        inputs.append(path)
        expected = 40 if "SENTINEL" in name else (54 if "1567" in name else (24 if "0739" in name else 27))
        require(len(doc["rows"]) == expected, f"Incomplete review: {name}", "review")
        for key in ("manifest_sha256", "source_manifest_sha256"):
            if key in doc:
                pin(U / "PARTIAL_VIEWS_MANIFEST.json", doc[key])
        if "input_manifest" in doc:
            pin(doc["input_manifest"]["path"], doc["input_manifest"]["sha256"])
        if "provenance" in doc and "manifest_sha256" in doc["provenance"]:
            pin(doc["provenance"]["manifest_path"], doc["provenance"]["manifest_sha256"])
        for row in doc["rows"]:
            require(row["view_id"] not in reviews, "Duplicate direct review", "review")
            reviews[row["view_id"]] = (row, path, sha(path))
    expected_ids = {v["view_id"] for v in views if v["construction_kind_audit_only"] != "native"}
    require(set(reviews) == expected_ids, "Missing/unexpected direct view review", "review")

    page_catalog = {}
    for v in views:
        for p in v["review_pages"]:
            page_catalog[p["path"]] = p
    for target in targets["rows"]:
        for p in target["reviewed_pages"]:
            page_catalog[p["path"]] = p
        pin(target["source_annotation"], target["source_annotation_sha256"])

    compiled = []
    exclusions = collections.Counter()
    relation_total = interval_total = 0
    for v in views:
        q = target_by[v["query_id"]]
        src = source_by[v["query_id"]]
        require(v["role"] == q["role"] == "fit" and 1 <= v["recorded_person_id"] <= 9,
                "Non-FIT or prohibited identity", "source")
        require(v["recorded_person_id"] == q["person"] == src["person"], "Identity mismatch", "source")
        require(v["physical_query"] == q["physical_query"] == src["physical_query"], "Query mismatch", "source")
        require(v["clip_id"] == q["clip_id"] == src["clip_id"], "Clip mismatch", "source")
        require(v["source_sha256"] == q["source_sha256"] == src["source_sha256"], "Source lineage mismatch", "source")
        pin(v["source_path"], v["source_sha256"])
        pin(v["original_review_path"], v["original_review_sha256"])
        native = {f["frame"]: f for f in src["frames"]}
        fm = v["ordered_frame_map"]
        require([f["output_index"] for f in fm] == list(range(len(fm))), "Output indices not dense", "mapping")
        require(len({f["source_native_frame"] for f in fm}) == len(fm), "Unexpected repeated source frame", "mapping")
        source_frames = [f["source_native_frame"] for f in fm]
        require(source_frames == sorted(source_frames), "Non-monotonic source order", "mapping")
        expected_seams = [i for i in range(1, len(fm)) if source_frames[i] != source_frames[i - 1] + 1]
        require(expected_seams == v["seam_output_indices"], "Seam map mismatch", "mapping")
        for f in fm:
            n = native[f["source_native_frame"]]
            require((f["source_native_timestamp_us"], f["source_frame_path"], f["source_frame_sha256"])
                    == (n["timestamp_us"], n["path"], n["sha256"]), "Native frame/time/hash binding mismatch", "mapping")
            require(f["pixel_operation"] in ("copy", "blackout") and f["image_available"] == (f["pixel_operation"] == "copy"),
                    "Invalid pixel availability", "mapping")
            pin(n["path"], n["sha256"])
        view_times = [f["view_timestamp_us"] for f in fm]
        require(view_times[0] == 0 and all(a < b for a, b in zip(view_times, view_times[1:])), "Output times not increasing", "mapping")
        # Independent exact rational accumulation: preserve native time steps,
        # use the frozen native median period only across a removed seam.
        source_times = [x["timestamp_us"] for x in src["frames"]]
        period = Fraction(str(statistics.median(b - a for a, b in zip(source_times, source_times[1:]))))
        speed = Fraction(str(v["speed"]))
        expected_time = Fraction(0)
        for i in range(1, len(fm)):
            delta = period if i in expected_seams else Fraction(fm[i]["source_native_timestamp_us"] - fm[i - 1]["source_native_timestamp_us"])
            expected_time += delta / speed
            require(abs(view_times[i] - round(expected_time)) <= 1,
                    "Output clock inconsistent with frozen source time/speed", "mapping")
        covered = []
        for p in v["review_pages"]:
            pin(p["path"], p["sha256"])
            covered.extend(p["output_indices"])
            require([fm[i]["source_native_frame"] for i in p["output_indices"]] == p["source_native_frames"], "Sheet frame binding", "page")
        require(covered == list(range(len(fm))), "Artifact pages do not cover all outputs", "page")

        is_native = v["construction_kind_audit_only"] == "native"
        if is_native:
            review = native_review(q, v)
            review_path, review_sha = U / "PROCESS_TARGETS.json", sha(U / "PROCESS_TARGETS.json")
        else:
            review, review_path, review_sha = reviews[v["view_id"]]
        require(review["query_id"] == v["query_id"] and review["physical_query"] == v["physical_query"]
                and review["pixel_view_key"] == v["pixel_view_key"], "Review view/query binding", "review")
        require(review["reviewed_output_indices"] == list(range(len(fm))), "Incomplete direct output review", "review")
        value = review["observed_process_evidence"]
        require(value in (0, 1, None), "Invalid process evidence", "review")
        require(review["visible_chain"] in ("yes", "partial", "uncertain"), "Invalid chain value", "review")
        require(value != 1 or review["visible_chain"] == "yes", "Positive without observed chain", "review")
        for field in ("whole_word_correctness", "learner_error", "required_repetitions"):
            require(review[field] is None, f"Prohibited semantic truth: {field}", "scope")

        viewed_coverage = []
        for p in review["reviewed_pages"]:
            pin(p["path"], p["sha256"])
            require(p["path"] in page_catalog, "Uncatalogued viewed page", "page")
            declared = page_catalog[p["path"]]
            require(p["sha256"] == declared["sha256"], "Reviewed page/catalog hash mismatch", "page")
            viewed_coverage.extend(declared.get("source_native_frames", declared.get("frames", [])))
        require(viewed_coverage == source_frames, "Actual viewed pages do not cover same ordered native pixels", "page")
        own_paths = {p["path"] for p in v["review_pages"]}
        reviewed_paths = {p["path"] for p in review["reviewed_pages"]}
        reuse = None
        if own_paths != reviewed_paths:
            if is_native:
                require(all(f["image_available"] for f in fm), "Native reuse with missing pixels", "page")
                reuse = {"kind": "native_original_review_pages", "same_native_pixels_verified": True}
            else:
                require(v["construction_kind_audit_only"] == "speed_1_25", "Unsupported page reuse", "page")
                peer = view_by[v["query_id"] + "__speed_0_8"]
                require(pixel_signature(peer) == pixel_signature(v), "Speed content is not identical", "page")
                require(reviewed_paths == {p["path"] for p in peer["review_pages"]}, "Wrong reused speed pages", "page")
                own = review["artifact_pages_verified_not_opened"]
                require({p["path"]: p["sha256"] for p in own} == {p["path"]: p["sha256"] for p in v["review_pages"]},
                        "Unviewed own speed artifact provenance missing", "page")
                reuse = {"kind": "identical_native_pixel_order_speed_variant", "peer_view_id": peer["view_id"],
                         "actual_retimed_video_or_speed_performance_review": False}
        for extra in review.get("additional_native_images_reviewed", []):
            if isinstance(extra, dict):
                ep = extra.get("path", extra.get("source_frame_path"))
                eh = extra.get("sha256", extra.get("source_frame_sha256"))
                if ep and eh:
                    pin(ep, eh)

        points = {key: [bind_point(p, v) for p in review[key]] for key in ("setup_witnesses", "terminal_witnesses")}
        direct_motion = [bind_motion(m, v) for m in review["motion_witnesses"]]
        require(value != 1 or any(m["visible_contiguous_witness_validated"] and
                (m["label"] == 1 or not m["label_was_explicit"]) for m in direct_motion),
                "Positive process lacks visible motion witness", "review")
        for rel in review.get("relations", []):
            bind_point(rel, v)

        by_source = {f["source_native_frame"]: f for f in fm}
        relations, excluded_relations = [], []
        for i, rel in enumerate(q["relations"]):
            f = by_source.get(rel["frame"])
            reason = "not_retained" if f is None else ("missing_or_changed_pixels" if not f["image_available"] or f["pixel_operation"] != "copy" else None)
            if reason:
                excluded_relations.append({"source_relation_index": i, "source_frame": rel["frame"], "reason": reason})
                exclusions["relation_" + reason] += 1
                continue
            require(rel["timestamp_us"] == f["source_native_timestamp_us"], "Sparse target clock mismatch", "target")
            require(rel["input_image"]["sha256"] == f["source_frame_sha256"], "Sparse target pixel mismatch", "target")
            relations.append({"source_relation_index": i, "source_annotation": copy.deepcopy(rel), **binding(f),
                              "setup_relation": rel["setup_relation"], "terminal_relation": rel["terminal_relation"]})
        intervals, excluded_intervals = [], []
        for i, m in enumerate(q["motion_intervals"]):
            selected, reason = complete_interval(fm, m["start"], m["end"])
            if reason:
                excluded_intervals.append({"source_motion_interval_index": i, "source_interval": [m["start"], m["end"]],
                                           "reason": reason, "inherited_target": None})
                exclusions["motion_" + reason] += 1
                continue
            oi = [f["output_index"] for f in selected]
            require(not any(oi[0] < seam <= oi[-1] for seam in v["seam_output_indices"]), "Inherited interval crosses seam", "target")
            intervals.append({"source_motion_interval_index": i, "source_annotation": copy.deepcopy(m), "label": m["label"],
                              "source_frames": [f["source_native_frame"] for f in selected], "output_indices": oi,
                              "native_timestamps_us": [f["source_native_timestamp_us"] for f in selected],
                              "view_timestamps_us": [f["view_timestamp_us"] for f in selected],
                              "supervision_unit": "one fully retained reviewed interval; not frame-wise labels"})
        relation_total += len(relations)
        interval_total += len(intervals)
        compiled.append({"view_id": v["view_id"], "query_id": v["query_id"], "clip_id": v["clip_id"],
            "word_id": q["word_id"], "recorded_person_id": v["recorded_person_id"], "role": "fit",
            "grouping_key": "recorded_person_" + str(v["recorded_person_id"]),
            "physical_query": v["physical_query"], "predicate_definitions": copy.deepcopy(q["predicate_definitions"]),
            "pixel_view_key": v["pixel_view_key"], "query_review_key": v["query_review_key"],
            "construction_kind_audit_only": v["construction_kind_audit_only"], "source_path": v["source_path"],
            "source_sha256": v["source_sha256"], "ordered_frame_map": copy.deepcopy(fm),
            "seam_output_indices": v["seam_output_indices"], "original_scope_variant": copy.deepcopy(v.get("original_scope_variant")),
            "artifact_pages": copy.deepcopy(v["review_pages"]), "direct_review": copy.deepcopy(review),
            "review_source": {"path": str(review_path), "sha256": review_sha}, "page_content_reuse": reuse,
            "observed_process_evidence": value, "visible_chain": review["visible_chain"], "evidence_reason": review["evidence_reason"],
            "bound_setup_witnesses": points["setup_witnesses"], "bound_terminal_witnesses": points["terminal_witnesses"],
            "bound_direct_motion_witnesses": direct_motion,
            "sparse_relation_targets": relations, "fully_retained_motion_interval_targets": intervals,
            "excluded_source_relations": excluded_relations, "excluded_source_motion_intervals": excluded_intervals,
            "whole_word_correctness": None, "learner_error": None, "required_repetitions": None,
            "unlisted_is_unknown": True, "dense_frame_targets_created": False})

    native_counts = collections.Counter(str(r["observed_process_evidence"]) for r in compiled if r["construction_kind_audit_only"] == "native")
    require(native_counts == {"1": 44, "None": 1}, "Original native physical-chain scope changed", "scope")
    identity08 = next(r for r in compiled if "WORD0739_REAL08" in r["view_id"])
    require(identity08["observed_process_evidence"] is None and identity08["original_scope_variant"] is not None,
            "ID08 physical variation was lost or forced negative", "scope")
    counts = {"views": len(compiled), "source_queries": len(target_by), "source_clips": len({r["clip_id"] for r in compiled}),
              "sparse_relation_records": relation_total, "fully_retained_motion_intervals": interval_total,
              "observed_process_evidence": dict(collections.Counter(str(r["observed_process_evidence"]) for r in compiled)),
              "by_construction_audit_only": {k: dict(collections.Counter(str(r["observed_process_evidence"]) for r in compiled if r["construction_kind_audit_only"] == k)) for k in sorted(manifest["counts"])},
              "excluded_source_targets": dict(exclusions)}
    pins = {str(path.resolve()): sha(path) for path in inputs + [Path(__file__).resolve()]}
    result = {"at": datetime.now(timezone.utc).isoformat(), "status": "COMPILED_DIRECT_PROCESS_VIEW_TARGETS_NOT_MODEL_EVALUATION",
        "pins": pins, "rows": compiled, "counts": counts,
        "supervision_scope": "Physical query process evidence somewhere in the shown view; not linguistic sufficiency, full word completion or learner error.",
        "source_targets_unchanged": True, "review_records_embedded_unchanged": True,
        "unknown_and_variant_fields_preserved": True, "partly_retained_intervals_inherited": False,
        "schema_compatibility_note": "1567 direct review motion witnesses have source/output indices and scope but no numeric interval label. They are preserved as visible witness support; no numeric interval target is invented. Native sequence links may cite several adjacent originally labelled intervals, which remain separate.",
        "seam_or_blackout_known_motion_inherited": False, "dense_labels_generated": False,
        "dependent_views_are_not_new_people": True, "actual_speed_performance_evaluated": False,
        "independent_human_expert": False, "new_inference": 0, "new_training": 0, "whole_goal_complete": False}
    # Confirm source files did not change during the compilation.
    for path, expected in pins.items():
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        require(h == expected, "Input changed during compilation: " + path, "final_pin")
    write_new(OUT, result)
    verified = {"at": datetime.now(timezone.utc).isoformat(), "status": "COMPILER_SELF_CHECK_PASS_NOT_INDEPENDENT_SEMANTIC_VALIDATION",
        "output_path": str(OUT), "output_sha256": sha(OUT), "input_pins": pins,
        "checks": dict(CHECKS), "checks_total": sum(CHECKS.values()), "unique_hashed_files": len(HASHES),
        "counts": counts, "page_reuse_scope": "Native original sheets and speed1.25 peer sheets verified by exact source pixel/order mapping. Own artifact hashes also verified; no false claim that all artifact sheets were directly viewed.",
        "source_video_hashes_verified": len({v["source_path"] for v in views}),
        "no_semantic_or_performance_validation_claim": True}
    write_new(VERIFIED, verified)
    text = ("공개 과정 관측 정답 연결 결과\n\n"
            f"36개 공개 영상의 45개 질문에서 만든 {len(compiled)}개 비교 자료를 연결했습니다.\n"
            "자료를 자른 방법으로 정답을 만들지 않고 직접 검토 기록을 보존했습니다.\n"
            f"과정 관측 값(1/0/미확인): {counts['observed_process_evidence']}\n"
            f"실제로 남아 있는 순간 관계 {relation_total}개, 통째로 연속 유지된 움직임 구간 {interval_total}개를 연결했습니다.\n"
            "일부만 남은 원본 움직임 구간은 이어받지 않았습니다. 가려진 화면과 편집 연결점에는 원본 정답을 붙이지 않았습니다.\n"
            "0739 ID08의 목·어깨 변이는 그대로 보존하고 좁은 몸통 도착 기준은 미확인으로 남겼습니다.\n"
            "다른 반복 과정이 남아 있으면 그 실제 관측을 유지했습니다. 단어 전체의 정오나 필요한 반복 횟수는 정하지 않았습니다.\n"
            "속도 자료는 같은 이미지 순서와 바뀐 시간표에 대한 관측 연결입니다. 실제 속도별 모델 성능 검증은 아직 아닙니다.\n"
            "이 검산은 파일·시간·해시·연결 규칙의 자체 검산이며 독립 수어 전문가의 정답 검증이 아닙니다.\n"
            "새 학습·추론·영상 추출·운영 모델 변경은 없습니다. 다음 단계는 별도로 고정한 학습 비교입니다.\n")
    with REPORT.open("x", encoding="utf-8") as f:
        f.write(text)
    print(json.dumps({"status": verified["status"], "counts": counts, "output_sha256": verified["output_sha256"],
                      "checks": verified["checks_total"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = U / ("PROCESS_VIEW_COMPILE_FAILURE_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".txt")
        with failure.open("x", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise
