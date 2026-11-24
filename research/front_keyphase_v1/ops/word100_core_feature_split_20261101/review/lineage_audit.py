"""Read-only, score-blind public source audit; writes only this review's audit files."""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import json
import pathlib
import re
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
OPS = HERE.parents[1]
BODY = OPS / "word100_body_boundary_20261017"
PUBLIC_ROOT = pathlib.Path(r"W:\WordsInSilence\data\AIHub_103")
PINS = {}


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    p = pathlib.Path(path)
    assert p.is_relative_to(OPS), p
    raw = p.read_bytes()
    PINS[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def checked_asset(path, expected=None):
    p = pathlib.Path(path)
    assert p.is_relative_to(OPS) or p.is_relative_to(PUBLIC_ROOT), p
    exists = p.is_file()
    actual = sha(p) if exists else None
    return {"path": str(p), "exists": exists, "sha256": actual,
            "expected_sha256": expected, "matches_pin": exists and (not expected or actual == expected)}


def main():
    prep = read(BODY / "PREP_CONTEXT.json")
    train = read(BODY / "training_CONTEXT.json")
    prior = read(OPS / "word100_visible_temporal_targets_20261101/SELECTION.json")
    historical = [
        "word100_visible_temporal_targets_20261101/SELECTION.json",
        "word100_stage_feedback_cycle_20261031/SELECTION.json",
        "word100_feature_support_streaming_20261031/fresh/SELECTION.json",
        "word100_signrep_temporal_head_20261029/INPUTS.json",
        "word100_shared_feature_bundle_20261027/FROZEN.json",
        "word100_semantic_episode_pilot_20261031/AUTHORIZATION.json",
        "word100_description_evidence_feedback_20261101/AUTHORIZATION.json",
    ]
    history = []
    known_ids = set()
    for name in historical:
        data = read(OPS / name)
        ids = set(re.findall(r"NIA_SL_WORD\d+_REAL\d+_[A-Z]", json.dumps(data)))
        known_ids |= ids
        history.append({"path": str(OPS / name), "sha256": PINS[str(OPS / name)],
                        "clip_ids": sorted(ids), "clip_count": len(ids)})
    known_words = {re.search(r"WORD\d+", cid).group() for cid in known_ids}
    prep_rows = prep["ordered_rows"]
    prep_by_id = {r["clip_id"]: r for r in prep_rows}
    train_ids = {r["clip_id"] for r in train["ordered_rows"] if r["role"] == "train_candidate"}
    records = {}
    source_groups = collections.defaultdict(list)
    for r in prep_rows:
        rec = read(BODY / "data/records" / (r["clip_id"] + ".json"))
        records[r["clip_id"]] = rec
        source_groups[rec["source"]["sha256"]].append(r["clip_id"])
    available_words = sorted({r["word_code"] for r in prep_rows if r["role"] == "train_candidate"} - known_words)
    support_words, eval_words = available_words[:8], available_words[8:16]
    assert len(support_words) == len(eval_words) == 8
    requested = [(w, p, "fit" if p in (1, 2, 3) else "development")
                 for w in support_words for p in (1, 2, 3, 4)]
    requested += [(w, p, "new_feature_expression_evaluation") for w in eval_words for p in (7, 8)]
    archive_handles = {}
    candidates = []
    for word, person, role in requested:
        cid = f"NIA_SL_{word}_REAL{person:02d}_F"
        r, rec = prep_by_id[cid], records[cid]
        assert r["role"] == "train_candidate" and cid in train_ids
        source = dict(rec["source"])
        ap = pathlib.Path(source["archive"])
        assert ap.is_relative_to(PUBLIC_ROOT)
        archive_available = ap.is_file()
        assert archive_available, ap
        if str(ap) not in archive_handles:
            archive_handles[str(ap)] = zipfile.ZipFile(ap, "r")
        zf = archive_handles[str(ap)]
        zi = zf.getinfo(source["member"])
        h = hashlib.sha256()
        with zf.open(zi, "r") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        actual_sha = h.hexdigest()
        assert actual_sha == source["sha256"]
        assert zi.CRC == source["crc32"] and zi.file_size == source["bytes"]
        obs = rec["views"]["native"]["observation"]
        obs_asset = checked_asset(obs, rec["outputs"][obs])
        assert obs_asset["matches_pin"]
        rp = BODY / "data/records" / f"{cid}.json"
        retained = r.get("retained_video")
        retained_asset = checked_asset(retained, actual_sha) if retained else None
        aliases = source_groups[source["sha256"]]
        assert not (set(aliases) & known_ids)
        candidates.append({
            "clip_id": cid, "word_id": word, "label": r["canonical_label"], "person": person,
            "role": role, "intervals": r["intervals"], "record": str(rp),
            "record_sha256": PINS[str(rp)], "observation": obs,
            "observation_sha256": obs_asset["sha256"], "archive_source": source,
            "archive_available": archive_available, "archive_member_crc_bytes_sha_verified": True,
            "source_sha256": actual_sha, "source_sha_duplicate_ids": aliases,
            "retained_video": retained, "retained_asset": retained_asset,
            "frames": rec["frames"], "box": rec["calibration"]["box_source_pixels"],
            "known_feature_history_matches": [], "prior_tcn_train": True,
            "signrep_cache_in_audited_manifests": None,
            "label_support": "UNKNOWN_BEFORE_DIRECT_OBSERVATION",
        })
    for zf in archive_handles.values():
        zf.close()

    cache_names = ["word100_stage_feedback_cycle_20261031", "word100_feature_support_streaming_20261031/fresh"]
    cached = {}
    for name in cache_names:
        data = read(OPS / name / "INPUTS.json")
        visual = read(OPS / name / "VISUAL_INDEX.json")
        visuals = {r["alias"]: r for r in visual["rows"]}
        for r in data["rows"]:
            v = visuals[r["alias"]]
            files = [f["file"] for f in v["frames"]]
            asset = checked_asset(r["input"], r["input_sha256"])
            assert asset["matches_pin"]
            cached[r["clip_id"]] = {
                "clip_id": r["clip_id"], "word_id": r["word_id"], "person": r["person"],
                "historical_role": r["role"], "input": r["input"], "input_sha256": asset["sha256"],
                "observation": r["observation"], "source": r["source"],
                "source_sha256": r["source_sha256"], "visual_index": str(OPS / name / "VISUAL_INDEX.json"),
                "visual_alias": r["alias"], "frame_cache_count": len(files),
                "frame_cache_present": sum(pathlib.Path(f).is_file() for f in files),
                "sequence": v["sequence"], "states": v["states"], "fresh_evaluation_eligible": False,
            }
    old_fitdev = [r for r in prior["rows"] if r["role"] in ("fit", "development")]
    assert len(old_fitdev) == 48
    old_rows = []
    for r in old_fitdev:
        old = {k: r[k] for k in ("clip_id", "word_id", "label", "person", "role", "intervals", "record",
                                 "record_sha256", "observation", "observation_sha256", "archive_source", "box",
                                 "frames", "input", "input_sha256", "centers", "alias")}
        old["history"] = "Prior feature-head fit/development; retained same role, never fresh evaluation"
        old_rows.append(old)
    selected = old_rows + candidates
    selected_ids = {r["clip_id"] for r in selected}
    assert len(selected_ids) == 96
    assert not (set(eval_words) & {r["word_id"] for r in selected if r["role"] in ("fit", "development")})
    assert not ({r["person"] for r in selected if r["role"] == "fit"} & {7, 8})
    selected_sha = [r["archive_source"]["sha256"] for r in selected]
    assert len(selected_sha) == len(set(selected_sha))

    result = {
        "status": "PUBLIC_COHORT_SOURCE_READY_LABEL_SUPPORT_NOT_YET_READY",
        "at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
        "reviewer": "Codex lineage audit agent; independent code/metadata review, not independent human KSL truth",
        "selection_fixed_before_direct_new_annotations": True,
        "selection_rule": "Exclude every word in seven named historical feature cohorts. Lexicographic first8 remaining words x recordedpeople1/2/3 fit,4 development; next8 x people7/8 evaluation. Retain only prior48fit/dev with unchanged roles. No outcome selection or replacement.",
        "cohort_size_reason": "Bounded new48-clip preparation budget; NOT a statistical sufficiency or rare-feature guarantee.",
        "source_scope": "Existing public AIHub103 only; ZIPs read-only; no extraction, inference, scores, training, download, private/mentor access.",
        "history_scope_limit": "Absence checked against seven named feature-head/evidence cohorts only. This is NOT an exhaustive entire-project unseen proof. All selected sources have prior TCN TRAIN/native observation/reference-bank lineage.",
        "known_feature_clip_count": len(known_ids), "known_feature_words": sorted(known_words),
        "remaining_public_train_words": available_words, "new_support_words": support_words, "new_evaluation_words": eval_words,
        "counts": {"public_train_candidates": len(train_ids), "public_development_only_excluded": 300,
                   "existing_fitdev": 48, "new_fit": 24, "new_development": 8, "new_evaluation": 16,
                   "selected_total": 96, "roles": dict(collections.Counter(r["role"] for r in selected)),
                   "new_archive_sha_verified": len(candidates), "new_observation_sha_verified": len(candidates),
                   "existing_cached_feature_clips": len(cached)},
        "duplicate_source_sha_groups_all1199": {k: v for k, v in source_groups.items() if len(v) > 1},
        "new_candidates": candidates, "rows": selected, "existing_reusable_caches": list(cached.values()),
        "historical_manifests": history,
        "consumed8": [r["clip_id"] for r in prior["rows"] if r["role"] == "new_feature_expression_evaluation"],
        "support_gate": {"status": "PENDING_LABELS", "requirements": [
            "Keep prior feature definitions and readiness thresholds; no inferred labels from word/guide.",
            "Count feature positive/negative/unknown by independent clip, expression, person and split.",
            "Audit rare open-hand positives, brief transitions and bilateral crossings explicitly.",
            "More clips, neighboring windows and seeds do not guarantee more independent positive examples.",
            "Preserve official intervals only as source metadata, never educational correctness/phase truth.",
            "No score-based replacement if held-out positive support is insufficient; report unsupported scope."]},
        "new_eval_scores_read": False, "new_visuals_read": False,
        "script_sha256": sha(__file__), "read_source_pins": PINS,
    }
    dest = HERE / "LINEAGE_AUDIT.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ("공개 자료 계보 감사 완료. 새 점수·새 영상·개인자료 열람 없음.\n"
            f"후보96영상: 기존fit36/dev12 유지 + 신규fit24/dev8/미사용표현평가16. 알려진 특징시험 원본{len(known_ids)}개/20표현은 새 평가에서 제외.\n"
            "신규48 원본 ZIP member CRC/크기/SHA와 기존 native observation SHA 모두 검증. 7개 명시적 특징계보에서 원본ID·표현·동일SHA 중복 없음. 전체프로젝트 미사용이 아니며 모든 후보는 과거TCN TRAIN 계보.\n"
            "신규지원8표현: " + ", ".join(support_words) + " (사람1/2/3 fit,4 dev).\n"
            "신규평가8표현: " + ", ".join(eval_words) + " (사람7/8).\n"
            "기존112개 SignRep 입력캐시 해시 검증; 이미 소비된 held/fresh 집합은 개발·보존진단용만 가능. 이번선정은 그중기존48만유지.\n"
            "신규48의 native 관절은 재사용 가능. 이번에 감사한 SignRep 입력 명세에는 신규48이 없으므로 별도 고정본 특징준비 필요. 기존 retained_video가 있으면 해당SHA 확인 후 재사용 가능.\n"
            "희소손모양·전이·교차 정답지원은 아직 미확인. 8+8표현 제한은 준비범위이며 충분성 기준이 아니다. 원본전체와 공식구간 메타를 보존하고 같은3중앙창 직접주석 후 지원판정. 소모8은 평가에서 제외.\n")
    (HERE / "LINEAGE_AUDIT_KO.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "audit": str(dest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
