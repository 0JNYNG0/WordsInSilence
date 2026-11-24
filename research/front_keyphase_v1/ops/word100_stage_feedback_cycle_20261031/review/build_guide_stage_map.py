"""Reconcile saved guide and RGB-review evidence without creating new labels.

Writes only this review directory. Does not decode videos, infer, train or query
the network. Frozen source artifacts and public cached images are read-only.
"""
from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
OPS = ROOT / "research/front_keyphase_v1/ops"
GUIDE = OPS / "word100_guide_video_feedback_20261028"
TOKEN = OPS / "word100_source_token_alignment_20261027"
TEMPORAL = OPS / "word100_signrep_temporal_head_20261029"
SHARED = OPS / "word100_shared_feature_bundle_20261027"
sources: dict[str, dict] = {}
images: dict[str, dict] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source(path: Path) -> dict:
    key = relative(path)
    if key not in sources:
        sources[key] = {"path": key, "sha256": digest(path), "bytes": path.stat().st_size}
    return {"path": key, "sha256": sources[key]["sha256"]}


def read(path: Path):
    source(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evidence_image(path: Path, recorded_rgb_sha256: str | None = None) -> str:
    key = relative(path)
    if key not in images:
        images[key] = {"path": key, "file_sha256": digest(path), "bytes": path.stat().st_size}
    if recorded_rgb_sha256:
        images[key]["recorded_rgb_sha256"] = recorded_rgb_sha256
        images[key]["rgb_hash_scope"] = "Copied source decoded-RGB hash; this reconciliation verifies file hash only, without re-decoding."
    return key


def ref(path: Path, pointer: str) -> dict:
    return {**source(path), "json_pointer": pointer}


cards = read(GUIDE / "GUIDE_CARDS.json")
links = read(GUIDE / "GUIDE_CLAIM_LINKS.json")
forms = read(TOKEN / "SOURCE_FORM_LINKS.json")
profiles = read(TOKEN / "SOURCE_PROFILES.json")
tokens = read(TOKEN / "TOKEN_OBSERVATIONS.json")
decisions = read(TOKEN / "review/SOURCE_TOKEN_DECISIONS.json")
coarse = read(TOKEN / "review/OBSERVATIONS_STAGE1.json")
motion_review = read(TOKEN / "review/MOTION_SCOPE_REVIEW.json")
labels = read(TEMPORAL / "LABELS.json")
label_review = read(TEMPORAL / "LABEL_REVIEW.json")
head_contract = read(TEMPORAL / "FROZEN.json")
draft = read(SHARED / "DRAFT_LABELS.json")
source(OPS / "word100_feedback_next_plan_20261031/PLAN_KO.txt")

static_lookup = {(r["alias"], r["frame_id"]): (i, r) for i, r in enumerate(labels["static"])}
draft_lookup = {(r["alias"], r["frame_id"]): r for r in draft["rows"]}
form_lookup = {r["word_id"]: (i, r) for i, r in enumerate(forms["links"])}
coarse_lookup = {r["alias"]: (i, r) for i, r in enumerate(coarse["observations"])}

NOTES = {
    "WORD0003": {
        "physical_summary_ko": "두 손의 검지 펴짐이 옛 시점 초안에서 지지된다. 시간창 검토는 오른 검지 형태 유지와 양손 교대 움직임을 기록했다.",
        "unverified_ko": ["표제어와 사전11290의 전체 표현형 일치", "손등이 밖으로 향하는 정밀 방향", "엇갈린 회전의 전체 경로·언어적 필수 순서", "왼 검지 특징은 현재5항목 판정부 밖이며 10월29일 정적 재검토 대상도 아님"],
        "disabled_ko": ["두 손의 특정 시작 배치·회전 방향을 모든 수행의 필수 규칙으로 적용", "일부 검지 표본만으로 전체 사전 동작 충족 판정"]},
    "WORD0071": {
        "physical_summary_ko": "짧은 시간창의 검지/펼친 손 변화가 RGB로 재검토되었다. 사전7145의 전체 표현형과 뜻을 대조한 확정 기록은 없다.",
        "unverified_ko": ["화상 표제어의 정확한 뜻과 사전7145 표현형 대응", "두 손의 손등·손끝 방향", "상하 교대 움직임의 전체 횟수·단계"],
        "disabled_ko": ["사전의 두 번을 필수 횟수 정답으로 복사", "화면상 겹침을 실제 접촉으로 해석"]},
    "WORD0500": {
        "physical_summary_ko": "오른 검지 펴짐의 부분 연결이 있다. 짧은 시간창 검토에는 얼굴 쪽에서 다른 손 쪽으로 이동하면서 검지 형태가 유지되는 관찰이 있다.",
        "unverified_ko": ["목 중앙이라는 정밀 신체 지점", "오른 검지 끝 바닥과 왼 검지 바닥의 실제 접촉", "사전6280과 전체 표현형·단계 경계 일치"],
        "disabled_ko": ["목/검지 접촉을 필수 단계 정답으로 적용", "얼굴 쪽이라는 넓은 관찰을 목 중앙 접촉으로 세분화"]},
    "WORD0678": {
        "physical_summary_ko": "손이 측면으로 보이는 구간과 이후 손가락 형태 변화가 시간창에서 검토되었다. 사전10796의 두 손가락/눈 스침은 별도 검증되지 않았다.",
        "unverified_ko": ["실명 표제어의 뜻과 사전 표현형 일치", "검지·중지 두 손가락의 전체 펼침", "두 눈을 스치는 실제 접촉과 하향 경로"],
        "disabled_ko": ["검지만 펴짐 판정부를 두 손가락/눈 스침 검출기로 해석", "사전 문구로 보이지 않는 손가락·접촉 정답 채우기"]},
    "WORD0739": {
        "physical_summary_ko": "짧은 시간창에 펼친 손에서 접힌 손가락으로의 변화가 검토되어 있다. 중간 모양은 확실한 형태로 채우지 않았다.",
        "unverified_ko": ["사전12209의 주먹과 영상 전체 표현형 일치", "오른팔 옆 들기·내리기의 정확한 시작/끝과 필수성", "완전한 주먹의 엄지·손가락 구성"],
        "disabled_ko": ["open_four=0을 주먹이라고 단정", "사전의 주먹 상태를 준비·변화 구간 모두에 적용"]},
    "WORD0949": {
        "physical_summary_ko": "세 영상의 표본에서 한 손 검지가 얼굴 옆에 왔다가 밖으로 향한다. 사전8171의 두 손 시작과 맞지 않는다.",
        "unverified_ko": ["올바른 뜻과 공식 표현형 연결", "얼굴의 정확 접촉 지점", "표본 사이 짧은 움직임의 존재/부재"],
        "disabled_ko": ["거절된8171의 두손 시작·1/5지 맞대기·포물선 요구 전체", "사전 불일치를 수행자 수어 오류로 해석"]},
    "WORD1133": {
        "physical_summary_ko": "시간창 검토는 평평한 손의 팔뚝 쪽 움직임을 기록한다. 옆면으로 가려지는 네 손가락 펴짐은 최신 정적 검토에서 보류되었다.",
        "unverified_ko": ["9543/41/누리 후보 중 이 영상의 정확한 표현형", "손바닥과 왼 팔등의 실제 접촉", "엄지 펴기 후보와 두 번 반복 후보의 적용 여부"],
        "disabled_ko": ["서로 다른 사전 후보의 팔등 스침·엄지 움직임을 합쳐 필수 경로 생성", "가려진 펼침을 출처 설명으로 복원"]},
    "WORD1139": {
        "physical_summary_ko": "왼 펼친 손과 오른 검지 같은 모양의 접근은 큰 형태상 잠정 일치한다. 정확한 R_index_only는 손 사이 가림/단축 때문에 보류된 시점이 있다.",
        "unverified_ko": ["검지 옆면 대 끝의 실제 접촉", "세부 손 방향과 가려진 관절", "같은 언어적 변이라는 최종 확인"],
        "disabled_ko": ["큰 관찰의 index-like를 엄격한 index_only=1로 승격", "화면상 양손 겹침을 사전의 옆면 접촉으로 해석"]},
    "WORD1290": {
        "physical_summary_ko": "두 펼친 손과 위아래 접근의 큰 구조만2372와 잠정 일치한다.12468은 검사·감찰 동음이의/다른 형태로 제외한다.",
        "unverified_ko": ["아래 왼손의 손바닥/손등 방향", "오른 새끼손가락 옆면의 실제 접촉", "정확히 두 번의 접촉과 언어적으로 허용되는 횟수·높이"],
        "disabled_ko": ["아래손 손바닥위처럼 보였다는 초기 관찰을 확정적인 사전 방향 오류로 주장", "손등 아래방향·정확두번·접촉을 필수 교정 규칙으로 적용", "12468의 눈 방향·회전을 현재 영상에 적용"]},
    "WORD1567": {
        "physical_summary_ko": "시간창 검토에는 엄지 위/다른 손가락 접힘과 교대 수직 움직임이 기록되어 있다. 엄지 자체는 현재5항목 판정부의 학습 목표가 아니다.",
        "unverified_ko": ["사전6521의 전체 표현형 확인", "엄지의 정밀 상태와 양손 이동 경계", "교대 횟수 및 필수 순서"],
        "disabled_ko": ["index_only=0 또는 open_four=0을 엄지 펴짐 정답으로 바꾸기", "교차·겹침 때 손 정체성이나 접촉 추정"]},
    "WORD2523": {
        "physical_summary_ko": "비스듬한 평평한/삼각형 손 배치가 시간창에서 검토되었다. 네 손가락 각각의 펼침은 단축과 측면 때문에 일부 최신 정적 라벨에서 보류되었다.",
        "unverified_ko": ["23905/5990 중 정확 표현형", "두 손끝의 실제 접촉", "내린 뒤 각각 손끝을 모으는 단계의 적용·필수성"],
        "disabled_ko": ["두 후보 설명의 차이를 하나의 필수 단계로 합치기", "삼각형 외형을 개별 네 손가락 펼침이나 손끝 접촉으로 승격"]},
    "WORD2551": {
        "physical_summary_ko": "얼굴/이마 앞 오른손의 모음·굽힘과 펼침이 표본과 시간창에서 보인다.6163과 부분 연결이며 별1/별2 변이는 미확정이다.",
        "unverified_ko": ["별1/별2의 정확한 변이 ID", "왼쪽에서 오른쪽까지 전체 경로", "얼굴 앞 대 머리 위 높이의 언어적 허용 범위", "전체 반복 횟수", "기존V11 표본 범위 밖 종료 동작"],
        "disabled_ko": ["머리 위 높이·가로 이동을 필수 규칙으로 강제", "사전 문구로 종료·반복수·중간 손모양 생성", "모음/굽힘을 모두 주먹으로 해석"]},
}

COMMON_FEATURES = {
    "R_index_only": {"hand": "R", "definition_ko": "오른 검지만 펴진 것으로 직접 구분된 이진 물리 특징", "not_implied_ko": ["주먹 전체 형태", "엄지의 정확 상태", "접촉", "사전의 검지 같은 모양과 항상 동일함"]},
    "R_open_four": {"hand": "R", "definition_ko": "오른손 네 손가락이 펴진 것으로 직접 구분된 이진 물리 특징", "not_implied_ko": ["손바닥 방향", "손가락 사이 벌림의 정확 정도", "0이면 주먹", "필수 단계의 성공"]},
    "L_open_four": {"hand": "L", "definition_ko": "왼손 네 손가락이 펴진 것으로 직접 구분된 이진 물리 특징", "not_implied_ko": ["손바닥 방향", "0이면 주먹", "휴식 중 손 모양의 오류"]},
    "R_face_overlap": {"hand": "R", "definition_ko": "화면상 오른손과 얼굴 영역의2D 겹침", "not_implied_ko": ["물리 접촉", "목/눈/뺨의 정밀 접촉점", "깊이", "올바른 얼굴 위치"]},
    "hands_overlap": {"hand": "both", "definition_ko": "화면상 양손 영역의2D 겹침", "not_implied_ko": ["물리 접촉", "어느 손가락 면이 닿았는지", "필수 단계", "양손 접근 움직임 자체"]},
}


def case_note(case_id: str) -> dict:
    for key, note in label_review["review_notes"].items():
        start, end = key.split("-")
        if case_id[0] == start[0] and int(start[1:]) <= int(case_id[1:]) <= int(end[1:]):
            return {"note": note, "source": ref(TEMPORAL / "LABEL_REVIEW.json", "/review_notes/" + key)}
    raise AssertionError(case_id)


expressions = []
for card_index, card in enumerate(cards["cards"]):
    wid = card["word_id"]
    note = NOTES[wid]
    found_form = form_lookup.get(wid)
    alias_set = {c["alias"] for c in card["clips"]}
    candidate_decisions = []
    for i, decision in enumerate(decisions["entries"]):
        if set(decision["aliases"]) & alias_set:
            candidate_decisions.append({**decision, "source": ref(TOKEN / "review/SOURCE_TOKEN_DECISIONS.json", f"/entries/{i}")})
    candidates = []
    for candidate_index, candidate in enumerate(card["candidates"]):
        candidates.append({**candidate, "source": ref(GUIDE / "GUIDE_CARDS.json", f"/cards/{card_index}/candidates/{candidate_index}"), "time_labels_from_text_allowed": False, "mandatory_requirement_enabled": False})
    atoms = []
    for atom_index, atom in enumerate(links["atoms"]):
        if atom["word_id"] != wid:
            continue
        samples = []
        for sample_index, sample in enumerate(atom["samples"]):
            latest = static_lookup.get((sample["alias"], sample["frame_id"]))
            if latest and sample["feature"] in latest[1]["labels"]:
                latest_value = latest[1]["labels"][sample["feature"]]
                latest_status = "RECHECKED_KNOWN" if latest_value is not None else "RECHECKED_UNKNOWN"
                latest_ref = ref(TEMPORAL / "LABELS.json", f"/static/{latest[0]}/labels/{sample['feature']}")
            else:
                latest_value = None
                latest_status = "NOT_IN_LATEST_FIVE_TARGET_REVIEW"
                latest_ref = None
            samples.append({**sample, "image": evidence_image(Path(sample["image"])), "latest_review_value": latest_value,
                "latest_review_status": latest_status, "latest_review_source": latest_ref,
                "current_sample_support": latest_value == atom["expected_value"] if latest_status != "NOT_IN_LATEST_FIVE_TARGET_REVIEW" else None,
                "source": ref(GUIDE / "GUIDE_CLAIM_LINKS.json", f"/atoms/{atom_index}/samples/{sample_index}")})
        atoms.append({**{k:v for k,v in atom.items() if k != "samples"}, "samples": samples,
            "source": ref(GUIDE / "GUIDE_CLAIM_LINKS.json", f"/atoms/{atom_index}"),
            "alignment_status": "PARTIAL_TEXT_TO_PHYSICAL_SAMPLE_ASSOCIATION_ONLY", "source_stage_alignment_verified": False,
            "educational_correction_enabled": False})
    clip_records = []
    for clip in card["clips"]:
        alias = clip["alias"]
        static_samples = []
        for i, row in enumerate(labels["static"]):
            if row["alias"] != alias:
                continue
            static_samples.append({"frame_id": row["frame_id"], "pts_us": row["pts_us"], "labels": row["labels"],
                "image": evidence_image(Path(draft_lookup[(alias, row["frame_id"])]["image"])),
                "source": ref(TEMPORAL / "LABELS.json", f"/static/{i}"), "status": "SAVED_CODEX_RGB_REVIEW_NOT_HUMAN_TRUTH"})
        windows = []
        for i, seq in enumerate(labels["sequences"]):
            if seq["source_alias"] != alias:
                continue
            frames = []
            for j, frame in enumerate(seq["frames"]):
                frames.append({"frame_id": frame["frame_id"], "pts_us": frame["pts_us"], "direct": frame["direct"],
                    "full_image": evidence_image(Path(frame["full_image"])), "focus_image": evidence_image(Path(frame["focus_image"])),
                    "source": ref(TEMPORAL / "LABELS.json", f"/sequences/{i}/frames/{j}")})
            windows.append({"case_id": seq["case_id"], "hand": seq["side"], "window_only": True,
                "reviewed_frame_extent_seconds": [frames[0]["pts_us"] / 1e6, frames[-1]["pts_us"] / 1e6],
                "reviewed_frame_count": len(frames), "frames": frames, "review": case_note(seq["case_id"]),
                "full_clip_dense_truth": False, "semantic_stage_boundary": False, "text_used_to_assign_times": False})
        groups = []
        for i, group in enumerate(tokens["annotations"]):
            if group["alias"] != alias:
                continue
            evidence = [{**e, "file": evidence_image(TOKEN / e["file"], e["rgb_sha256"])} for e in group["evidence"]]
            groups.append({**group, "evidence": evidence, "source": ref(TOKEN / "TOKEN_OBSERVATIONS.json", f"/annotations/{i}")})
        prior_coarse = coarse_lookup.get(alias)
        clip_records.append({**clip,
            "annotation_intervals_semantics": "AI Hub lexical annotation extent copied as metadata; NOT verified semantic-stage labels or dense physical truth.",
            "original_role_semantics": "Historical experiment role only; all36 clips are already-viewed development material in the current cycle, not new locked evaluation.",
            "static_observations": static_samples, "rgb_reviewed_windows": windows, "sampled_feature_groups": groups,
            "prior_coarse_sequence_observation": ({**prior_coarse[1], "source": ref(TOKEN / "review/OBSERVATIONS_STAGE1.json", f"/observations/{prior_coarse[0]}"),
                "status": "SPARSE_SAMPLED_OBSERVATION_NOT_EXACT_PHASE_TRUTH"} if prior_coarse else None),
            "latest_uncertainty_corrections": [v for v in label_review["uncertain_downgrades"] if v["alias"] == alias]})
    feature_counts = {}
    for feature in COMMON_FEATURES:
        cnt = collections.Counter()
        for clip in clip_records:
            for row in clip["static_observations"]:
                value = row["labels"][feature]
                cnt["unknown" if value is None else str(value)] += 1
        feature_counts[feature] = {k: cnt[k] for k in ("0", "1", "unknown")}
    expressions.append({
        "word_id": wid, "label": card["label"],
        "meaning": {"display_label": card["label"], "semantic_truth_available": False,
            "status": "SAVED_COARSE_SOURCE_REVIEW_ONLY" if found_form else "LEXICAL_LABEL_ONLY_NO_INDEPENDENT_SENSE_CONFIRMATION",
            "note_ko": found_form[1]["sense"] if found_form else "표제어와 후보 사전 연결만 보존한다. 동음이의 구별이나 의미 정답을 새로 만들지 않았다.",
            "source": ref(TOKEN / "SOURCE_FORM_LINKS.json", f"/links/{found_form[0]}") if found_form else ref(GUIDE / "GUIDE_CARDS.json", f"/cards/{card_index}/label")},
        "form_match": {**card["link"], "full_guide_match_verified": False,
            "candidate_decisions": candidate_decisions, "source": ref(GUIDE / "GUIDE_CARDS.json", f"/cards/{card_index}/link")},
        "guide_candidates": candidates,
        "supported_partial_guide_features": atoms,
        "confirmed_required_semantic_features": [],
        "physical_summary_ko": note["physical_summary_ko"],
        "unverified_or_mismatched_ko": note["unverified_ko"],
        "disabled_requirements_ko": note["disabled_ko"],
        "forbidden_inferences": {"physical_contact": True, "exact_required_repeat_count": True,
            "mandatory_semantic_stage": True, "linguistic_acceptability_bounds": True,
            "hidden_finger_or_palm_direction_from_guide": True},
        "comparison_policy": {"physical_reference_comparison_allowed": True,
            "provisional_guide_atoms_display_allowed": bool(atoms),
            "guide_as_frame_label_allowed": False, "required_semantic_stage_correction_enabled": False,
            "allowable_outputs": ["OBSERVED_PHYSICAL_DIFFERENCE", "OBSERVED_PHYSICAL_AGREEMENT", "INSUFFICIENT_OBSERVATION", "GUIDE_FORM_UNRESOLVED"],
            "forbidden_outputs": ["SIGN_IS_WRONG", "MISSING_REQUIRED_STAGE", "MUST_TOUCH", "MUST_REPEAT_EXACT_COUNT", "VARIANT_IS_UNACCEPTABLE"],
            "reference_path_policy": "Compare each complete observed reference separately; do not stitch stages from different performers into a fictitious reference."},
        "static_feature_counts_only_not_independent_sample_sizes": feature_counts,
        "clips": clip_records})

reconciled = [{"word_id": e["word_id"], "feature": a["shared_features"], "alias": s["alias"], "frame_id": s["frame_id"],
    "legacy_value": s["value"], "latest_value": s["latest_review_value"], "latest_status": s["latest_review_status"]}
    for e in expressions for a in e["supported_partial_guide_features"] for s in a["samples"]
    if s["latest_review_status"] != "NOT_IN_LATEST_FIVE_TARGET_REVIEW" and s["value"] != s["latest_review_value"]]

summary = {
    "expressions": len(expressions), "public_clips": sum(len(e["clips"]) for e in expressions),
    "previous_direct_source_form_review_expressions": len(forms["links"]),
    "fully_verified_guide_variants": 0, "mandatory_semantic_stage_labels": 0,
    "legacy_guide_atoms": len(links["atoms"]), "legacy_guide_atom_samples": sum(len(a["samples"]) for a in links["atoms"]),
    "guide_atom_expressions": len({a["word_id"] for a in links["atoms"]}),
    "latest_reconciled_value_changes": len(reconciled),
    "latest_static_samples": len(labels["static"]), "latest_rgb_reviewed_windows": len(labels["sequences"]),
    "latest_rgb_window_frames": sum(len(s["frames"]) for s in labels["sequences"]),
    "old_sparse_feature_groups": len(tokens["annotations"]),
    "source_files": len(sources), "evidence_image_files": len(images),
    "new_video_decodes": 0, "new_inference_runs": 0, "new_training_runs": 0,
    "private_data_access": 0, "new_expert_or_independent_human_labels": 0,
}

result = {
    "schema_version": "wis.guide_stage_map.v1",
    "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "status": "SAVED_EVIDENCE_RECONCILED_PHYSICAL_REFERENCE_MAP_NO_SEMANTIC_CORRECTION_TRUTH",
    "author": "Codex /root/guide_stage_map", "independent_verification": "Pending parent review; builder assertions are self-checks.",
    "scope": "All12 existing public expressions retained; saved text, sparse RGB reviews and latest short-window labels reconciled. No fresh visual interpretation or training.",
    "summary": summary,
    "evidence_policy": {
        "labels": "Saved Codex RGB-visible physical annotations are provisional research labels, not independent expert or human correctness labels.",
        "text": "A dictionary description is an expectation about its own entry. No text span generates frame labels, times, requiredness or tolerance.",
        "time": "Sparse sample extents are first/last observed samples only. Only explicitly listed13-frame windows have consecutive image review; never extend them to entire lexical intervals.",
        "chronology": "Sampled feature-group index is not stage order; groups may overlap/interleave. Sort actual PTS and preserve hand identity/uncertainty.",
        "unknown": "null remains unknown; it is neither normal, wrong nor absent. Preserve position evidence when shape is unknown.",
        "2d": "2D overlap and proximity do not establish physical contact, depth or palm direction.",
        "meaning": "Word IDs and labels select references after observation; they are never observation-model inputs or physical ground truth.",
        "split": "All36 old clips have been inspected and are development material; no claim of a new held-out evaluation.",
        "normal_variation": "Different normal-looking references do not establish a linguistic tolerance boundary. Differences are descriptive only.",
        "gratitude_refinement": "SOURCE_TOKEN_DECISIONS refines the earlier palm-up-looking observation: lower-hand occlusion prevents declaring a definite orientation contradiction. Preserve both records.",
    },
    "common_physical_feature_contract": COMMON_FEATURES,
    "additional_observation_scope": {
        "L_index_only": "9 legacy guide-linked samples; outside current5-target head and latest static review. Keep draft support separate.",
        "movement_or_palm_direction": "Coarse saved sequence notes may be displayed with their evidence, but are not newly validated dense direction/orientation labels or additional trained outputs.",
        "change": "Compare known endpoints or reviewed windows; changes between sparse samples have interval uncertainty, not exact boundary truth.",
    },
    "legacy_to_latest_reconciliation": reconciled,
    "expressions": expressions,
    "source_ledger": list(sources.values()),
    "evidence_image_ledger": list(images.values()),
    "next_usable_scope_ko": [
        "5항목 관측과 모양 변화의 시점을 개별 참고 영상과 비교하고 근거 프레임/미관측을 표시하는 연구 경로를 계속 구현할 수 있다.",
        "사전 부분 연결은 해당 출처와 잠정 상태를 함께 보여주는 설명으로만 사용한다. 필수 단계/정확접촉/반복횟수 교정은 비활성 상태를 유지한다.",
        "8표현의 전체 뜻·표현형 대조, 멀다의 대체 출처, 감사의 면 방향/접촉/횟수, 별의 변이·경로 등만 추후 수어 사용자/교육자 확인 대상으로 좁힌다.",
        "지원표와 학습/평가 분할은 별도 새 주기의 근거로 정하며 이 지도는 추가 공개자료의 관측을 생성하거나 기존 완료 실험을 재학습하지 않는다.",
    ],
}

checks = []


def check(name: str, passed: bool, detail=None):
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


check("exact12expressions36clips", len(expressions) == 12 and summary["public_clips"] == 36)
check("all_source_labels_and_ids_preserved", {e["word_id"] for e in expressions} == {c["word_id"] for c in cards["cards"]})
check("all108static36windows468frames_preserved", (summary["latest_static_samples"], summary["latest_rgb_reviewed_windows"], summary["latest_rgb_window_frames"]) == (108,36,468))
check("source_hash_link_intact", labels["review_sha256"] == digest(TEMPORAL / "LABEL_REVIEW.json"))
check("no_semantic_correction_enabled", all(not e["comparison_policy"]["required_semantic_stage_correction_enabled"] and not e["confirmed_required_semantic_features"] for e in expressions))
check("no_time_labels_generated_from_text", all(not w["text_used_to_assign_times"] for e in expressions for c in e["clips"] for w in c["rgb_reviewed_windows"]))
check("latest17downgrades_preserved", sum(len(c["latest_uncertainty_corrections"]) for e in expressions for c in e["clips"]) == 17)
check("rejected_source_candidates_not_enabled", all(not c["mandatory_requirement_enabled"] for e in expressions for c in e["guide_candidates"] if c["status"] == "REJECTED_FOR_CURRENT_VIDEO_FORM"))
check("all_static_labels_exactly_preserved", all(row["labels"] == static_lookup[(c["alias"],row["frame_id"])][1]["labels"] for e in expressions for c in e["clips"] for row in c["static_observations"]))
check("all_frozen_sources_unchanged", all(digest(ROOT / r["path"]) == r["sha256"] for r in sources.values()))
check("all_images_exist_and_have_sha256", all((ROOT / r["path"]).is_file() and len(r["file_sha256"]) == 64 for r in images.values()))
check("all_windows_monotonic_consecutive", all(all(b["frame_id"] == a["frame_id"]+1 and b["pts_us"]>a["pts_us"] for a,b in zip(w["frames"],w["frames"][1:])) for e in expressions for c in e["clips"] for w in c["rgb_reviewed_windows"]))
check("all63legacyguideatomsamples_preserved", sum(len(a["samples"]) for e in expressions for a in e["supported_partial_guide_features"]) == 63)
if not all(c["pass"] for c in checks):
    raise AssertionError([c for c in checks if not c["pass"]])

output_path = OUT / "guide_stage_map.json"
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
verification = {"status": "PASS_SELF_CHECK_INDEPENDENT_REVIEW_PENDING", "author": result["author"],
    "output": relative(output_path), "output_sha256": digest(output_path), "checks": checks,
    "source_ledger": list(sources.values()), "summary": summary,
    "not_run": ["training", "model inference", "new image/video interpretation", "private-data access", "network", "root documentation writes"]}
(OUT / "GUIDE_STAGE_MAP_SELF_CHECK.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["가이드·시간 관측 연결 검토 — 2026-10-31", "작성: Codex guide_stage_map 하위 작업. 독립 검증: 상위 담당자 검토 대기.", "",
    "완료: 기존12표현/36공개영상 전체를 남기고 사전 후보·뜻/표현형 상태·물리 표본·연속 검토 시간창·미확인 규칙을 연결했다.",
    "직접 뜻/형태 대조 기록은4표현이며, 완전한 사전 변이 확정은0개다. 필수 의미단계·접촉·정확횟수·언어적 허용 경계의 정답은 만들지 않았다.",
    "108정적시점과36개13프레임창(468프레임), 기존38표본특징묶음 및63개가이드연결시점을 그대로 연결했다.",
    f"출처{len(sources)}파일과 근거이미지{len(images)}개의 경로/파일SHA256을 보존했다. 옛 RGB배열SHA는 원기록 값을 별도 필드로 보존하고 재디코딩 검증했다고 주장하지 않는다.",
    f"이전 가이드 표본값과 최신 재검토값의 차이는{len(reconciled)}개다. 최신 보류를 옛 사전/초안으로 다시 채우지 않는다.", "",
    "핵심 구분:",
    "- 멀다8171: 현3영상의 한손 형태와 달라 요구사항 비활성. 수행 오류 판정이 아니다.",
    "- 믿다6700: 편 검지 같은 모양+반대 펼친손 접근만 잠정 연결. 가려진 R_index_only는 계속null.",
    "- 감사2372: 두펼친손/접근만 잠정 연결. 아래손 방향은 불일치 확정이 아니라 미확인.12468은 동음이의/형태불일치 제외.",
    "- 별6163: 모음/펼침만 부분 연결. 별1/2·높이허용·전체가로경로·반복수 미확정. 옛V11의 표본밖 종료를 생성하지 않음.",
    "- 나머지8표현: 사전 후보와 직접 물리 관측은 있으나 전체 뜻/표현형 대조가 확정되지 않음.", "",
    "계속 가능한 범위: 개별 참고의5항목 관측 차이·모양 변화·시간 대응·관측 부족을 근거와 함께 연구 화면에 표시. 가이드 전체를 강제 정답으로 쓰거나 수정 필요로 높이지 않음.",
    "필수성/허용범위/실제오류를 확인할 독립 근거가 없어 교육적 교정 정확도는 미검증. 이 한계를 전체 물리 비교 경로 구현의 중단 사유로 삼을 필요는 없음.", "",
    "변경: review/build_guide_stage_map.py, review/guide_stage_map.json, review/GUIDE_STAGE_MAP_SELF_CHECK.json, 이 보고서.",
    f"실행: python review/build_guide_stage_map.py; 자체구조/보존검사{len(checks)}개PASS. 상위 담당자의 별도검증은 아직 별도.",
    "새 학습/추론/다운로드/영상재디코딩/개인자료접근/운영웹/DTW/C/T3-F/루트문서 변경0. 실행 종료."]
(OUT / "GUIDE_STAGE_MAP_REVIEW_KO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"output": relative(output_path), "sha256": digest(output_path), "summary": summary, "self_checks": len(checks)}, ensure_ascii=False))
