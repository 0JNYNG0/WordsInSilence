"""Append the verified two-clip preparation milestone; preserve historical state."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
STATE = ROOT / "research/front_keyphase_v1/ops/STATE.json"
CONTEXT = ROOT / "WordsInSilence_PROJECT_CONTEXT.md"
HANDOVER = ROOT / "research/HANDOVER_2026-09-18.md"
stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
assert result["status"] == "PASS_INPUT_CONTRACT_ONLY_NOT_TRAINING_OR_GENERALIZATION"
assert result["other_34_clips_media_processed"] == result["training_runs"] == result["classifier_inference_calls"] == 0
assert len(result["source_hashes_before"]) == 11
quality = {"status": "REVIEW_REQUIRED_MISSING_HANDS_IN_TRANSITIONS_AND_REST",
           "input_contract_status": "PASS", "reviewed_prefix": "00", "other34_media_processed": 0,
           "MEOKDA": {"both_hands_unusable_frames": 0,
                       "visually_reviewed_right_missing_transition_frames": [9, 11, 54, 56]},
           "GONGBU": {"both_hands_unusable_frames": 14, "both_missing_ranges": [[13, 16], [56, 56], [58, 66]],
                       "visually_reviewed_both_missing_transition_frames": [13, 16, 56, 60],
                       "visually_reviewed_later_rest_with_left_missing": [70, 88]},
           "observation": "Visible motion blur accompanies missing observations in selected raising/lowering frames; not a proven sole cause.",
           "threshold_implication": "observed=false has no stored coordinates; lowering usable thresholds alone cannot restore absent observations",
           "action_boundaries_created": False, "semantic_approval": False,
           "training_approved": False, "joint_coordinate_ground_truth_available": False}
(OUT / "source_quality_triage.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
next_action = (
    "Use normal official access to AIHub636 Training 수어스크립트_TL.zip (display372.37MB/key61896), then inspect actual gloss labels and signer/source-link fields for the nine unresolved targets before acquiring large media or approving any training source. "
    "Do not treat Korean-text substring hits as sign labels, login HTML as ZIP success, or official Validation as training. If normal access is unavailable, record the concrete access step and evidence without repeated unchanged retries; inquiry remains cancelled. "
    "KSL77 prefix00 input preparation is complete; no repeat. Preserve34 media-uninspected KSL candidates and all unassigned splits. Signer-code semantics/cross-source overlap, lexical equivalence/usage conditions and observed transition missingness remain KSL training-source dependencies. "
    "Maintain18-target scope and AIHub103 unknown254 WORD-to-ZIP mappings. No new training, model change, heldout inference, production edit, backup, deletion, Unity build or scheduling."
)
stage_key = "ksl77_input_prep_20260919"
stage = {"status": result["status"], "at": stamp,
         "report": "research/front_keyphase_v1/evidence/ksl77_input_prep_20260919/RESULT.md",
         "evidence": "research/front_keyphase_v1/evidence/ksl77_input_prep_20260919",
         "diagnostic_cache": "research/front_keyphase_v1/runs/ksl77_input_prep_v1",
         "new_clips_processed": 2, "prefix": "00", "raw_and_tcn_frames": [88, 89], "channels": 280,
         "left_all21_usable_frames": [77, 44], "right_all21_usable_frames": [82, 74],
         "both_hands_unusable_frames": [0, 14], "input_contract_verified": True,
         "actual_source_pts_exact": True, "dropped_frames": 0, "extractor_id": "2683eadbc5f1c588",
         "processing_seconds": result["elapsed_seconds"], "output_bytes": result["output_bytes"],
         "input_code_contract_pins_unchanged": 11,
         "quality_status": quality["status"], "action_annotations_created": False,
         "training_approved": False, "training_runs": 0, "classifier_inference_calls": 0,
         "heldout_runs": 0, "other34_media_processed": 0, "production_changes": 0,
         "inquiry_cancelled": True, "new_backups": 0, "deletions": 0, "unity_builds": 0, "scheduled_tasks": 0}


def read(path):
    data = path.read_bytes()
    enc = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    return data.decode(enc).replace("\r\n", "\n"), enc, "\r\n" if b"\r\n" in data else "\n"


text, enc, newline = read(STATE)
before = json.loads(text)
assert stage_key not in before["stage_results"]
scalars = {"updated_date_kst": stamp, "current_stage": "TCN_KSL77_PREFIX00_INPUT_PASS_TRANSITION_MISSINGNESS_REVIEW_REQUIRED",
           "next_single_action": next_action}
updated = text
for key, value in scalars.items():
    updated, count = re.subn(r'^  "' + key + r'": .*,$', lambda m: '  "' + key + '": ' + json.dumps(value, ensure_ascii=False) + ',', updated, count=1, flags=re.M)
    assert count == 1
insertion = "\n".join("  " + line for line in json.dumps({stage_key: stage}, ensure_ascii=False, indent=2)[2:-2].splitlines())
anchor = '  "stage_results": {\n'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, anchor + insertion + ",\n", 1)
expected = {**before, **scalars, "stage_results": {stage_key: stage, **before["stage_results"]}}
assert json.loads(updated) == expected

current = f"""### {stamp} 최우선 최신 상태 — KSL77 검토용2개 입력 계약 PASS, 전환 중 손 누락 확인

- 새 KSL77 00_16/eat·00_48/study만 기존 MediaPipe→웹TCN 입력 경로로 별도 진단 캐시에 처리했다. 원본/입력88·89프레임,280채널. 원본 SHA·실제 PTS·결측/usable 규칙·신체/손 정규화·마스크/시간차·저장/웹 입력 일치를 독립 검증했다. 원본/코드/계약 등11개 해시 전후 동일,8.753초,출력218,805바이트,추출기2683eadbc5f1c588이다.
- 입력 수치 PASS와 관측 품질은 구분한다. 왼손21점 전부 유효77/88·44/89,오른손82/88·74/89,양손 공백0·14프레임이다. 원본 PTS 최대 간격67.167·100.700ms를 보존했으며 처리 중 누락0이다. 유효 비율은 인식률이나 실제 좌표 정확도가 아니다.
- 해당 원본 프레임 대조에서 손을 올리고 내리는 전환 중 흐림과 관측 공백을 확인했다. 공부13~16·56·58~66의 양손 공백을 전부 휴식으로 간주하지 않는다. 일부 후반 왼손 공백은 손이 아래에서 겹친 자세에 있다. 관측 자체가 없는 프레임은 usable 임계값만 낮춰 복원할 수 없다. 단어별 예외·임계값 조정0이다.
- 두 영상 모두 현행 첫~마지막 유효 손 범위가 전체 클립이다. 동작시간 정답 라벨은 없으며 임의 구간을 만들거나 양호한 구간만 잘라 품질 성공으로 세지 않았다. 학습원 승인·TCN분류추론·새학습·보류시험·나머지34개 미디어 처리0이다. KSL 모든 분할UNASSIGNED와 촬영자 코드/의미 미확정을 유지한다.
- 근거: `research/front_keyphase_v1/evidence/ksl77_input_prep_20260919/RESULT.md`, `result.json`, `source_quality_triage.json`, `missingness_frame_evidence.json`. 캐시: `research/front_keyphase_v1/runs/ksl77_input_prep_v1/`. production 수정·백업·삭제·Unity빌드·예약작업0,문의 취소 유지.

"""
next_section = f"""### {stamp} 다음 단일 작업 — AI Hub636의 작은 Training 수어스크립트 라벨 확인

- 공식 정상 접근 경로로 AI Hub636 `수어스크립트_TL.zip`(표시372.37MB/key61896)을 확보해 미해결9목표의 실제 gloss 포함 여부와 촬영자/원본 연결 필드를 확인한다. 한국어 문장 문자열만으로 수어 라벨 포함을 세지 않는다. 큰 원천영상·89.91GB JSON 선행 다운로드나 학습원 등록은 하지 않는다. 공식Validation은 학습으로 전환하지 않는다.
- 로그인 HTML이나 버튼 클릭만으로 다운로드 성공을 주장하지 않는다. 정상 접근이 안 되면 구체적인 접근 단계/근거를 기록하고 진행 없는 재시도를 반복하지 않는다. 제공자 문의는 취소 상태로 유지한다.
- KSL77 검토용00 두 원본 입력 검증은 완료됐으므로 반복하지 않는다. 나머지34개 미디어 미검토·전체UNASSIGNED를 유지한다. 촬영자 코드 명세/출처 간 중복·한국어 의미/사용조건·전환 중 관측 공백은 학습원 결정의 미해결 의존성이다. 기존18목표·AI Hub103 WORD254개 ZIP매핑 과제·사용자기록 비학습·보류시험·Claude Code production작성/현재환경 실행·독립검증 역할을 유지한다.

"""
context, context_enc, context_newline = read(CONTEXT)
context_updated, count = re.subn(r"^최종 갱신: .*$", "최종 갱신: " + stamp, context, count=1, flags=re.M)
assert count == 1
for marker, value in (("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n", current), ("## 9. 다음 단일 작업\n\n", next_section)):
    assert context_updated.count(marker) == 1
    context_updated = context_updated.replace(marker, marker + value, 1)
handover, handover_enc, handover_newline = read(HANDOVER)
handover_update = f"""## 후속 갱신 — {stamp}

KSL77 검토용00_16/eat·00_48/study 두 원본의 기존 공통 MediaPipe→TCN 입력 준비를 완료했다.88·89프레임/280채널,실제PTS·결측·정규화·마스크·저장/웹 입력일치 PASS.11개 원본/코드/계약 등 해시 전후 동일,8.753초,출력218,805바이트다. 기존AI Hub2개 PASS는 반복하지 않았다.

자료 품질은 별도다. 왼손21점 전부 유효77/88·44/89,오른손82/88·74/89,공부 양손 공백14프레임. 원본 대조에서 손을 올리고 내리는 전환 중에도 흐림/관측 공백이 있어 전부 휴식으로 볼 수 없다. 수치PASS는 실제 관절 좌표 정확도·단어분류·새사람 성능PASS가 아니다. 임의 행동구간·단어예외·임계값변경·학습원 등록0이다.

**다음 단일 작업은 AI Hub636 공식 Training 수어스크립트_TL.zip372.37MB/key61896의 실제 gloss/촬영자·원본 연결 필드 확인**이다. 정상 접근이 실패하면 구체적 증거를 남기고 같은실패를 반복하지 않으며 문의는 재개하지 않는다. KSL두개 입력 검증은 완료,나머지34개 미디어 미검토,모두UNASSIGNED와 의미/촬영자 미확정을 유지한다.18목표와AI Hub103 WORD254개 ZIP매핑 과제는 그대로다.

근거: `front_keyphase_v1/evidence/ksl77_input_prep_20260919/RESULT.md`. production변경·TCN분류추론·새학습·보류시험·사용자기록 사용·삭제·백업·Unity빌드·예약작업0. 최신 기준은 루트8·9절 맨 위와 실제 `front_keyphase_v1/ops/STATE.json`이다.

"""
marker = "# WordsInSilence — 새 대화 인계\n\n"
assert handover.startswith(marker)
handover_updated = handover.replace(marker, marker + handover_update, 1)
for path, data, encoding, nl in ((STATE, updated, enc, newline), (CONTEXT, context_updated, context_enc, context_newline), (HANDOVER, handover_updated, handover_enc, handover_newline)):
    path.write_bytes(data.replace("\n", nl).encode(encoding))
assert json.loads(STATE.read_text(encoding=enc)) == expected
assert read(CONTEXT)[0] == context_updated and read(HANDOVER)[0] == handover_updated
validation = {"status": "PASS", "at": stamp, "previous_state_fields_and_stage_history_preserved": True,
              "document_latest_sections_match": True, "new_stage_key": stage_key,
              "document_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (STATE, CONTEXT, HANDOVER)}}
(OUT / "document_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "at": stamp, "stage": stage_key, "historical_state_preserved": True}, ensure_ascii=False))
