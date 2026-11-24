"""Record this acquisition milestone without rewriting previous state history."""
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
stage_key = "additional_source_labels_20260918"
next_action = (
    "Prepare only the newly acquired KSL77 prefix00 clips 00_16/eat and 00_48/study through the existing shared MediaPipe-to-280-channel TCN input path in an isolated diagnostic cache; verify source hashes, actual PTS, hand usability/missingness and normalization without inventing annotated action boundaries. "
    "Do not rerun the already-passed AIHub103 two-clip preparation. All36 KSL clips remain unassigned/not training-approved;34 other clips remain media-uninspected, and review-exposed prefix00 must not enter a future blind holdout. "
    "Filename prefixes are not verified signer identities; semantic equivalence, signer metadata/cross-source overlap and source usage conditions remain prerequisites for training-source decisions. "
    "Keep18 targets, AIHub103 unknown254 WORD-to-ZIP mappings and AIHub636 target coverage UNKNOWN. Inquiry stays cancelled; no training, model replacement, heldout inference, production changes, cleanup, backup, Unity build or automation."
)
stage = {
    "status": "PASS_SOURCE_ACQUISITION_NOT_TRAINING_OR_GENERALIZATION",
    "at": stamp,
    "report": "research/front_keyphase_v1/evidence/additional_source_labels_20260918/RESULT.md",
    "source_repository": "https://github.com/Yangseung/KSL",
    "author_nonempty_labels": 77, "actual_archive_class_directories": 67,
    "actual_archive_videos": 1229, "independent_directory_rows_exact": 1296,
    "raw_candidate_videos_acquired": 36, "raw_bytes_acquired": 129466477,
    "source_classes": {"16_eat_MEOKDA_candidate": 17, "48_study_GONGBU_candidate": 19},
    "all_available_members_of_selected_classes_acquired": True,
    "stored_bytes_crc_sha_verified": 36,
    "reviewed_prefixes": ["00"], "new_videos_decoded_for_visual_triage": 2,
    "new_videos_media_uninspected": 34,
    "selected_distinct_filename_prefixes": 19, "verified_signer_count": None,
    "signer_status": "UNVERIFIED_CODE_SEMANTICS_AND_CROSS_SOURCE_OVERLAP",
    "split_status": "ALL_UNASSIGNED_SEPARATE_KSL77_NAMESPACE;prefix00_REVIEW_EXPOSED",
    "semantic_merges_approved": 0, "new_training_rows_approved": 0,
    "aihub636": "Public schema and training script ZIP372.37MB/key61896 observed;actual target/signer coverage UNKNOWN;no local sample ZIP verified",
    "aihub103_unknown_word_zip_mappings": 254,
    "training_runs": 0, "classifier_inference_calls": 0, "held_out_data_runs": 0,
    "production_code_changes": 0, "new_backups": 0, "deletions": 0,
    "unity_builds": 0, "scheduled_tasks": 0, "inquiry_cancelled": True,
    "evidence": "research/front_keyphase_v1/evidence/additional_source_labels_20260918",
    "raw_path": "research/front_keyphase_v1/runs/ksl77_source_review_v1/raw"
}


def read_preserving(path):
    raw = path.read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    newline = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode(encoding).replace("\r\n", "\n"), encoding, newline


state_text, state_encoding, state_newline = read_preserving(STATE)
before = json.loads(state_text)
assert stage_key not in before["stage_results"], "Milestone already recorded; do not duplicate"
new_state = state_text
new_scalars = {"updated_date_kst": stamp,
               "current_stage": "TCN_KSL77_TWO_TARGET_CANDIDATE_RAW_ACQUISITION_PASS",
               "next_single_action": next_action}
for key, value in new_scalars.items():
    new_state, count = re.subn(r'^  "' + re.escape(key) + r'": .*,$',
                               lambda match: '  "' + key + '": ' + json.dumps(value, ensure_ascii=False) + ',',
                               new_state, count=1, flags=re.M)
    assert count == 1
stage_text = json.dumps({stage_key: stage}, ensure_ascii=False, indent=2)[2:-2]
stage_text = "\n".join("  " + line for line in stage_text.splitlines())
anchor = '  "stage_results": {\n'
assert new_state.count(anchor) == 1
new_state = new_state.replace(anchor, anchor + stage_text + ",\n", 1)
after = json.loads(new_state)
expected = dict(before)
expected.update(new_scalars)
expected["stage_results"] = {stage_key: stage, **before["stage_results"]}
assert after == expected

current_section = f"""### {stamp} 최우선 최신 상태 — KSL77 공부·먹다 후보 원본36개 확보·무결성 PASS

- AI Hub636의 공개 스키마/예시와 파일목록을 확인했으나 실제 목표9단어·촬영자 값은 미확인이다. 공개 예시 gloss는 날씨1이며, 한국어 문장 문자열을 수어 라벨로 세지 않는다. 익명 샘플은 로그인 요구 HTML이었고 브라우저 경로도 로컬 ZIP 저장을 검증하지 못했다. 재접근 경로는 수어스크립트_TL.zip372.37MB/key61896이며 큰 JSON89.91GB를 선행하지 않는다.
- 연구자 공개 KSL77 라벨을 실행 없이 정적 해석해 study=48/eat=16을 찾았다. 4.28GiB 전체 ZIP 대신 파일목록과 두 클래스만 범위 요청했다. 공부19개·먹다17개, 두 클래스에 실제 있는36개 전부 확보했다. 원본129,466,477바이트(약123.5MiB), 저장 길이·CRC32·SHA-25636/36 PASS. 표준 zipfile 독립 목록1,296항목도 exact다.
- 논문77단어·20명을 실제 확보량으로 오해하지 않는다. 공개 ZIP은67개 클래스·1,229영상이며, 확보한36개의 파일명 앞 코드는19종이다. 코드가 촬영자 ID라는 명세와 출처 간 사람 중복은 미확인이다. 모두 별도 KSL77 공간 UNASSIGNED, 학습원 승인0이다.
- 사전 지정00 코드2개만 디코드/정지프레임 대조했다. 먹다88·공부89프레임,1280×720,단조 PTS 확인. 기존 시범과 큰 동작 형태는 유사하지만 의미 동등성·수행 정확성 승인0이다. 00은 검토 노출로 향후 맹검 보류에서 제외하며 나머지34개 미디어는 미검토·미추출·미추론이다. 얼굴 비교로 신원 판단하지 않았다.
- 기존18목표와 AI Hub103 WORD원본254개 ZIP미확인을 유지한다. hi 라벨도 INSA에 자동 병합하지 않는다. 이번 원본 확보는 새 학습·새사람 일반화 완료가 아니다. 문의 취소,사용자기록 비학습,production 무변경,백업·삭제·Unity빌드·예약작업0을 유지했다.
- 근거: `research/front_keyphase_v1/evidence/additional_source_labels_20260918/RESULT.md`, `ksl77_acquisition.json`, `independent_source_verification.json`, `ksl77_coverage_and_grouping.json`. 원본: `research/front_keyphase_v1/runs/ksl77_source_review_v1/raw/`.

"""
next_section = f"""### {stamp} 다음 단일 작업 — KSL77 새 원본00 코드2개의 기존 공통 입력 준비

- 새 `00_16.MP4`(eat/먹다후보)·`00_48.MP4`(study/공부후보)만 기존 MediaPipe→TCN280채널 경로로 별도 진단 캐시에 처리한다. 출처 SHA·실제 PTS·양손21관절의 관측/결측·신체/손 정규화·저장/웹 입력 일치를 검증한다. 학습원 등록이나 새 학습은 하지 않는다. 동작시간 라벨이 없는 자료에 임의 정답 구간을 만들지 않는다.
- 기존 AI Hub WORD1544/2485 입력 PASS는 반복하지 않는다. 새 KSL77 나머지34개는 미디어 미검토로 보존한다. 모든 KSL 분할은 UNASSIGNED이며 촬영자 코드 명세·출처 간 중복·의미/형태·사용조건 확인 전 train/dev/holdout을 확정하지 않는다. 검토 노출00은 이후 맹검 보류에서 제외한다.
- AI Hub103 WORD254개 ZIP매핑과 추가 목표자료 확보는 별도 의존성이다. 636목표 포함 여부는 UNKNOWN,없음이 아니다. 문의는 다시 진행하지 않는다. 전체18목표·사용자기록 비학습·보류시험·Claude Code production코드작성/현재환경 실행·독립검증 역할을 유지한다.

"""
context_text, context_encoding, context_newline = read_preserving(CONTEXT)
context_updated, count = re.subn(r"^최종 갱신: .*$", "최종 갱신: " + stamp, context_text, count=1, flags=re.M)
assert count == 1
for marker, addition in (("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n", current_section),
                         ("## 9. 다음 단일 작업\n\n", next_section)):
    assert context_updated.count(marker) == 1
    context_updated = context_updated.replace(marker, marker + addition, 1)
handover_text, handover_encoding, handover_newline = read_preserving(HANDOVER)
handover_section = f"""## 후속 갱신 — {stamp}

AI Hub636 공개 스키마/예시·파일목록은 확인했으나 실제 목표9단어·촬영자 포함 여부는 UNKNOWN이다. 다음 접근 후보는 공식 Training 수어스크립트_TL.zip372.37MB/key61896이다. 실제 샘플 ZIP 저장을 검증하지 못했으며 문의를 재개하지 않았다.

대체 연구자 공개 KSL77에서 study=48/eat=16 라벨과 ZIP 실제 멤버를 확인하고 **공부19개·먹다17개 후보 원본36개 전부**를 선택 확보했다. 저장129,466,477바이트(약123.5MiB),저장파일 CRC/SHA36/36,표준 zipfile 독립 목록1,296항목 일치 PASS. 전체 ZIP4.28GiB는 받지 않았다. 공개 ZIP 자체는67클래스·1,229영상이며 논문77단어·20명과 실제 확보수를 구분한다.

촬영자 코드는 아직 명세로 검증하지 못했다. 모든 새 자료는 KSL77 별도 공간 UNASSIGNED·학습 미승인이다. 미리 정한00 코드2개만 디코드/형태 대조했고,먹다88·공부89프레임의 큰 동작 형태 유사성을 관찰했다. 의미 동등성·수행 정확성 승인은 아니다. 00은 향후 맹검 보류에서 제외하고 나머지34개는 미디어 미검토/미추출로 보존한다. 기존AI Hub 분할과18목표를 유지한다.

**다음 단일 작업은 새 KSL77 00_16/00_48 두 원본의 기존 공통 MediaPipe→TCN280채널 입력 준비·손 관측/시간/정규화 검증**이다. 라벨 없는 동작시간을 임의로 만들지 않는다. 기존AI Hub2개 PASS는 반복하지 않는다. 새 학습·모델변경·보류시험·사용자기록 사용·production변경·삭제·백업·Unity빌드·예약작업은 하지 않았다.

근거: `front_keyphase_v1/evidence/additional_source_labels_20260918/RESULT.md`. 최신 기준은 루트8·9절 맨 위와 실제 `front_keyphase_v1/ops/STATE.json`이다.

"""
assert handover_text.startswith("# WordsInSilence — 새 대화 인계\n\n")
handover_updated = handover_text.replace("# WordsInSilence — 새 대화 인계\n\n",
                                        "# WordsInSilence — 새 대화 인계\n\n" + handover_section, 1)

# All parsing and uniqueness assertions precede mutations.
for path, content, encoding, newline in ((STATE, new_state, state_encoding, state_newline),
                                        (CONTEXT, context_updated, context_encoding, context_newline),
                                        (HANDOVER, handover_updated, handover_encoding, handover_newline)):
    path.write_bytes(content.replace("\n", newline).encode(encoding))
assert json.loads(STATE.read_text(encoding=state_encoding)) == expected
assert read_preserving(CONTEXT)[0] == context_updated
assert read_preserving(HANDOVER)[0] == handover_updated

prior = json.loads((OUT.parent / "isolated_word_input_prep_20260918/result.json").read_text(encoding="utf-8"))
pins = {path: value for path, value in prior["source_hashes_before"].items()
        if "/src/" in path.replace("\\", "/") or path.replace("\\", "/").endswith("/config/contract.json")}
pin_checks = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in pins.items()}
assert pin_checks and all(pin_checks.values())
validation = {"status": "PASS", "at": stamp, "new_stage_key": stage_key,
              "previous_state_fields_and_stage_history_preserved": True,
              "root_latest_sections_and_handover_match": True,
              "production_and_contract_pins_unchanged_from_prior_prep": pin_checks,
              "documents": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (STATE, CONTEXT, HANDOVER)}}
(OUT / "document_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "at": stamp, "old_state_history_preserved": True,
                  "production_contract_pins_exact": len(pin_checks), "raw_videos_verified": 36,
                  "next": "KSL77 prefix00 two-clip shared-input preparation"}, ensure_ascii=False))
