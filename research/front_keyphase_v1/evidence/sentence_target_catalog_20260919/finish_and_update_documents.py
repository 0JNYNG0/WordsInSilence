"""Record observed acquisition state and preserve earlier project history."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
EVIDENCE = OUT.parent
STAMP = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


native = load(OUT/'apology_native_prep.json')
catalog = load(OUT/'catalog.json')
raw = load(OUT/'apology_acquisition.json')
assert native['status'] == 'PASS_NATIVE_CANDIDATE_INPUT_ONLY'
assert native['prepared_clips'] == 12 and native['prepared_frames'] == 417
assert native['reserved_test_arrays_read'] == native['training_runs'] == 0
assert native['pins_unchanged'] and not native['failures']
assert {r['signer_id'] for r in native['rows']} == {1,2,3,4,5,6,7,8,9,11,12,13}

access = {
    'at': STAMP, 'dataset': 636, 'status': 'NORMAL_DOWNLOAD_APPLICATION_GATE_NOT_PROCEEDED',
    'evidence_kind': 'Observed browser UI and bounded download-directory checks in this conversation',
    'user_instruction_verbatim': '근데 데이터신청 문의는 하지마.',
    'operational_constraint': 'Do not submit data applications, accept use agreements, or submit provider inquiries.',
    'login': 'Completed manually by user; authenticated header observed',
    'download_page': 'https://aihub.or.kr/aihubdata/data/dwld.do?currMenu=&topMenu=&dataSetSn=636',
    'observed_gate': 'Data-application purpose dropdown and required use-agreement checkbox before download',
    'application_submissions': 0, 'agreement_acceptances': 0, 'inquiry_submissions': 0,
    'credentials_read_or_copied': False,
    'sample': {'observed_link': 'https://aihub.or.kr/file/down.do?fileSn=15287&smpleFileSn=15287&dataSetSn=636',
               'action': 'Clicked public sample link once after user login',
               'saved_artifact': 'UNVERIFIED; no new sample ZIP found in checked download locations',
               'viewport': '425px hid sample controls; temporary1280x900 exposed link; reset afterward'},
    'final_page': 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=636',
    'target_gloss_coverage': 'UNKNOWN', 'retry_without_new_evidence': False,
    'next_route': 'Use already-owned local AIHub103 sources; no application or inquiry dependency'}
dump('aihub636_access_stop.json', access)
triage = {
    'status': 'FORM_COMPATIBLE_CANDIDATE_NOT_SEMANTIC_APPROVAL',
    'reviewed_image': str(OUT/'apology_comparison.jpg'),
    'source': raw['path'], 'source_label': raw['source']['raw_label'],
    'reference': str(ROOT/'Tools/videos/죄송하다.mp4'),
    'observation': 'Selected frames in both show the right hand near the forehead followed by a lowering movement toward the left hand; hand placement and bow differ.',
    'limits': 'Broad visual compatibility only; no qualified lexical-equivalence or correct-performance judgment.',
    'semantic_target_id': None, 'training_approved': False, 'other_signers_rgb_read': 0}
dump('apology_visual_triage.json', triage)

previous = load(EVIDENCE/'isolated_word_input_prep_20260918/coverage_18.json')
ksl = load(EVIDENCE/'additional_source_labels_20260918/ksl77_coverage_and_grouping.json')
by_target = {r['target']:r for r in catalog['summary']}
ksl_by_target = {r['target_candidate']:r for r in ksl['classes']}
ledger = []
for old in previous['rows']:
    target = old['word_id']
    sentence = by_target[target]
    row = {**old,
           'sentence_candidate_annotation_rows': sentence['candidate_annotation_rows'],
           'sentence_candidate_raw_labels': sentence['raw_labels'],
           'sentence_candidate_signers': sentence.get('signer_ids_in_matched_metadata', []),
           'sentence_originals_acquired_in_this_acquisition_series': 2 if target == 'INSA' else 1 if target == 'JOESONGHADA' else 0,
           'new_native_candidate_inputs_train_dev': 12 if target == 'JOESONGHADA' else 0,
           'ksl_original_candidates_acquired': ksl_by_target.get(target, {}).get('files_available_and_acquired',0),
           'ksl_common_input_review_clips': 1 if target in ksl_by_target else 0,
           'ksl_signer_identity': 'UNVERIFIED' if target in ksl_by_target else 'NOT_APPLICABLE',
           'new_source_training_approved': False}
    if target == 'JOESONGHADA':
        row['additional_sentence_discovery'] = 'SEN0356: 16 metadata rows; REAL01 raw; 9 train-candidate +3 development native inputs; semantic equivalence pending'
    ledger.append(row)
assert len(ledger) == 18
dump('coverage_18.json', {'schema':'wis_target18_source_readiness_v2','at':STAMP,'rows':ledger,
    'notes':['Existing-training counts inherited from the prior audited ledger; not new training.',
             'Zero discovery hits are not proof of corpus-wide absence.',
             'Same clip can contain more than one annotation; annotation count is not unique-video count.',
             'AIHub WORD seed ledger remains 256 metadata rows: 2 acquired originals, 254 unresolved ZIP mappings.',
             'KSL36 stays UNASSIGNED; only prefix00 two clips reviewed; other34 media untouched.',
             'Native-input preparation is not MediaPipe extraction, training eligibility or generalization success.']})
lines = ['# 18개 목표별 추가 자료 준비 현황', '', f'갱신: {STAMP}', '',
         '기존 AI Hub 열은 기존 학습 클립 수다. 이후 열은 새 후보 자료이며 학습 미승인이다. 0은 이번 검색/확보에서 0이며 전체 자료에 없다는 뜻이 아니다.', '',
         '|목표|기존 AI Hub 학습 클립|WORD 라벨 행 / 새 원본|문장 후보 라벨 행 / 새 원본|새 관절 입력|KSL 원본 / 검토 입력|',
         '|---|---:|---:|---:|---:|---:|']
for r in ledger:
    lines.append(f"|{r['label']}|{r['existing_aihub_sentence_training_clips']}|{r['seed_label_rows']} / {r['seed_raw_videos_acquired']}|{r['sentence_candidate_annotation_rows']} / {r['sentence_originals_acquired_in_this_acquisition_series']}|{r['new_native_candidate_inputs_train_dev']}|{r['ksl_original_candidates_acquired']} / {r['ksl_common_input_review_clips']}|")
lines += ['', '새 관절 입력12개는 죄송하다 연결 후보인 미안합니다 자료다. 학습후보9명·개발3명이며 보류14~16의 배열은 읽지 않았다. 문장 라벨에는 보류 메타데이터가 포함된다.',
          '기존 NIKL 시범은 목표마다1개다. 표현의 의미 동등성·새 사람 성능을 이 표로 승인하지 않는다. KSL 파일명 코드의 사람 식별 의미는 미확정이다.',
          'AI Hub636 대상 포함 여부 UNKNOWN. WORD 원본254개 ZIP 위치 미해결, 기존18목표 범위 유지.', '']
(OUT/'COVERAGE_18.md').write_text('\n'.join(lines), encoding='utf-8')

next_action = ('Compare the newly acquired SEN0356 REAL01 front video through the existing MediaPipe/web280-channel path with its native OpenPose candidate on the same source. '
               'Verify source timing/frame-origin alignment against actual PTS and annotation bounds before interpreting detector differences; keep full clip and label interval separate. '
               'This is a new source-domain diagnostic, not repeat verification of prior WORD/KSL inputs. Preserve native confidence, actual missingness and source/form provenance; no per-word threshold exceptions or automatic semantic approval. '
               'The 12-person native candidate is 9 train candidates plus3 development, not trained. Keep REAL10 unassigned, REAL14-16 metadata only, officialValidation out of training, KSL34 media untouched. '
               'No data application, agreement acceptance or provider inquiry. AIHub636 route stops at observed application gate, target coverage unknown. '
               'Maintain18 targets, WORD254 ZIP-mapping dependency, no user-test training, no new model training/replacement, heldout inference, production edits, backup, deletion, Unity build or schedule. Production authorship remains Claude Code; Codex operations and independent verification.')
report = f'''# 로컬 문장 후보 확장과 촬영자별 입력 준비

갱신: {STAMP}

- AI Hub636 로그인은 사용자 직접 완료. 다운로드가 데이터 신청·사용조건 동의 화면으로 연결돼 진행을 중단했다. 신청·동의·문의 제출0. 공개 샘플 버튼은 확인했으나 저장된 ZIP을 검증하지 못했으므로 확보로 세지 않는다. 사용자의 신청·문의 금지 지시를 유지한다.
- 보유 AI Hub103 문장 라벨 중 REAL01 정면2,000개를 검색하고 관련14문장의 촬영자01~16 메타데이터를 연결했다. 7목표의 후보221라벨행이며 18목표 전체를 충족한 것이 아니다. 기존9빈칸 중 새 검색에 나타난 것은 이미 알려진 인사 후보뿐이다. 미검출을 전체 코퍼스 부재로 해석하지 않는다.
- 신규 유용 후보: SEN0356의 미안합니다15·죄송하다1 라벨, 촬영자01~16. 기존80클래스 manifest에 두 라벨은0개였다. REAL01 정면 원본을 보유 ZIP에서 선택 추출했다. {raw['bytes']:,}바이트, CRC32 {raw['crc32']}, 저장 SHA 검증 PASS. 131프레임 원본이며 다른 사람 영상은 확보하지 않았다.
- 기존 독자 관절 리더와 공통 TCN 어댑터로 REAL01~09·11~13의12클립/417프레임을 별도 캐시에 준비했다. 학습 후보9명·개발3명, 각26~42프레임,280채널. 선택 멤버 CRC·스키마·유한값·마스크·시간차·저장/재읽기 일치 PASS. 6개 입력/코드/manifest pin과12원본 ZIP stat 전후 동일. 처리{native['elapsed_seconds']}초, 캐시{native['output_bytes']:,}바이트. REAL10·14~16 배열 읽기0.
- 이 좌표는 제공된 OpenPose 원본이며 MediaPipe 검출 결과가 아니다. 시간은 frame index/명목30fps로 실제 PTS·프레임 시작번호 정렬 미검증이다. coordinate_present와 native confidence는 좌표 정확성 점수가 아니다. 의미 동등성·완전한 동작 경계·웹 검출기와의 차이는 미확정이고 학습원 등록0이다.
- 정지 프레임의 큰 동작 형태는 기존 죄송하다 시범과 호환되는 후보로 보였지만 의미 승인으로 삼지 않았다. 기존18목표·KSL36 UNASSIGNED·그중34미디어 미검토·WORD원본254개 ZIP미확인을 유지한다.
- 최초 감사 스크립트의 계약파일 경로 오류는 캐시 생성/배열 읽기 전에 실패했다. 실패 기록을 보존하고 새 감사 스크립트 경로만 고쳤다. production 변경0, 학습·분류추론·보류시험·사용자 기록 사용·새백업·삭제·Unity빌드·예약작업0.

다음 단일 작업: 새 SEN0356 REAL01 원본을 기존 MediaPipe/웹 입력으로 준비하고 실제 PTS와 제공 관절의 프레임 정렬을 먼저 확인한 다음, 같은 원본에서 검출 방식에 따른 차이를 분석한다. 기존 통과 입력 검증은 반복하지 않는다.

근거: `aihub636_access_stop.json`, `catalog.json`, `candidate_rows.json`, `apology_acquisition.json`, `apology_visual_triage.json`, `apology_native_prep.json`, `apology_native_frame_provenance.json`, `coverage_18.json`, `COVERAGE_18.md`. 새 캐시: `research/front_keyphase_v1/runs/apology_native_prep_v1/`.
'''
(OUT/'RESULT.md').write_text(report, encoding='utf-8')

stage_key = 'sentence_target_catalog_20260919'
stage = {'status':'APOLOGY12_NATIVE_CANDIDATES_PREPARED_NOT_TRAINED','at':STAMP,
         'report':'research/front_keyphase_v1/evidence/sentence_target_catalog_20260919/RESULT.md',
         'target_readiness':'research/front_keyphase_v1/evidence/sentence_target_catalog_20260919/coverage_18.json',
         'aihub636_access':access['status'],'user_prohibits_data_application_and_inquiry':True,
         'application_submissions':0,'agreement_acceptances':0,'inquiry_submissions':0,
         'local_sentence_candidate_rows':221,'candidate_targets':7,'new_apology_raw_clips':1,
         'native_input_clips':12,'native_input_frames':417,'native_roles':native['role_counts'],
         'actual_video_alignment':'UNVERIFIED','semantic_approval':False,'training_runs':0,
         'heldout_arrays_read':0,'production_changes':0,'new_backups':0,'deletions':0,'unity_builds':0,'scheduled_tasks':0}


def read_text(path):
    b=path.read_bytes();enc='utf-8-sig' if b.startswith(b'\xef\xbb\xbf') else 'utf-8'
    return b.decode(enc).replace('\r\n','\n'),enc,'\r\n' if b'\r\n' in b else '\n'


state_path=ROOT/'research/front_keyphase_v1/ops/STATE.json'
text,enc,nl=read_text(state_path);before=json.loads(text)
assert stage_key not in before['stage_results']
assert before['current_stage']=='TCN_KSL77_PREFIX00_INPUT_PASS_TRANSITION_MISSINGNESS_REVIEW_REQUIRED'
scalars={'updated_date_kst':STAMP,'current_stage':'TCN_APOLOGY12_NATIVE_INPUTS_READY_DOMAIN_ALIGNMENT_PENDING','next_single_action':next_action}
updated=text
for k,v in scalars.items():
    updated,n=re.subn(r'^  "'+k+r'": .*,$',lambda m:'  "'+k+'": '+json.dumps(v,ensure_ascii=False)+',',updated,count=1,flags=re.M)
    assert n==1
insert='\n'.join('  '+s for s in json.dumps({stage_key:stage},ensure_ascii=False,indent=2)[2:-2].splitlines())
anchor='  "stage_results": {\n';assert updated.count(anchor)==1
updated=updated.replace(anchor,anchor+insert+',\n',1)
expected={**before,**scalars,'stage_results':{stage_key:stage,**before['stage_results']}}
assert json.loads(updated)==expected

current=f'''### {STAMP} 최우선 최신 상태 — 신청·문의 중단, 로컬 사과표현12명 입력 준비

- 사용자 지시로 데이터 신청·사용조건 동의·제공자 문의를 하지 않는다. AI Hub636은 로그인 후 신청/필수동의 화면이 확인돼 중단했다. 공개 샘플 ZIP 저장은 미검증, 실제 목표 포함 여부 UNKNOWN이다. 신청·동의·문의 제출0이며 이 경로 재시도를 다음 작업으로 두지 않는다.
- 기존 문장 라벨 REAL01 정면2,000개에서 관련14문장을 찾고 촬영자01~16의7목표 후보221라벨행을 정리했다. 새로 활용 가능한 사과표현 SEN0356(미안합니다15·죄송하다1)은 기존80클래스 manifest에 없었다. REAL01 원본1개4,637,346바이트를 로컬 ZIP에서 확보·CRC/SHA·131프레임 디코드 검증했다. 의미 승인0.
- 보유 관절 ZIP에서 학습후보01~09·개발11~13의12클립/417프레임을 기존 리더/280채널 어댑터로 별도 준비했다. CRC·스키마·유한값·마스크·시간차·저장 재읽기 PASS. 캐시457,444바이트,9.655초,6개 pin 및 원본12개 stat 전후 동일. 미배정10·보류14~16 배열 읽기0.
- 제공 관절은 OpenPose이고 실제 웹캠은 MediaPipe다. 입력수치 PASS는 검출방식 일치나 인식성능 PASS가 아니다. frame index/명목30fps는 실제 PTS/시작번호 정렬 미검증이며 의미 동등성·완전한 행동경계도 미확정이다. 새 학습원 등록·학습·분류추론·모델교체0.
- 18목표 확보표를 갱신했다. 기존 WORD원본254개 ZIP매핑 미해결, KSL36 UNASSIGNED/34미디어 미검토 유지. 사용자기록 비학습·보류시험 보존. production변경·삭제·백업·Unity빌드·예약작업0. 최초 감사 경로 오류는 별도 보존 후 감사스크립트만 수정했다.
- 근거: `research/front_keyphase_v1/evidence/sentence_target_catalog_20260919/RESULT.md`, `apology_native_prep.json`, `COVERAGE_18.md`.

'''
next_section=f'''### {STAMP} 다음 단일 작업 — 새 사과표현 원본의 웹 입력과 제공 관절 정렬 확인

- SEN0356 REAL01 정면 원본1개를 기존 MediaPipe→웹280채널 경로로 별도 준비하고, 실제 PTS와 제공 관절 frame index/명목30fps의 정렬을 먼저 확인한다. 전체 영상과 라벨 구간을 구분하고 정렬이 미확정이면 검출 방식 차이로 단정하지 않는다. 기존 WORD/KSL 통과 검증은 반복하지 않는다.
- 12명분 native 후보는 학습 미승인이다. 원래 confidence·결측·출처·형태를 보존하며 의미 병합·단어별 예외·임계값 조정으로 성공을 만들지 않는다. 새로운 분류 성능/새 사람 일반화는 아직 미검증이다.
- 데이터 신청·약관 동의·문의 금지. AI Hub636 재시도 대신 보유 원본으로 진행한다. 기존18목표·WORD254개 ZIP매핑·10미배정/14~16보류 메타데이터 전용·공식Validation 비학습·KSL34미디어 미검토·사용자기록 비학습을 유지한다. Claude Code production 작성/현재환경 실행·독립검증 역할 유지. 새 학습·기본모델 교체·production변경·삭제·백업·Unity빌드·예약작업은 선행하지 않는다.

'''
context_path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
context,ce,cn=read_text(context_path)
context,n=re.subn(r'^최종 갱신: .*$', '최종 갱신: '+STAMP,context,count=1,flags=re.M);assert n==1
for marker,value in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n',current),('## 9. 다음 단일 작업\n\n',next_section)]:
    assert context.count(marker)==1;context=context.replace(marker,marker+value,1)
handover_path=ROOT/'research/HANDOVER_2026-09-18.md'
handover,he,hn=read_text(handover_path);marker='# WordsInSilence — 새 대화 인계\n\n';assert handover.startswith(marker)
handover=handover.replace(marker,marker+f'## 후속 갱신 — {STAMP}\n\n'+current.split('\n\n',1)[1]+next_section,1)
for p,data,e,newline in [(state_path,updated,enc,nl),(context_path,context,ce,cn),(handover_path,handover,he,hn)]:
    p.write_bytes(data.replace('\n',newline).encode(e))
assert load(state_path)==expected
assert read_text(context_path)[0]==context and read_text(handover_path)[0]==handover
dump('document_validation.json',{'status':'PASS','at':STAMP,'prior_state_fields_and_history_preserved':True,
     'latest_sections_match':True,'user_constraint_persisted':True,
     'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [state_path,context_path,handover_path]}})
print(json.dumps({'status':'PASS','at':STAMP,'stage':stage_key,'target_rows':len(ledger),'native_clips':12},ensure_ascii=False))
