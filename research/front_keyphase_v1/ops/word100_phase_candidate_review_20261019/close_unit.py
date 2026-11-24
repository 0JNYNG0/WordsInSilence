from common import *
import re
s=read(U/'SCOPE.json');v=read(U/'INDEPENDENT.json');pts=read(U/'VIDEO_PTS_CHECK.json');g=read(U/'GUARD_TESTS.json');qa=read(U/'LAYOUT_QA.json');c=read(U/'CONCLUSION.json');plan=read(U/'NEXT_PLAN.json')
assert not (U/'FINAL_AUDIT.json').exists() and not (U/'RUNNING.json').exists()
assert v['status']==pts['status']==g['status']==qa['status']=='PASS'
assert v['catalog_sha256']==sha(U/'CATALOG.json') and v['verifier_sha256']==sha(U/'verify.py')
assert v['spans']==315 and v['images_verified']==945 and qa['video_metadata_count']==100
assert c['expert_reviewed_contracts']==0 and c['training_runs']==c['new_model_inference']==c['user_recordings_read']==0
assert read(U/'VISUAL_REVIEW.json')['frames_viewed']==111
assert not plan['training_authorized_now'] and not plan['automatic_adoption']
assert sha(U/'before/verify_initial.py')==read(U/'FIRST_VERIFY_ERROR.json')['verifier_sha256']
check(s['pins'])
write(U/'RECOVERY_NOTES.json',dict(at=now(),status='RESOLVED_TECHNICAL_VERIFICATION_ISSUES',
 timestamp_check=dict(first_error='FIRST_VERIFY_ERROR.json',cause='Initial verifier used frame_id / container average FPS as a per-frame timestamp;16container average rates differ from the actual frame presentation times.',
  evidence='VIDEO_PTS_CHECK.json:945actual decoded-frame timestamps match saved observation timestamps within0.000000334s.',
  recovery='Changed verifier to actual presentation timestamps, without changing source observations, candidate times or images.',
  initial_verifier='before/verify_initial.py reconstructed byte-exact against first-error SHA; initial newline mismatch during reconstruction was corrected.'),
 export_check=dict(first_error='FIRST_UI_QA_ERROR.json',cause='Headless Chrome native download was canceled; reason not established.',
  recovery='Visible full JSON generation and select/copy fallback added and tested, along with local draft persistence and100source-bound records. Optional native file-save button remains not verified.',
  qa_export='QA_EXPORT_NOT_HUMAN_LABEL.json contains synthetic browser test opinions only; never semantic ground truth.'),
 scientific_result_changed=False,models_changed=False))
with (U/'RESULT_KO.txt').open('a',encoding='utf8') as f:
 f.write('\n검증 중 정정:16개 파일에서 평균FPS로 계산한 시간이 실제 프레임시각과 약0.01초 달라 최초 검산이 실패했다. 실제945장 프레임 재생시각을 직접 확인하니 저장시각과 일치하여 검산식만 정정했다. 관측·장면·후보시각은 바꾸지 않았다. 최초 오류 보존. 브라우저 자동 다운로드는 테스트에서 취소되어 원인 미확정이며, JSON 전체를 화면에 생성·선택·복사하는 경로를 추가 검증했다. 선택형 파일 저장 버튼의 실제 다운로드는 미검증이다.\n')
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in docs:assert sha(p)==sha(U/'before'/p.name),'Concurrent document change: '+str(p)
stamp=now();key=U.name
heading='### 2026-10-19 공개100단어 단계 후보·검토 화면 완료 — 필수/선택·반복 의미 확인 대기'
next_action='100단어 단계 후보 검토 화면을 사용자와 확인한다. 자동315구간을 의미 단계로 확정하지 말고, 구간 합치기/나누기·필요한 손을 검토한 다음 한국수어를 아는 검토자에게 필수/선택·선후·반복 허용 의견을 받을 소규모 검토를 준비한다. 이후 다른 공개 TRAIN 촬영자 정상 변형을 확인하고 이견/가림은 보류한다. 추가 학습은 결과 함께 확인한 뒤 별도 계획으로 진행; C·T3-F·웹·DTW 유지, 사용자·멘토 촬영 학습/튜닝/선택 금지.'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 사용자 승인 공개100단어 동작 단계 후보 표시·검토자료 준비 완료. 추가학습0/모델추론0/개인·멘토접근0/ID14~18원시접근0. C·T3-F·웹·DTW·원본 보존.
- 미리 고정한 공개TRAIN ID01 참고100영상. 위치변화139+손가락굽힘정보변화76=215책갈피,연속검토315구간,시작/중간/끝945장,100모음그림. 내부표시95영상/표시없음5영상. 관절근거완전한전후창만비교,가림을가로질러변화계산하지않음. 이름/모델점수/시간3등분 미사용. 최대5표시·최소.30초간격은 검토예산이며 수어오류기준아님.
- 전체원본재생·구간앞뒤재생·좌우21점관측부족띠·후보장면·필수/선택·필요손·선후·반복·경계수정의견·브라우저임시저장·JSON생성/선택복사 화면구현. 직접파일저장은초기headless취소로미검증,복사경로검증완료. QA_EXPORT는합성시험의견으로실제수어검토아님.
- 모든100영상연결/주석4921프레임구간누락없음/945장이미지/215변화량별도검산PASS(최대{v['max_numeric_error']:.3g}). 3618보호파일해시유지. 같은Codex별도산술이며독립agent/수어전문가아님. Chrome100원본metadata·구간이동·의견저장/복원/JSON·desktop/mobile확인PASS.
- 새시각검토12영상111표본. 냉커피양손→입근처한손→가슴앞변화관찰;믿다2구간·수영4구간은하나의연속동작을과분할했을가능성. 성남/전주후반위치유사하나도시구분단계미확정. 기존100영상관찰메모출처표시재사용,새100경계전부수동확정아님. 다른사람변형/허용반복미검증,전문가확정단계규칙0개.
- 초기검산평균FPS가정16파일에서약.01초어긋남→실제945장PTS직접확인저장시각일치(최대3.34e-7초),검산식만정정. 최초오류/해시정확초기검산코드/RECOVERY_NOTES보존. 입력/모델결함으로오해금지. 관측/장면/후보시각변경없음.
- RESULT_KO.html/txt,CATALOG,records,frames,sheets,VISUAL_REVIEW,INDEPENDENT,VIDEO_PTS_CHECK,GUARD_TESTS,LAYOUT_QA,CONCLUSION,NEXT_PLAN,FINAL_AUDIT. 학습성능향상아닌평가기준준비. 필수/선택/허용반복은모두미확정,검토의견은자동학습정답으로사용하지않음. 실행20074/12762/1005종료0,진행중없음.

'''
text=docs[0].read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in text and h9 in text and heading not in text
text=text.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_STAGE_EVALUATION_COMPLETE_SEMANTIC_REVIEW_PENDING'
state.update(current_stage='WORD100_PHASE_CANDIDATE_REVIEW_READY_SEMANTICS_PENDING',updated_at=stamp,updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]=dict(status=c['status'],at=stamp,unit=key,public_words=100,public_clips=100,source_signer_ids=[1],candidate_marks=215,candidate_spans=315,images=945,
 new_visual_review_clips=12,new_visual_review_frames=111,expert_reviewed_contracts=0,training_runs=0,new_model_inference=0,user_recordings_read=0,heldout_raw_read=0,
 existing_models_preserved=True,production_changed=False,independent_scalar_checks=215,max_numeric_error=v['max_numeric_error'],protected_files=3618,
 first_errors_preserved=True,native_file_download_verified=False,json_copy_export_verified=True,running_processes=[],automatic_training=False,
 report=str(U/'RESULT_KO.html'),next_plan=str(U/'NEXT_PLAN.json'))
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 공개100단어 영상의 관측 가능한 단계 후보를 표시하고 필수/선택·반복 검토를 준비. 이번 검토 화면/기술검증 완료, 수어 의미 확인 및 추가 학습은 사용자와 결과 검토 후다.

- 현재단위 research/front_keyphase_v1/ops/{key}. SCOPE/CATALOG/records/frames/sheets/VISUAL_REVIEW/INDEPENDENT/VIDEO_PTS_CHECK/GUARD_TESTS/LAYOUT_QA/CONCLUSION/NEXT_PLAN/RECOVERY_NOTES/RESULT_KO.html/txt/FINAL_AUDIT.
- 공개TRAIN 고정 ID01 영상100개만. 215변화책갈피(위치139·굽힘76),315자동관찰구간,945장(구간시작/중간/끝),100모음그림. 내부표시없는5영상은전체주석1검토구간,의미1단계확정아님. 다른사람799영상은이번에새분석안함.
- .133333초전후창동일손21점+어깨확보시에만위치/굽힘정보변화를계산. 손관절근거부족은별도띠로표시,미검출≠동작생략. 최대내부5표시/간격.30초는검토화면예산이고수어오류기준아님. 표정·빠른단계·미세손모양을모두찾는다고보장안함. 모델점수/단어명에맞춤/고정3등분없음.
- 시간·원본연결100/주석4921프레임/이미지945/변화량215별도코드검산PASS. 최대산술차{v['max_numeric_error']:.3g},3618보호파일유지. 원본metadata100개/구간재생/로컬의견복원/100단어JSON생성·전체선택/desktop1360·mobile390확인. 같은Codex별도산술이며독립agent/전문가검토아님.
- 새시각검토12영상111표본. 냉커피양손→입근처한손→가슴앞관찰,믿다2구간/수영4구간은연속동작과분할가능. 성남/전주후반유사하나구별의미단계미확정. 기존100개표본메모재사용출처표시. 새100개경계전부수동검토/전체동영상전문가판정아님. 확정필수/선택/선후/허용반복규칙0개.
- 검토UI는필수/선택/준비·연결·마무리/합치기,필요손,순서·반복·시간수정의견을입력. JSON은검토의견이며학습정답자동승인금지. QA_EXPORT_NOT_HUMAN_LABEL은자동브라우저합성시험값이고실제의견아님. 임시저장만믿지말고전달할JSON보관.
- 초기검산이평균FPS시간가정으로실패(16파일약.01초). 실제945장decodedPTS와저장시각일치(3.34e-7초),검산식만수정. 원자료/장면/후보변경없음. FIRST_VERIFY_ERROR·before/verify_initial.py(최초SHA일치)·TIMESTAMP_DIAGNOSIS·RECOVERY_NOTES보존.
- 초기Chrome native download 취소,원인미확정. FIRST_UI_QA_ERROR보존. UI에JSON본문생성·전체선택/복사추가검증. 선택형파일저장버튼실제다운로드미검증. 연구결과/모델오류아님.
- 기존C/T3-F/웹/DTW/원본/과거동결파일보존. 학습0/새모델추론0/개인·멘토촬영읽기0/ID14~18원시0. Git/카메라/다운로드/외부전송없음. 완료산출물재계산불필요. 실행20074/12762/1005종료0,현재진행중없음.
- 다음: 사용자와후보화면확인→구간합치기/나누기·필요손관측검토→KSL검토자의필수/선택/선후/반복의견→다른공개TRAIN ID02~09정상변형확인→이견/가림보류→정답기준동결/검증후추가학습계획. 범용방법목표이나현재100단어·1명참고범위를넘는정확도보장없음.

다음 한 작업: {next_action}
'''
docs[0].write_text(text,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for p in U.glob('*.json'):read(p)
for p in docs:assert key in p.read_text(encoding='utf8')
check(s['pins'])
write(U/'FINAL_AUDIT.json',dict(status='PASS_REVIEW_ARTIFACT_READY_SEMANTICS_PENDING',at=stamp,protected_files=len(s['pins']),
 input_clips=100,candidate_spans=315,candidate_marks=215,images=945,source_signer_ids=[1],new_visual_review_clips=12,new_visual_review_frames=111,expert_reviewed_contracts=0,
 training_runs=0,new_model_inference=0,user_recordings_read=0,heldout_raw_read=0,production_changed=False,running=False,
 implementation='Codex local',verification='Same Codex independent scalar code and browser tests; not independent agent or KSL expert.',
 numeric_pass=True,video_pts_pass=True,missing_guards_pass=True,browser_100_videos_pass=True,json_visible_copy_export_pass=True,native_download_verified=False,
 initial_verifier_failure_preserved=True,initial_ui_download_failure_preserved=True,
 doc_hashes={str(p):sha(p)for p in docs},files={p.name:sha(p)for p in U.iterdir()if p.is_file()}))
print('COMPLETE:100public clips,315review spans; semantics pending; C/T3F unchanged; docs synchronized.')
