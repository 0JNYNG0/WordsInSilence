"""Audit readiness from actual saved metadata and close this local preparation unit."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib

U=Path(__file__).resolve().parent;B=U.parents[1];R=B.parents[1]
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

assert not (U/'RESULT.json').exists(),'Already closed'
build=read(U/'BUILD_RESULT.json');features=read(U/'FEATURE_CHECK.json');browser=read(U/'BROWSER_CHECK.json');ui=read(U/'UI_REGRESSION.json')
assert all(x['status']=='PASS' for x in [build,features,browser,ui])
assert sha(build['manifest_path'])==build['manifest_sha256']
for file,key in [('app.js','app_sha256'),('index.html','index_sha256'),('style.css','style_sha256')]:
    assert sha(B/'web'/file)==browser[key]
manifest=read(Path(build['manifest_path']))
assert all(sha(s['feature_path'])==s['feature_sha256'] and sha(s['source_cache_path'])==s['source_cache_sha256'] for s in manifest)
assert sha(B/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt')=='83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'

people=Counter(str(s['signer_metadata_only']) for s in manifest)
readiness={
 'status':'NOT_READY_FOR_NEW_PERSON_FUSION_ACCURACY_EXPERIMENT',
 'basis':'Actual saved input-index metadata, not assumed Modu release data',
 'segments':len(manifest),'unique_source_clips':len({s['clip_id_metadata_only'] for s in manifest}),
 'unique_labels':len({s['label_metadata_only'] for s in manifest}),'segments_per_known_person':dict(people),
 'semantic_alignment_verified_records':sum(s['semantic_alignment_verified'] is True for s in manifest),
 'split_labels':dict(Counter(s['split'] for s in manifest)),
 'gates':{
  'fixed_encoder_and_full_cached_features':'PASS',
  'pose_rgb_time_side_alignment':'PASS_TECHNICAL_ALIGNMENT_ONLY',
  'word_form_and_boundary_semantic_review':'NOT_RECORDED_AS_COMPLETE',
  'paired_rgb_for_people_unseen_by_parent_tcn':'MISSING_IN_CURRENT_PREPARED_DATA',
  'person_and_original_video_separated_train_dev_test':'NOT_CREATED',
  'frozen_training_and_acceptance_policy':'PENDING_ACTUAL_DATA',
  'modu_release_received':'NOT_RECEIVED_PER_LATEST_USER_REPORT'},
 'next_required_files':['Actual release guide and annotation sample','Video listing and person-ID scope documentation','Then corresponding authorized videos'],
 'do_not_infer':['01/02 split is new-person evaluation for pretrained TCN832','Correct filename timing proves the sign meaning','This113 fixture is a received Modu release','Technical feature parity is recognition improvement']}
write_new(U/'DATA_READINESS.json',readiness)
write_new(U/'VISUAL_CHECK.json',{'status':'PASS','reviewer':'Codex visual inspection of actual Chrome screenshots',
 'screenshots':['web_explicit_model_desktop_final.png','web_explicit_model_narrow_final.png'],
 'viewports':[1280,900], 'observed':'Checkbox caption, separate model selector and chosen ID legible; no clipping in selected-model control',
 'limit':'Camera preview is intentionally empty; session interactions mocked, no live inference'})

next_action='Receive the actual Modu release guide, annotation sample and video/person listing; verify semantic word boundaries and person identity scope, then freeze a person-separated A/B/C fusion training comparison. Local features and actual browser checks are complete; no role approval pending, no prepared-data rerun or accuracy claim.'
result={'status':'COMPLETE_EXISTING_DATA_FEATURE_PREPARATION_AND_BROWSER_VERIFICATION', 'updated_at_kst':stamp,
 'artifact_root':str(U.relative_to(R)).replace('\\','/'),'output_root':str((B/'runs/rgb_features_v1').relative_to(R)).replace('\\','/'),
 'report':'RESULT_KO.txt','author':'Codex local implementation and separate-path full verification',
 'counts':build['counts'],'feature_bytes':build['bytes'],'feature_dimension_per_hand':576,
 'independent_feature_max_abs_error':features['max_feature_abs_error'],'frozen_tolerance':features['tolerance_frozen_before_run'],
 'browser':'Actual installed headless Chrome; all HTTP mocked, no camera or real service requests',
 'browser_checks':len(browser['checks']),'existing_ui_assertions':ui['assertions'],'visual_checked_widths':[1280,900],
 'code_changes':['Isolated feature producer/loader/independent verifier','Web research control layout and asset versions only'],
 'preserved':'Frozen source caches/metadata and TCN832. No training, labels as inputs, user data, new data download, default replacement or Git change.',
 'first_failure':'Browser test tried a button in collapsed details. Evidence preserved; test opens panel before click. No product recognition defect inferred.',
 'readiness':readiness['status'],'next':next_action,'active_processes':[]}
write_new(U/'RESULT.json',result)

p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
updates={
'## 3. 운영 역할':'''\n\n- 2026-10-10 사용자 “그럼 작업 완료할때까지 멈추지 말고 계속 진행해.”에 따라 동일 로컬 승인 역할로 기존 자료의 전체 RGB 특징 준비와 실제 브라우저 검증을 완료했다(`ops/rgb_features_and_browser_20261010`). 수어 학습·새 자료 다운로드·카메라·운영 교체는 하지 않았다. 역할 질문 없이 진행했다.\n''',
'## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인':'''\n\n### 2026-10-10 RGB 특징113구간 전수 준비·실제 웹 화면 검증 완료

최신 `ops/rgb_features_and_browser_20261010/RESULT_KO.txt`·`RESULT.json`. 준비된105영상/113구간/69단어의 정상5315+추정78=5393사진을 손당576차원 특징으로 저장했다(`runs/rgb_features_v1/MANIFEST.json`, 약13MiB). 기존2789행/좌우/시각/280관절/상자/관측수 보존,185사진누락 유지. A관절만/B정상RGB/C정상+추정영역RGB 입력을 제공하며 라벨·사람ID는 입력 밖 메타데이터다. 학습/단어정답 채점 없음.

별도 공식모델 경로로5393사진 전수검산 PASS, 최대특징오차0.000004529953(사전허용0.00001). A/B/C 관절·시각·정상관측특징 동일성/전수메타데이터/오류거절5검사 PASS. 원입력/모델/색인 보존. 이는 특징 저장 검증이며 RGB 인식률 향상이 아니다.

이전 미완료였던 실제 화면은 설치된 headless Chrome으로 확인했다. 모든HTTP모의/카메라·실제서버접근0. 선택/잠금/재생오류복구/모의촬영취소 등6항목·기존 UI38확인 PASS,1280/900폭 화면육안확인 완료. 긴 모델명에 밀리던 설명을 선택줄과 분리하고 JS/CSS버전을 갱신했다. 첫 검사에서 접힌 재생패널 버튼 클릭이 시간초과한 기록을 보존하고 검사순서만 정정했다. 실제촬영 검증으로 확대하지 않는다.

현재 자료의 사람01/02는 기존TCN학습사람이며113구간의 의미/경계 검토 완료기록은0이다(틀렸다는 판정이 아님). DATA_READINESS.json에 조건을 명시했다. 새사람 대응자료/의미검토/사람분리/학습계약은 미충족이며 자료 수신 전 임의로 만들지 않는다. 현재 준비단위 완료/모든 실행 종료.\n''',
'## 9. 다음 단일 작업':'''\n\n### 2026-10-10 다음 — 실제 배포 자료 수신 후 단어·사람·영상 연결 감사

기존자료 RGB 특징 준비와 실제 브라우저 표시 검증은 완료했다. `ops/rgb_features_and_browser_20261010/DATA_READINESS.json`을 재개점으로 쓴다. 다시 추출/준비/과거실험을 반복하지 않는다. 역할 승인이나 브라우저 도구 복구를 기다리는 상태도 아니다.

모두의 말뭉치 실제 배포 설명서·주석 샘플·영상 목록을 받으면 단어·시각·사람ID 범위·원영상 연결을 검증하고 의미상 동작 구간을 확인한다. 그 뒤 기존TCN 사전학습 인물까지 고려해 학습/개발/평가를 분리하고 A관절만/B정상RGB/C공백실제RGB의 비교학습 계약을 동결한다. 현재 사람01/02만 분리해 새사람 평가라 하지 않는다. 사용자 본인 인증·신청/약정은 사용자 단계이며 아직 미확보다. 새자료 없이 신뢰할 만한 RGB 일반화 성능 수치를 만들 수 없다. 학습·튜닝·운영교체 자동 시작 없음.\n'''}
for head,block in updates.items():
 assert text.count(head)==1
 text=text.replace(head,head+block,1)
p.write_text(text,encoding='utf-8')

p=B/'ops/STATE.json';state=read(p)
state['current_stage']='RGB_FEATURES_AND_ACTUAL_BROWSER_VERIFIED_AWAITING_REAL_PAIRED_DATA'
state['next_single_action']=next_action;state['updated_date_kst']='2026-10-10'
items=list(state.items());state=dict(items[:6]+[(U.name,result)]+items[6:])
p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

p=R/'NEW_CHAT_HANDOFF.md';handoff=p.read_text(encoding='utf-8')
start=handoff.index('## 현재 목표와 마지막 완료');end=handoff.index('## 역할')
latest='''## 현재 목표와 마지막 완료
- 목표: 여러 사람의 수어 단어 인식과 이후 구체적 피드백. 사용자 자료를 학습/튜닝/후보 선택에 쓰지 않는다.
- 최신 완료: `ops/rgb_features_and_browser_20261010/RESULT_KO.txt`·`RESULT.json`. 기존113구간의 RGB576특징 캐시 전체와 실제 웹 화면 검증 완료. 역할 질문 없이 Codex 로컬 구현·별도 계산 검증.
- 정상5315+추정78=5393사진,2789행/105원영상/69단어.185칸 누락 유지. `runs/rgb_features_v1/MANIFEST.json`과113npz 약13MiB. pose280/시각/좌우/관측수/상자 보존. 라벨/사람은 모델 입력 밖 메타데이터.
- 별도 공식 인코더 경로로5393특징 전수검산 PASS. 최대차0.000004529953≤사전허용0.00001. A/B/C 입력 보존과 오류거절5검사 PASS. 학습·단어정답률 측정0.
- 설치된 headless Chrome으로 실제 화면/선택/잠금/모의 요청·오류복구/취소6항목 PASS. 모든HTTP모의,카메라·실제서버 접근0. 기존UI38확인·1280/900폭 육안확인 PASS. 설명/목록 줄바꿈 정리,JS/CSS버전 갱신. 실제촬영은 미실행.
- 첫 브라우저 검사는 접힌 재생패널을 열지 않아 시간초과. BROWSER_FAILURE.json 보존, 검사순서 정정 후 PASS. CUA도구복구 대기는 headless검증으로 해소됐고 브라우저미검증이라는 이전현재상태는 과거다.
- 직전 `ops/rgb_pose_bridge_20261010/`:공식MobileNetV3-Small 가중치 확보, 고정TCN832에 양손RGB576+관측표시를64차원으로 더하는 격리 연결부 구현. 연결부73984파라미터0초기화로 기존점수와 같음. 수어학습 전이며 RGB를 웹 목록에 등록하지 않았다. WEIGHTS.json에 전체 해시.

## 다음 한 작업
실제 모두의 말뭉치 배포 설명서/주석 샘플/영상 목록을 받으면 단어·사람·시각·원영상 연결과 의미상 동작 경계를 확인한다. 신청·확보는 아직 되지 않았다는 것이 최신 사용자 정보다.
`ops/rgb_features_and_browser_20261010/DATA_READINESS.json`:현재 사람01/02는 기존TCN학습사람,113구간의 의미검토 완료기록0(틀렸다는 판정 아님). 새사람대응RGB/사람분리/학습계약 미충족.
자료 확인 후 A관절만/B정상사진/C공백실제사진의 같은 학습 기회·사람 분리 비교를 동결한다. 01/02끼리 나눠 새사람 효과를 주장하지 않는다. 현재 준비단위 완료이며 수어학습·RGB 성능 입증은 자료 조건이 남아 있다.
역할 승인 질문 없음. 브라우저 확인도 완료. 입력 준비/5393특징/완료 검사/과거실험을 반복하지 않는다. 현재 실행 중 명령 없음.

'''
handoff=handoff[:start]+latest+handoff[end:]
lines=handoff.splitlines();lines[1]=f'갱신: {stamp}. 현재 웹 STATE는 `research/front_keyphase_v1/ops/STATE.json`이다.'
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
write_new(U/'CLOSURE_CHECK.json',{'status':'PASS','at_kst':stamp,'hashes':{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',B/'ops/STATE.json',B/'web/index.html',B/'web/style.css']},
 'preserved_prior_results':True,'data_dependency':readiness['status'],'active_processes':[]})
print('Closed',U.name,'people',dict(people),'labels',readiness['unique_labels'])
