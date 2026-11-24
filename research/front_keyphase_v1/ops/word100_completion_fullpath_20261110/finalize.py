from common import *
v=read(F/'VERIFICATION.json');assert v['status']=='PASS';comp=read(F/'COMPARISON.json');assert sha(F/'COMPARISON.json')==v['comparison_sha256']
diag=[]
for r in read(F/'FRESH_COMPLETE.json')['rows']:
 if r['kind']!='full_normal':continue
 d=read(r['result'])
 for c in d['cards']:
  if c['display_status']!='uncertain':continue
  diag.append(dict(clip_id=r['clip_id'],card=c['id'],query_indices=c['query_indices'],unknown_fraction=c.get('unknown_fraction'),checked_components=c.get('checked_components'),unverified_components=c.get('unverified_components',[]),source_applicability=[x.get('source_applicability') for x in c.get('coverage_requirements',[])],completion_conflict=any(x['conflict'] for x in c['completion_identity_audit']),cause='No trained/validated component evidence for this relation' if not c.get('checked_components') else 'Observed open-hand evidence does not validate fingertip convergence; source variant unresolved',not_detector_loss=c.get('unknown_fraction')==0,action_taken='No evaluation-dependent change; remain explicitly uncertain'))
write(F/'REMAINING_CAUSE_AUDIT.json',dict(rows=diag,method='Read frozen saved result fields, no scoring or retraining',limitation='A high stage probability locates a candidate phase, not proof that every required relation/shape is correct'))
nextplan=dict(objective='Teach/check observable components separately from event presence, using existing public FIT only',steps=[
 'Audit existing FIT01-09 labels across expressions for hand convergence/separation, visible finger flexion, palm orientation, and contact; separate source-description variant conflicts from sensor failure. Do not label fingertip contact from wrist distance.',
 'Prepare observable positive/negative and pre-core/core/optional-arrival labels with raw RGB provenance. Use explicitly uncertain labels when hidden; no target word filling of an absent event. Focus review on existing ambiguous FIT boundaries rather than relabeling DEV.',
 'Freeze a small component/progress-head comparison before training; keep current joint observation backbone and C/T3F/TCN intact. Train only if FIT support covers multiple recorded IDs and sufficient contrasting examples; otherwise report the exact missing labels without pretending a model fix.',
 'Check normal/speed/crossing and partial-action/sensor-loss separately. ID10 is now exposed and may only be a regression set, not a new independent validation set. Only then consider an explicitly selected research web candidate; default replacement still requires result review.'
],no_rerun=['Completed E six trainings and 237 comparisons','Completed F raw60/fresh20 neural outputs and extraction519','Completed saved-output verification without a new issue'],not_a_claim='This unit completes full raw-path integration validation, not all educational feedback or all100-word accuracy',constraints=['No private/mentor access, training, tuning or selection','No external download/transmission/camera/Git','No tuning on recorded ID10 or prior DEV','No word-specific exception or forced success'])
write(F/'NEXT_PLAN.json',nextplan)
report='''공개 자료의 실제 분석 경로 연결·새 촬영 비교 완료 — 2026-11-10

이번에 한 일
지난번 만든 보완을 실제 분석 코드에 격리 연결했습니다.
RTMW 관절 → 불안정한 관절 제외 → 기존 TCN·피드백 → 다른 동작을 잘못 연결했는지 확인 → DTW 재생 정보 → 마지막 관측 확인까지 모두 통과시켰습니다.
앞 동작의 근거만 있는데 뒤 동작까지 잘했다고 표시하지 않도록 하는 보조 검사입니다. 기존 손모양·방향·시간 정렬 계산은 유지했습니다.
현재 사용 중인 웹8814의 기본 모델이나 코드를 교체하지는 않았습니다.

이전 → 현재 (같은 과거 공개 입력을 실제 전체 분석 경로로 비교)
- 정상 원본12개: 전체 필수 특징 비슷함9/보류3 → 9/3, 차이 경고0 유지.
- 속도 변경24조건: 기존 결과 유지. 조상 양손 교차도 유지.
- 마지막 동작을 지운6조건: 잘못된 전체 비슷함2 → 0. 두 경우 모두 확인 보류로 바뀌었고, 정확한 교정 문장을 완성한 것은 아닙니다.
- 마지막 동작의 손 관절만 숨긴6조건: 잘못된 전체 비슷함2 → 0. 관절이 안 보인다는 이유로 실제 행동을 생략했다고 단정하지 않았습니다.
- 전체 손 관절을 숨긴12조건: 전체 비슷함0 유지.
이60조건은 이미 사용한 개발자료에서 만든 조건입니다. 지난 개선이 실제 처리 경로에서도 유지되는지 확인한 것이며 새로운 독립 정확도 증명은 아닙니다.

새 공개 촬영에서 확인한 결과
현재 학습·개발 입력 목록 및 확인한 이전 공개평가 목록에 없는 기록ID10의4영상을 사용했습니다.
동일4단어×기록ID01~18의72파일에서 바이트 중복을 확인했습니다. 외형으로 동일인/다른 사람을 판정하지 않았습니다.
프로젝트의 모든 과거 시각 노출까지 없었다는 증명은 아니며, 새로운 한 기록ID의 작은 보충 시험입니다.
- 정상4개: 조상·순종·잘하다는 특징 비슷함, 이집트는 확인 보류. TCN1위는4개 모두 해당 단어였습니다.
- 느리게/빠르게 만든8조건: 정상 때와 같은 결과, 추가 차이 경고0. 양손 교차 포함.
- 마지막 동작을 지운2조건: 둘 다 전체 비슷함으로 통과하지 않음.
- 마지막 손 관절만 숨긴2조건 및 전체 손 관절을 숨긴4조건: 둘 다 실제 누락/오답으로 단정하지 않고 확인 보류.
새20조건에서는 기존 방식과 보완 방식의 결과가 같았습니다. 새 자료에서도 추가 성능 향상이 입증됐다고 말할 수는 없습니다.
전체는 공개 원본16개에서 만든80조건입니다. 서로 독립적인80명/80촬영이 아닙니다.

이집트는 왜 보류했나
이번 새 영상은 앞·뒤 주요 구간을 찾았고, 그 구간의 손 관절도 들어왔습니다.
그런데 첫 구간은 '양손 손끝 관계'를 확정할 학습·판정 근거가 아직 없고,
뒤 구간은 손을 편 근거만으로 마지막 손끝 모양까지 확인했다고 할 수 없습니다.
즉, 이번 보류는 손이 전부 사라져서가 아니라 아직 가르치고 검증한 세부 특징이 부족해서입니다.
새 평가 결과에 맞춰 기준을 느슨하게 바꾸지는 않았습니다.

처리시간
관절 값이 이미 준비된 뒤 TCN·피드백·DTW·보완 검사까지의 계산 중앙값은 과거60조건0.0525초, 새20조건0.0510초였습니다.
추가 보완 검사 자체는 약0.0105~0.0106초였습니다. 웹캠 촬영·관절 추출·화면 표시 대기시간과는 다릅니다.
새 원본4영상519프레임의 RTMW 추출은 기존 GPU.0 환경에서 한 번만 수행했고 영상당4.85~5.56초였습니다(파일 읽기/추출/저장 포함).
GPU모델 초기 준비5.64초+예열5.25초는 별도입니다. 이 파일 시험으로 실제 촬영 종료 후 대기시간이0.05초라고 주장하지 않습니다.

남은 문제와 다음 작업
1. 빠진 동작을 정상 완료로 넘기는 일부 오류를 막았지만, 아직 생략과 관측 부족을 구체적인 교정 안내로 모두 구별하지는 못합니다.
2. 손끝 모으기·엄지·정확한 접촉·반복 횟수 등은 별도 특징 정답과 설명 변이 정리가 남아 있습니다.
3. 생략 시험은 공개 영상의 일부를 제거한 합성 대조입니다. 실제 학습자의 자연스러운 실수에 대한 교육적 정확도는 미검증입니다.
다음에는 기존 공개 학습자료에서 여러 표현·기록ID의 보이는 손끝/굽힘/양손 관계 정답을 먼저 보강합니다. 구간을 찾는 능력과 구간 안에서 어떤 모양·관계인지 판단하는 능력을 나눠 연결합니다.
이번 ID10과 기존 개발자료로 학습/문턱/후보를 조정하지 않습니다. ID10은 이제 사용된 평가자료로 기록했습니다.

검증·보존
별도 저장출력 산술/프레임/변경 범위/해시 검사2343항목 통과. 같은Codex의 별도 검산이며 독립 전문가 정답은 아닙니다.
실행환경과 결과 기록의 오류는 최초 로그를 보존하고 복구했습니다. 이미 완료한 추론은 다시 돌리지 않았고,80조건에서 실제 추론 실패/분모 제외는0입니다.
새 학습0, 사용자·멘토 자료 접근0, 카메라0, 다운로드0. 기존C·T3-F·현재TCN·기본웹·원본 보존.
새 자료의 추가 정확도 향상은 확인되지 않아 새HTML은 만들지 않았습니다.

근거: COMPARISON.json / VERIFICATION.json / REPLAY_RECOVERED.json / FRESH_COMPLETE.json / LINEAGE_AUDIT.json / REMAINING_CAUSE_AUDIT.json / NEXT_PLAN.json
'''
(F/'RESULT_KO.txt').write_text(report,encoding='utf-8')
brief='공개RTMW→기존TCN/피드백→완료충돌검사→DTW→최종관측확인의 실제 전체경로 검증 완료. 과거60조건: 정상12 비슷함9/보류3·속도24·교차유지, 사건삭제6/관절가림6의 거짓전체비슷함 각각2→0(보류). 새 기록ID10 4영상519프레임을 GPU.0로1회추출,20고정조건 비교: 정상3비슷함/이집트1보류·속도8유지·생략2/마스크6 정상완료0, 기존대비추가이득0. ID10은감사한현재학습계보/기존평가에없었으나1기록ID/작은합성시험;이제노출된회귀자료. 이집트는관절누락이아닌손끝관계근거미완. 전체분석계산 중앙0.051~0.053초(관절추출/카메라/UI제외);추출영상당4.85~5.56초. 저장검산2343PASS(같은Codex). 완료추론재실행0/학습0/개인자료0/기본웹교체0.'
nexttext='완료F60+20추론/519추출/저장검산 및E6학습/237조건 재실행금지. 다음은공개FIT01~09의여러표현에서보이는손끝모임·굽힘·양손관계와진행/핵심/선택적도착정답보강;설명변이/관측불가/모델기능미완분리. FIT내사람분리로조건고정후충분한대조근거가있는출력부만학습검토. ID10·DEV사후튜닝금지,ID10은이제노출회귀자료. 새결과에맞춘단어예외/기준완화금지. 명시연구웹후보는전체회귀후검토;기본8814/C·T3F/현재TCN유지,개인자료·카메라·다운로드없음. 상세 word100_completion_fullpath_20261110/NEXT_PLAN.json.'
write(F/'RESULT.json',dict(status='COMPLETE_FULLPATH_VERIFIED_NO_INCREMENTAL_FRESH_GAIN',summary=brief,comparison=str(F/'COMPARISON.json'),verification=str(F/'VERIFICATION.json'),report=str(F/'RESULT_KO.txt'),remaining=str(F/'REMAINING_CAUSE_AUDIT.json'),trained=False,default_web_changed=False,private_access=False,core_model_modified=False,fresh_population_claim=False))
write(F/'CURRENT_EXECUTION.json',dict(status='COMPLETE_VERIFIED',active_process=None,training=False,extraction_complete=True,neural_calls=80,saved_checks=2343,web_changed=False))
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig');title='### 2026-11-10 완료 보조 판정의 실제 분석 경로·새 공개 촬영 검증'
assert title not in text
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',brief+' 근거 research/front_keyphase_v1/ops/'+F.name+'/RESULT_KO.txt.'),('## 9. 다음 단일 작업',nexttext)]:text=text.replace(header,header+'\n\n'+title+'\n\n'+body,1)
context.write_text(text,encoding='utf-8')
state=read(O/'STATE.json');state['current_stage']=F.name;state['last_updated']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();state['next_single_action']=nexttext;state[F.name]=dict(status='COMPLETE_FULLPATH_VERIFIED_NO_INCREMENTAL_FRESH_GAIN',summary=brief,next=nexttext,report=str(F/'RESULT_KO.txt'),result=str(F/'RESULT.json'),verification=str(F/'VERIFICATION.json'),execution=str(F/'CURRENT_EXECUTION.json'),active=None,default_web_changed=False,training=False,private_access=False);write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+brief+'\n\n다음 한 작업: '+nexttext+'\n\n추론·추출·검산 모두 종료. 기본웹8814와 C·T3-F·현재TCN은 변경하지 않았다. 새학습/서버기동/카메라/개인자료 접근 없음. 실제 consumer.raw를 격리 인스턴스에서 검증했지만 브라우저/실사용자 종단시험 완료는 아니다. 기록ID10의 네 원본은현재모델계보미사용확인,전체역사적노출없음/생체동일인분리증명은아님. 독립검증은같은Codex별도산술코드이며독립전문가정답아님. 최초오류와복구기록은F폴더에보존.\n\n결과 '+str(F/'RESULT_KO.txt')+'\n검증 '+str(F/'VERIFICATION.json')+'\n상태 '+str(F/'CURRENT_EXECUTION.json')+'\n',encoding='utf-8')
print('FINALIZED',F.name)
