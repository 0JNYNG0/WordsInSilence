"""Synchronize verified preparation/repair status; never declare learning done."""
from pathlib import Path
import datetime,hashlib,json,re,shutil
U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,d,exclusive=False):
    with Path(p).open('x' if exclusive else 'w',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert (ROOT/'AGENTS.md').exists()
    assert read(U/'LABEL_AUDIT.json')['status']=='NOT_READY_SUPPORT_OR_REVIEW'
    assert read(U/'review/ROOT_REMEDIATION_VERIFICATION.json')['status'].startswith('PASS')
    independent=read(U/'review/INDEPENDENT_REMEDIATION_REVIEW.json')
    assert independent['status']=='PASS_DIAGNOSTIC_REMEDIATION_TECHNICAL_REVIEW_ORIGINAL_TRAINING_HOLD'
    for p,h in independent['source_sha256'].items():assert sha(p)==h,p
    assert (U/'review/NEXT_PROTOCOL_REVIEW.json').is_file()
    assert not any((U/n).exists() for n in ('TRAIN_FROZEN.json','TRAIN_RUNNING.json','TRAINING.json','SCORES.json'))
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    files=['AUTHORIZATION.json','PLAN_KO.txt','PREP_COMPLETED.json','DATASET.json','LABEL_AUDIT.json','DATASET_MOTION_GUARDED.json','MOTION_GUARD_AUDIT.json','MOTION_SEQUENCE_DIAGNOSIS.json','ORDERED_MOTION_CONTRACT.json','REMEDIATION_DECISION.json','EVIDENCE_LINK_AUDIT.json','RESULT_KO.txt','review/INPUT_NUMERIC_VERIFICATION.json','review/INPUT_LABEL_VERIFICATION.json','review/ROOT_REMEDIATION_VERIFICATION.json','review/INDEPENDENT_REMEDIATION_REVIEW.json','review/NEXT_PROTOCOL_REVIEW.json']
    completion=dict(status='PREPARATION_AND_CAUSAL_REPAIR_COMPLETE_TRAINING_NOT_READY',updated_kst=now,
        original_four_step_plan_completed=False,
        steps={'1_public_inputs_and_annotations':'prepared96clips288windows; initial motion reliability failed',
               '2_joint_vs_protected_learning':'implementation reviewed; zero runs because preparation not ready',
               '3_dtw_guide_link':'exact-time coverage audit complete,14of117stages paired; no learned feedback integration',
               '4_whole_video_feedback_evaluation':'not executed; no validated new feature model or dense educational truth'},
        meaningful_change='Unresolved motion no longer leaves84 certain negative cells; ordered retrospective evidence preserves28handwindows364frames without semantic guesses.',
        model_accuracy_improvement=None,training_runs=0,html_created=False,active_execution=None,
        user_or_mentor_recording_access=0,production_or_C_T3F_changes=0,
        first_readiness_failure_preserved=True,
        next_action='Freeze fit-only representative temporal-label pilot; define centre support and separate ordered boundary evidence; verify before full same-rule annotation and new six-run contract.',
        source_pins={str(U/f):sha(U/f) for f in files})
    write_json(U/'COMPLETION.json',completion,True)
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
    backup=U/'before/preparation_close';backup.mkdir(parents=True,exist_ok=False)
    for p in [context,handoff,state,U/'CURRENT_EXECUTION.json']:shutil.copy2(p,backup/p.name)
    text=context.read_text(encoding='utf-8-sig')
    status='''### 2026-11-01 주요 동작 특징 분리 준비·오류 보완 완료 — 새 학습 준비 미통과

- `word100_core_feature_split_20261101`: 공개96영상/288개13연속원본창(학습60/개발20/새특징표현평가16) 준비종료0,잠금없음. 기존48해시재사용+신규48추출. 독립입력56,263·주석산술11,774검산PASS. 과거TCN TRAIN계보,프로젝트전체미사용평가아님.
- 신규144창 직접RGB·별도72창. 확인가능끼리 손모양0/323·높낮이0/46불일치,움직임28/113불일치(24.78%). 최초전체binary56/970(5.77%) 준비FAIL보존. 손모양일치/산술PASS로움직임신뢰도통과X. 신규학습/모델점수0,기존6회재실행0.
- 불일치28손창 전수원본검토: 유지→이동경계,회전/평행이동혼동,움직이는손확대화면의위치관찰편향 등. 단어정답으로가림불중요판정X. 위치/방향/모양사건분리,364손프레임전수유지(위치직접226/미확인138),17창복수상태. 같은Codex사후진단≠사람수어정답·성능향상.
- 카테고리불명시5움직임항목전체보류하는별도guard에서84음성셀철회,288창/중심정적정보/좌표보존. 원DATASET/FAIL/주석불변. 합성21·root저장11,363·에이전트순서1,833/DTW연결1,076검산PASS. 최초코드입력/경계출처검사누락수정이력보존.
- 기존36쌍117단계중양쪽직접중심정답쌍14단계만연결가능,전체단계/13창정렬정답아님. 원globalDTW/분모보존. C/T3F·운영·개인/멘토·원본/이전결과보존. HTML생략. RESULT_KO.txt/COMPLETION/review근거. 후속중심움직임정답은새버전범위이며기존whole13라벨·학습계약을그대로쓸수없음.

'''
    nxt='''### 2026-11-01 다음 단일 작업 — 시간별 움직임 정답 기준의 학습자료 한정 표본 검증

완료단위 `word100_core_feature_split_20261101` COMPLETION/RESULT_KO.txt/review/NEXT_PROTOCOL_REVIEW.json 참조. 입력96/288·검산·사후28진단 재실행금지,원TRAIN freeze는준비FAIL로차단유지. 다음은fit사람만의사전고정대표pilot에정상유지/느린이동/경계/회전/교차/짧은변화포함→중심시점움직임의근거시간범위와경계·순서별도기록규칙고정→별도RGB판독. 손모양은확대,평행이동은고정몸통화면. 경계가중심근거범위를덮으면보류,정지/가림/짧음으로언어불중요추론X. 초기28불일치만합의시켜쉬운점수금지.
통과시기존+신규전체고정표본의움직임을동일새규칙으로작성; 입력·정적중심·기존보호출력·DTW만호환범위해시재사용. 움직임별신뢰도/양클래스지원/사람표현분리/출력뜻·학습계약새버전동결후joint대조/보호손모양+분리움직임각3seed. 기존whole13정답을중심정답으로자동전환X. 이미소모평가를미사용이라부르지않음. 다음일반화주장은별도계보감사. 개인자료/운영교체/새모델다운로드금지,명확한개선전HTML없음. active실행없음; routine후속승인재질문불필요.

'''
    def replace_latest(body,header,replacement):
        pos=body.index(header);start=body.index('### ',pos);end=body.index('\n### ',start+4)
        return body[:start]+replacement+body[end+1:]
    text=replace_latest(text,'## 8. 현재 상태',status)
    text=replace_latest(text,'## 9. 다음 단일 작업',nxt)
    context.write_text(text,encoding='utf8')
    st=read(state);st['updated_date_kst']=now;st['current_stage']='CORE_FEATURE_INPUT_REPAIR_COMPLETE_MOTION_LABEL_PROTOCOL_NEXT'
    st['next_single_action']=completion['next_action']
    st['word100_core_feature_split_20261101']={**{k:st.get('word100_core_feature_split_20261101',{}).get(k) for k in ('authorization','candidate_manifest')},
        'status':completion['status'],'updated_kst':now,'clips':96,'windows':288,
        'roles':{'fit':60,'development':20,'new_feature_expression_evaluation':16},
        'prep_exec':6164,'prep_exit':0,'active_execution':None,'training_started':False,'training_runs':0,
        'numeric_scalar_checks':56263,'label_arithmetic_checks':11774,
        'original_motion_disagreement':{'conflict':28,'known_pair':113},
        'guard_negative_cells_withdrawn':84,'root_remediation_checks':11363,
        'ordered_diagnosis':{'hand_windows':28,'hand_frames':364,'translation_direct':226,'translation_unknown':138},
        'original_four_step_plan_completed':False,'new_html':False,'no_private_access':True,
        'report':str(U/'RESULT_KO.txt'),'completion':str(U/'COMPLETION.json'),'next_protocol_review':str(U/'review/NEXT_PROTOCOL_REVIEW.json')}
    write_json(state,st)
    handoff.write_text('''# WordsInSilence 최신 인수인계

2026-11-01 공개 입력 준비·움직임 정답 오류 보완까지 완료. 새 학습은 준비 미통과로0회.
목표: 한국수어 설명과 여러 공개참고로 손모양·관계·움직임·순서를 이해하여 주요 차이를 설명한다. 공식영상은단어정답참고이지프레임별가림/중요도정답은아님. 사용자·멘토자료사용금지.
현재: word100_core_feature_split_20261101. COMPLETION/RESULT_KO.txt/review/NEXT_PROTOCOL_REVIEW.json 우선. active실행/잠금없음. prepare exec6164종료0,96영상288개13원본창전수완료; 입력56,263·주석산술11,774검산PASS. 기존48재사용+새48,학습60/dev20/특징평가16. 과거TCN TRAIN계보이므로프로젝트전체fresh아님.
신규144창주석/별도72창: 손모양0/323·높낮이0/46불일치,움직임28/113(24.78%). 원LABEL_AUDIT NOT_READY와binary56/970보존. 산술/손모양일치로학습통과X. 원freeze차단유지.
보완: guard불명손창5움직임전체보류→84음성셀철회(별도DATASET_MOTION_GUARDED,원본불변). 28손창전수RGB사후진단: 유지/이동경계·회전혼동·고정몸통vs움직이는손crop관찰차이. 위치/방향/모양순서별도,364손프레임(직접위치226/미확인138)보존. root11,363·에이전트순서1,833/DTW1,076·합성21검사PASS. 같은Codex≠독립사람KSL정답. 원36쌍117DTW단계중14만양쪽정답중심쌍연결,전체동작정답아님.
다음: fit-only 대표pilot(정상유지/느린이동/경계/회전/교차/짧은변화) manifest와중심근거시간범위+순서기록규칙고정,별도판독부터. 불일치28만선별통과금지. 통과후기존/신규모든움직임에같은새규칙; 기존whole13라벨을중심정답으로재사용X. 입력/정적중심/보호출력/DTW만호환해시재사용. 출력뜻·계약새버전으로준비통과후각3seed. 원래4단계전체완료아님; 학습/새피드백성능미확인.
보호: C/T3F/운영DTW·웹/원본/과거결과읽기전용,개인자료접근0,새다운로드/카메라/Git/외부전송0. 개선명확전새HTML생략,JSON/TXT사용. 이전 visible_temporal_targets6학습/점수/속도재실행금지. Python ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8. routine후속작업승인재질문불필요.
''',encoding='utf8')
    write_json(U/'CURRENT_EXECUTION.json',dict(stage=completion['status'],updated_kst=now,active_execution=None,
        prep_exec=6164,prep_exit=0,prep_complete=True,training_started=False,
        duplicate_execution_forbidden=True,original_readiness='NOT_READY_SUPPORT_OR_REVIEW',
        completion=str(U/'COMPLETION.json'),next_action=completion['next_action']))
    for p,h in completion['source_pins'].items():assert sha(p)==h,p
    read(state);read(U/'CURRENT_EXECUTION.json');assert '28/113' in context.read_text(encoding='utf8') and '28/113' in handoff.read_text(encoding='utf8')
    for name in ('RESULT_KO.html','TRAIN_FROZEN.json','TRAINING.json'):assert not (U/name).exists()
    print(json.dumps({'status':'SYNCED_PREPARATION_REPAIR_NOT_FULL_PLAN_COMPLETION','completion':str(U/'COMPLETION.json'),'training_runs':0,'new_html':False},ensure_ascii=False))
if __name__=='__main__':main()
