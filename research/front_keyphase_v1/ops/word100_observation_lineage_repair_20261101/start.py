from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,re,shutil
U=Path(__file__).resolve().parent;ROOT=U.parents[3];PREV=U.parent/'word100_action_feedback_casebook_20261101'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert read(PREV/'CURRENT_EXECUTION.json')['phase']=='COMPLETE'
    assert not (PREV/'RUNNING.json').exists()
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    (U/'review').mkdir(exist_ok=True);(U/'before').mkdir(exist_ok=True)
    docs=['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json','Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    for name in docs:shutil.copyfile(ROOT/name,U/'before'/Path(name).name)
    new(U/'AUTHORIZATION.json',{'at':now,'request':'발견되는 보완·수정 작업을 계획하여 이어서 수행하고 검증 후 보고한다는 사용자 최신 직접 승인',
        'scope':'고정10창 저장 관측 계보 추적→근거가 확인된 공통 결함 수정→공개자료/기술 대조 검산; 같은 범위에서 드러난 후속 문제는 해결까지 이어간다.',
        'supersedes':'이전 다음단일작업 미실행 상태와 사소한 후속 단계별 재승인 대기',
        'unchanged_constraints':['C/T3F/운영웹·DTW/과거동결/원본보존','사용자·멘토자료 접근/학습/튜닝/선택금지','새다운로드/외부전송/카메라/Git없음','단어예외·결과맞춤임계값조정금지'],
        'stopping_conditions':['현범위에서 원검출/저장계보가없거나 필요한 관측이 없어 입증된 복구 불가인 경우 한계로 남긴다','실질적범위/위험확대나 해결불가능한중대문제만 사용자보고'],
        'plan':['기존10창·원관측/NPZ/producer 계보고정','검출/배정/영역/품질/특징계산의 누락원인구분','검증된결함만격리공통수정,필요후속연결계속','정상손·손내림·교차·짧은특징보존검산과쉬운결과보고'],
        'new_training':'only if supported target truth and isolated model failure justify a separately frozen comparison; do not train from inference or guide expectations',
        'roles':{'root':'contract/run/follow-through/report','semantic_data_audit':'saved array lineage','semantic_feedback':'producer code and candidate','semantic_model':'independent validation'}})
    v=read(PREV/'review/VISUAL_REVIEW.json');pins={str(PREV/p):sha(PREV/p) for p in ['FROZEN.json','COMPLETION.json','INPUTS.json','review/WINDOW_SELECTION.json','review/VISUAL_REVIEW.json','NEXT_PLAN_KO.txt']}
    for w in v['windows']:
        for a in ['query_axis','reference_axis']:
            d=w[a];assert sha(d['input'])==d['input_sha256'];pins[d['input']]=d['input_sha256']
    new(U/'INPUT_SCOPE.json',{'windows':v['windows'],'source_pins':pins,'fixed_windows':10,'unique_frames':303,'new_visual_scope_expansion':False,'source_model_scoring_repeated':False})
    new(U/'CURRENT_EXECUTION.json',{'phase':'LINEAGE_AUDIT_RUNNING','at':now,'active':'parallel saved-array/producer-code/method review','new_model_inference':0,'new_dtw':0,'new_training':0})
    stp=ROOT/'research/front_keyphase_v1/ops/STATE.json';st=read(stp)
    st.update(current_stage='OBSERVATION_LINEAGE_REPAIR_RUNNING',updated_date_kst=now,updated_at=now)
    st[U.name]={'status':'LINEAGE_AUDIT_RUNNING','at':now,'unit':str(U),'active':'read-only lineage and common fix assessment','new_training':0,'production_changed':False,'user_mentor_used':False}
    stp.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
    for n,text in [(8,'### 2026-11-01 관측 누락 원인 추적·공통 수정 승인 실행\n`word100_observation_lineage_repair_20261101` 시작. 최신사용자는발견문제의후속보완도바로계획/수정/검증후보고승인. 기존10창고정,원검출→배정→영역/품질→저장특징계보를에이전트와읽고검증된공통결함만격리수정. 기존casebook완료재실행X,C/T3F·운영·개인/멘토·원본보존. AUTHORIZATION/INPUT_SCOPE/CURRENT_EXECUTION참조.\n\n'),(9,'### 2026-11-01 현재 단일 작업 — 저장 정보 누락의 원인과 검증된 후속까지 해결\n고정10창누락을원검출없음/의도된영역제외/배정·품질/특징계산결함으로분리. 기존원관측근거가있을때만공통수정후보조건을고정하고정상손·교차·손내림·짧은변화보호와공개이전/이후검산. 사소한후속은다시승인요청없이이어가며자료부족을추정정답으로채우지않는다. P066시점문제는별도원인으로유지하되동일목표범위내근거있는후속해결가능성을검토. 운영교체/새모델순회/개인튜닝X.\n\n')]:
        m=re.search(r'^## '+str(n)+r'\.[^\n]*\n',s,re.M);assert m;s=s[:m.end()]+'\n'+text+s[m.end():].lstrip('\r\n')
    p.write_text(s,encoding='utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('''# WordsInSilence 최신 인수인계

현재 word100_observation_lineage_repair_20261101 진행. 최신 사용자: 문제발견후보고로멈추지말고필요한보완·수정계획을세워후속까지검증후보고. 기존C/T3F·운영웹/DTW·원본/과거동결보존,사용자/멘토자료접근·학습·튜닝·선택금지,다운로드/외부전송/카메라/Git없음.
마지막완료 action_feedback_casebook:4DEV×3FIT/39부모+810저장대조849연결/별도검산PASS. 고정10창303원프레임검토. P048f72–74손보임+저장각도축missing;P066q68↔r58서로다른손높이시점대응. 인식/교육정확도향상아님. 완료결과재계산X.
현재10창원검출→배정→영역/품질→NPZ특징계보추적. root계약/실행/보고,semantic_data_audit배열,semantic_feedback코드/후보,semantic_model별도검산. AUTHORIZATION/INPUT_SCOPE/CURRENT_EXECUTION먼저확인. 원검출에근거있는후처리/계산결함만격리공통수정,정상손내림제외보호/단어예외X. 자료없음을추정정답으로해결X. 후속필요시최초오류보존·조건검토후자율해결,큰문제만보고. 아직새모델/DTW/학습없음.
8800은이전hold보고서exec35320,운영카메라/서버교체X. 이번새실행있으면중복시작하지말고실제세션/잠금확인. 작업마감시§8§9·STATE·HANDOFF·이력동기화.
''',encoding='utf-8')
    print('new authorized unit initialized; fixed10 source pins verified')
if __name__=='__main__':main()
