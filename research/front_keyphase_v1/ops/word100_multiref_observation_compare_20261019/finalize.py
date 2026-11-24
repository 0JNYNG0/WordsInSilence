"""Close this authorized offline comparison and synchronize its handoff only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib
U=Path(__file__).resolve().parent; B=U.parents[1]; ROOT=B.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json')
    assert sha(U/'FROZEN.json')=='93d98ecfd34558676731941b33affbacbabc3844e1891eadde00f1b6b768e204'
    assert sha(U/'compare.py')==f['implementation_sha256']
    assert all(sha(p)==h for p,h in f['protected'].items())
    assert v['status']=='PASS' and read(U/'BROWSER_QA.json')['status']=='PASS'
    assert read(U/'COMPARE_EXIT.json')['exit']==0 and not (U/'RUNNING.json').exists()
    banks={k:read(U/f'BANK_{k}.json') for k in f['views']}
    for k,bank in banks.items():
        assert len(bank['words'])==100
        assert sum(len(w['references']) for w in bank['words'].values())==899
        assert bank['training_only'] and not bank['production_ready']
    artifacts=['SUMMARY.json','DEV_ROWS.json','TRANSITIONS.json','INDEPENDENT.json','CONCLUSION.json','VISUAL_REVIEW.json','RESULT_KO.html','RESULT_KO.txt','BROWSER_QA.json']+[f'BANK_{k}.json' for k in f['views']]
    audit={'status':'PASS','at':at,'protected_files':len(f['protected']),'protected_unchanged':True,'frozen_implementation_unchanged':True,'visual_frames_inspected':18,'visual_reports':6,'browser_qa':'PASS','same_operator_separate_verifier':True,'training':0,'classifier_inference':0,'private_recordings_read':0,'production_changes':0,'artifact_sha256':{n:sha(U/n) for n in artifacts}}
    write(U/'FINAL_AUDIT.json',audit)
    nextaction='사용자와 공개100단어 다중참고·관측비교 결과를 확인한다. 다음은 손 위치/굽힘의 항목별·구간별 관측 근거와 궤적 시작 기준을 분리하는 공통 DTW 후보의 조건 검토. 현재 C/T3-F·웹·DTW 유지. 인식보조·추가학습·사용자자료 튜닝은 시작하지 않는다.'
    status='COMPLETE_OFFLINE_COMPARISON_NOT_PROMOTED'
    statepath=B/'ops/STATE.json';state=read(statepath)
    state['current_stage']='WORD100_MULTIREF_OBSERVATION_COMPARISON_COMPLETE_REVIEW'
    state['updated_date_kst']=at[:10];state['next_single_action']=nextaction
    state[U.name]={
        'at':at,'status':status,'unit':str(U),'implementation':'Codex local','verification':'same Codex separate numerical verifier, no extra agent/expert',
        'classes':100,'training_references':899,'development_queries':300,'unique_videos':1199,
        'view_pairs':29643,'views':f['views'],'measurable_DEV_hand_metric_units':{k:s['views'][k]['totals']['measurable'] for k in f['views']},
        'DEV_hand_metric_denominator':1773,'known12_complete_DEV36':{k:s['views'][k]['known12_complete_DEV'] for k in f['views']},
        'known_roles':12,'unknown_role_words':88,'visual_frames':18,'visual_clips':6,
        'finding':'Intentional lowered-hand exclusion and in-frame active-hand evidence loss coexist; measurement coverage is not recognition accuracy.',
        'independent_verification':'PASS','comparison_exit':0,'comparison_session':64425,
        'protected_unchanged':True,'training':0,'classifier_inference':0,'user_recordings_read':0,'production_changes':0,
        'report':str(U/'RESULT_KO.html'),'audit':str(U/'FINAL_AUDIT.json'),'next_single_action':nextaction}
    write(statepath,state)
    title='### 2026-10-19 公開100 다중참고·DTW 관측비교 완료 — C 관측 단순 연결 미채택'.replace('公開','공개')
    sec8=f'''{title}
갱신 {at}. ops/{U.name}. 사용자 승인한 연구용 다중참고12단어108영상→100단어899영상 확장과3관측 비교 완료. TRAIN01~09/DEV11~13,고유1199영상,3597특징입력,7184TRAIN+2697DEV방향쌍/관측=29643쌍×관측. exec64425 종료0/잠금없음. 추가학습·인식보조 미실행.
- 수치측정가능 DEV손×항목1773개 기준 legacy1773/masked1380/sensor1369. 정확도 아님. 기존12단어DEV36에서3항목측정·기준비교가능36/30/35. sensor손실404중왼손403/오른손1; 내린손제외와영역내동작손근거누락이혼재하여일괄오류/불필요손판정금지.
- 공개6영상18장면 원본+저장관절 검토:흡연실 표본은내린왼손제외,후원은내림+겹침영역내손근거누락,믿다REAL12는영역내손sensor누락/왼손굽힘참고9→3(위치·궤적7),마음에들다REAL13경계부분관측. 전체누락전수의미검토/언어오류정답아님.
- 기존12역할유지/새88단어좌우별기술비교만. 정상TRAIN범위≠합격선. 관측변경은검출·배정·범위·시간구간묶음,한요소효과아님. 궤적첫관측기준변화주의. 측정가능수/낮은거리만으로개선판정금지.
- 별도산술검산3597입력/7312875특징/42725집계/29643경로/300고정쌍PASS,기존12결과일치. 최초검산구문/acos반올림오류증거보존→검산만보완PASS,점수조건무변경. Codex동일담당별도검산,타agent/전문가아님. FINAL_AUDIT 보호{len(f['protected'])}파일불변·화면QA PASS.
- C/T3-F/웹/DTW/원본유지,개인멘토자료미접근/학습튜닝선택금지,14~18원시미접근. RESULT_KO.html/txt,SUMMARY,INDEPENDENT,VISUAL_REVIEW,FINAL_AUDIT. 후속학습/인식보조는결과확인후.

'''
    cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ctx=cp.read_text('utf-8-sig')
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
    assert title not in ctx and ctx.count(h8)==1 and ctx.count(h9)==1
    ctx=ctx.replace(h8,h8+'\n'+sec8,1).replace(h9,h9+'\n'+title+'\n'+nextaction+'\n\n',1)
    cp.write_text(ctx,'utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 여러 공개 참고를 활용한 DTW 개선. 마지막 완료: 공개100단어 다중참고·기존/C관측경로 비교, 기존 C/T3-F 유지.

- 단위 {U}
- 완료: 연구용100단어899TRAIN참고,DEV300,고유1199영상×3관측,29643쌍×관측. 비교64425/시각자료20328 모두종료0,활성작업없음. 추출/계산/검산재실행불필요.
- 핵심: DTW손×항목 측정가능1773/1380/1369(분모1773),기존12단어DEV36 완전측정36/30/35. 인식정확도아님. 내려놓은손제외와영역내동작손누락이섞여C관측단순연결미채택.
- 공개6영상18장면확인:흡연실내림제외,후원겹침근거부족,믿다REAL12왼굽힘참고3<5/위치궤적7,마음에들다REAL13부분관측. 전체원인전수/언어적판정아님. 기존12역할유지,88단어손별비교만.
- 별도산술검산PASS,보호{len(f['protected'])}파일불변,브라우저QA PASS. 작성/실행/검산Codex(다른agent/전문가아님). RESULT_KO.html, SUMMARY.json, INDEPENDENT.json, VISUAL_REVIEW.json, FINAL_AUDIT.json 참조.
- 최초검산오류·수치반올림정정증거보존. compare.py/FROZEN/점수변경없음. Frozen SHA {sha(U/'FROZEN.json')}.
- C/T3-F/웹/DTW/원본보존. 사용자/멘토촬영미접근 및학습·튜닝·선택금지. 14~18원시미접근. 인식보조·추가학습0.

다음 한 작업: {nextaction}
''','utf-8')
    write(U/'STATUS.json',{'status':status,'at':at,'complete':300,'total':300,'verified':True,'report':str(U/'RESULT_KO.html')})
    execution=read(U/'CURRENT_EXECUTION.json');execution.update(status='EXITED',exit_code=0,completion_record='COMPARE_EXIT.json',finalized_at=at);write(U/'CURRENT_EXECUTION.json',execution)
    # Post-write consistency only; never rerun model or distance calculations.
    assert read(statepath)['current_stage']=='WORD100_MULTIREF_OBSERVATION_COMPARISON_COMPLETE_REVIEW'
    assert read(statepath)[U.name]['status']==read(U/'STATUS.json')['status']
    assert nextaction in cp.read_text('utf-8') and nextaction in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
    assert all(sha(p)==h for p,h in f['protected'].items())
    print(json.dumps({'status':status,'protected':len(f['protected']),'docs_synced':True},ensure_ascii=False))
if __name__=='__main__':main()
