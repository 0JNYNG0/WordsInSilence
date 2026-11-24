"""Synchronize the verified bounded task; no model or experiment execution."""
import json,re
from datetime import datetime,timezone,timedelta
from pathlib import Path
from pipeline import UNIT,read,sha,write

ROOT=UNIT.parents[3]
TASK=UNIT.name

def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (UNIT/'COMPLETION.json').exists()
    assert not (UNIT/'RUNNING.json').exists()
    verification=read(UNIT/'review/RESULT_VERIFICATION.json')
    review=read(UNIT/'review/REPORT_REVIEW.json')
    assert verification['status'].startswith('PASS')
    assert review['status'].startswith('PASS')
    frozen=read(UNIT/'FROZEN.json')
    pins={**frozen['source_pins'],**frozen['code_and_review_pins']}
    for p,h in pins.items():assert sha(p)==h,p
    ev=read(UNIT/'EVALUATION.json');dec=read(UNIT/'TRAINING_DECISION.json')
    assert ev['natural_counts_both_axes']['possible']==30560
    assert ev['new_training']==dec['new_training']==0
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    state_path=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(state_path)
    state.update(current_stage='ACTION_FEEDBACK_CASEBOOK_COMPLETE',updated_date_kst=now,updated_at=now)
    state[TASK]={
        'status':'COMPLETE_VERIFIED_EVIDENCE_LINKAGE_NOT_ACCURACY_GAIN','at':now,'active':None,
        'unit':str(UNIT),'report':str(UNIT/'RESULT_KO.txt'),'casebook':str(UNIT/'CASEBOOK_KO.txt'),
        'scope':{'cases':4,'references_per_case':3,'pairs':12,'parents':39,'technical':810,
                 'new_RGB_windows':10,'unique_RGB_frames':303,'new_reviewed_parents':10,'other_parents':29},
        'source_links':{'linked_parents':12,'physical_only':27,'clauses':30,'partial':24,'proximity':3,'unverified':3},
        'both_axis_opportunities':ev['natural_counts_both_axes'],
        'independent_verification':str(UNIT/'review/RESULT_VERIFICATION.json'),
        'independent_checks':2126681,'verification_failures':0,
        'selftest_checks':168,'visual_scope_checks':2286,
        'execution_session':81552,'execution_exit_code':0,
        'new_training':0,'new_model_inference':0,'new_DTW':0,'new_coordinate_comparison':0,
        'training_decision':dec['status'],'production_changed':False,'C_T3F_changed':False,'user_mentor_used':False,
        'findings':['P048 f72–74 visible hand silhouette but stored angle/axis support missing; raw detector versus later filtering cause unresolved',
                    'P066 q68↔P063 r58 opposite upper hand in stored temporal correspondence; not learner error'],
        'educational_accuracy':None,'alignment_accuracy_gain':None,
        'next_single_action':'고정10창에서 원검출→좌우배정→영역/품질→저장마스크의 기존 계보를 확인한다. 후처리의 잘못된 제외가 확인될 때만 공통 보존 후보; 원검출없음/계보없음은 명시종료. P066 대응문제 별도유지·재학습정답금지.',
        'next_plan':str(UNIT/'NEXT_PLAN_KO.txt'),'next_executed':False,
        'html_created':False,'existing_report_8800_is_previous_task':True}
    save(state_path,state)
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
    current='''### 2026-11-01 설명·동작 근거 연결과 차이 원인 사례4묶음 완료
`word100_action_feedback_casebook_20261101`의 승인4단계 완료. 기존설명카드/trace+최신독립손특징을4DEV×3FIT=12쌍39부모/810저장기술대조(849결과)에 연결. 사전고정10RGB창/8영상303고유프레임30page검토,다른29부모는신규시각검토X. 사전부분12/물리설명27,30clause=24부분/3근접/3미확인유지. 양축기회30560=direct24174/partial32/unknown6354 보존. 수치차이/원본관측/시점대응/설명지원/언어중요성분리;원mean/max/모든peak시점/미관측·3참고유지.
P048f72–74 RGB손보임+저장각도/축missing확인(원검출실패vs정상ROI/후처리원인미확정). P066q68↔P063r58은서로반대손높이시점대응확인;교육오답정답아님. 정상좌표·마스크·교차·기존경로불변. 자체168/별도저장검산2,126,681오류0/RGB범위2286PASS·보고검토PASS. 같은Codex검산≠독립수어전문가정답.
새학습·추론·DTW·좌표비교0:추가학습에필요한분리된판정부실패/시간정답근거불충분판정. 기존모델무오류뜻아님. 인식/정렬/교육정확도개선주장없음. exec81552종료0,새HTML없음/8800이전보고서유지. C/T3F·운영·사용자/멘토·원본·과거동결보존. RESULT_KO/CASEBOOK_KO/TRAINING_DECISION/COMPLETION/review참조.

'''
    nxt='''### 2026-11-01 다음 단일 작업 — 보이는 손의 저장 관측 누락 계보 확인
완료한casebook849연결/기존DTW/모델/좌표비교재실행X. 고정10창을그대로사용해기존원검출→좌우배정→상반신범위/품질처리→최종각도·방향마스크를추적한다. 몸옆비동작손의의도된제외와필요한손의잘못된제외를구분. 원검출에값이있고사용범위도맞는데후처리가잘못제외한근거가있을때만공통수정후보1개;원검출없음/계보부족이면미확정명시후해당범위종료. 단어예외/추정손정답채움/새모델·가중치순회금지. P066교대시점대응은별도한계유지,손모양학습오답으로전용X. NEXT_PLAN_KO.txt상세,아직미실행. 목표설명·주요동작피드백유지,교육정오·3D회전성능미검증. C/T3F·운영·개인/멘토·원본보존.

'''
    for num,body in ((8,current),(9,nxt)):
        m=re.search(r'^## '+str(num)+r'\.[^\n]*\n',text,re.M);assert m
        text=text[:m.end()]+'\n'+body+text[m.end():].lstrip('\r\n')
    context.write_text(text,encoding='utf-8')
    handoff=ROOT/'NEW_CHAT_HANDOFF.md'
    handoff.write_text('''# WordsInSilence 최신 인수인계

최종목표는 한국수어 설명과 다중 참고를 활용해 주요 동작의 손모양·관계·움직임·순서를 파악하고 중요한 차이와 정상 변이를 구분하는 피드백이다. 목표 단어는 참고 선택용이고 보이지 않는 손을 정답대로 채우지 않는다.

마지막 완료: word100_action_feedback_casebook_20261101 승인4단계. 4DEV×3FIT/39부모+810기술 저장결과849 연결, 신규10창/303고유RGB원프레임검토, 원분모/원값/모든peak/미관측/3참고보존. 사전부분12/물리27/30clause24·3·3유지. 양축30560 direct24174/partial32/unknown6354. 자체168·별도검산2,126,681오류0·RGB범위2286·보고검토PASS. 같은Codex검토이며 독립사람/KSL정답아님.
P048f72–74 손보임+저장지원누락, P066q68↔P063r58 서로반대손높이시점대응. 원검출실패/영역제외/후처리 원인미확정, 정상수어차이를교육오답으로지정하지않음. 숫자차이·원본관측·시점·설명지원분리기능완료이며 인식률/정렬/교육정확도상승아님. 학습정답/분리된판정부실패근거부족으로새학습0(기존판정부무오류뜻X). 추론/DTW/좌표재비교0.
실행exec81552종료0,RUNNING없음. 완료결과재실행금지. RESULT_KO.txt/CASEBOOK_KO.txt/TRAINING_DECISION.json/COMPLETION.json/review가근거. 새HTML없음,8800은이전hold보고서exec35320이며이번결과로교체X.

다음 한 작업(아직미실행): 고정10창에서 기존원검출→배정→범위/품질→최종마스크계보추적. 정상몸옆손제외와잘못된필요손제외구분. 실제원관측지원이있고후처리잘못이확인된경우에만공통보존후보;원검출없음/저장계보없음은미확정종료. P066대응한계는별도유지하고학습실패정답으로합산금지. NEXT_PLAN_KO.txt참조. 새검출기/거리식/가중치순회X.
기존C·T3F·운영웹/DTW·원본W:/sources/과거동결보존. 사용자·멘토자료 접근/학습/튜닝/후보선택금지,다운로드/외부전송/카메라/Git/운영교체X. 자율작업역할AGENTS유지,중복승인요청불필요하나현재완료와후속미실행구분.
''',encoding='utf-8')
    history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with history.open('a',encoding='utf-8') as f:
        f.write('''\n\n2026-11-01 — 설명과 비교값을 연결한 뒤 차이 원인까지 구분한 단계
목표 변화가 아니라 기존 피드백 목표의 근거 연결 보완이다. 기존 설명 카드 기능을 다시 만들지 않고 최신 독립 손가락 각도·2D손방향·팔상대방향에 원본 시점과 관측 상태를 연결했다. 4개 공개 개발표현의12쌍39범위를 보존하고 사전고정10창만 새로 확인했다. 잘하다에는 보이는 손의 저장지원 누락, 조상에는 서로다른 높이의 손 순간을 맞춘 대응을 확인했다. 두 원인은 판정부의 학습 오답으로 합치지 않았다.
별도검산849요약/2,126,681검사오류0은 근거보존 검증이며 교육정오·단어인식·정렬정확도 향상은 아니다. 추가학습 조건이 충족되지 않아 기존 모델을 유지했다. 다음은 새 모델 도입보다 저장관측 누락의 원검출/배정/범위/품질 계보 추적이다. 사용자자료는 사용하지 않았고 운영/C·T3F는 보존했다. 근거: research/front_keyphase_v1/ops/word100_action_feedback_casebook_20261101/RESULT_KO.txt,COMPLETION.json,NEXT_PLAN_KO.txt.
''')
    cur=read(UNIT/'CURRENT_EXECUTION.json')
    cur.update(phase='COMPLETE',active=None,at=now,exec_session=81552,exit_code=0,
               new_training=0,new_model_inference=0,new_dtw=0,next_executed=False)
    save(UNIT/'CURRENT_EXECUTION.json',cur)
    artifacts=['AUTHORIZATION.json','FROZEN.json','INPUTS.json','RESULT_MANIFEST.json','CASEBOOK.json',
        'EVALUATION.json','TRAINING_DECISION.json','RESULT_KO.txt','CASEBOOK_KO.txt','NEXT_PLAN_KO.txt',
        'report.py','render_casebook.py','finalize.py','CURRENT_EXECUTION.json',
        'review/RESULT_VERIFICATION.json','review/POSTEXECUTION_REVIEW.json','review/REPORT_REVIEW.json']
    docpaths=[context,handoff,state_path,history]
    write(UNIT/'COMPLETION.json',{
        'status':'COMPLETE_VERIFIED_BOUNDED_FOUR_STEP_TASK','at':now,
        'producer':'root + semantic_feedback','qualitative_RGB':'semantic_data_audit',
        'independent_saved_result_verifier':'semantic_model; same Codex, not independent human KSL truth',
        'four_steps':{'source_scope_reuse':'complete','evidence_join':'complete',
                      'bounded_visual_cause_review':'complete','training_necessity_decision':'complete; no new training justified'},
        'records':849,'natural':39,'technical':810,'cases':4,'new_visual_windows':10,
        'checks':{'selftest':168,'independent_result':2126681,'visual_scope':2286,'failures':0},
        'model_training':0,'new_inference':0,'new_DTW':0,'new_coordinate_comparison':0,
        'accuracy_improvement_claim':False,'production_changed':False,'C_T3F_changed':False,
        'user_mentor_used':False,'next_task_executed':False,
        'source_and_code_pins_checked':len(pins),
        'artifacts':{str(UNIT/p):sha(UNIT/p) for p in artifacts},
        'synchronized_documents':{str(p):sha(p) for p in docpaths}})
    for p in docpaths:assert TASK in p.read_text('utf-8-sig'),p
    assert read(state_path)[TASK]['active'] is None
    print('COMPLETE; four project records synchronized; protected frozen pins unchanged')

if __name__=='__main__':main()
