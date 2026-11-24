"""Current-state checkpoint; no training or inference."""
from common import *
def main():
    k=HERE/'temporal_rate_rank_learning'
    mode=sys.argv[1];session=int(sys.argv[2])if len(sys.argv)>2 else None
    statuses={'prepare':'ACTIVE_G2_RATE_INPUT_ENCODING','train':'ACTIVE_G2_RATE_RANK_TRAINING','evaluate':'ACTIVE_G2_RATE_RANK_COMPARISON','review':'ACTIVE_G2_RATE_RESULTS_REVIEW'}
    status=statuses[mode]
    summary='손 전체 회전과 손가락 변화의 혼동을 줄인 R, 합성 연결 경계의 잘못된 동작 묶음을 줄인 J 연구 개선을 확인했다. 기존/J 점수를 고정1:1로 합치면 새 빠른 정상 회귀2건은 복구되지만, 최초부터 지원되던 REAL11 seed43의33ms 중첩1건은 남아 G2 REWORK이다. 최초 시점판정·구간판정과 중간 baseline을 구분해 계보감사를 완료했다. 연구 결과이며 웹·TCN·공통100은 그대로다.'
    nxt='K 고정 .8/1.2 시간변형의 공개 FIT 학습을 진행한다. 288변형 항목 중210은 원래 중요시점이 보존되어 새 특징을 만들고,78은 보류한다. 원래178/J72 묶음을 재생하고 같은 사람·동작의 속도변형 수가 학습 비중을 늘리지 않도록 한다.3seed/1200단계/최종checkpoint와 단독·동일가중 결합2조건을 고정. 완료 실험은 재실행하지 않는다. 구간이 달라진 경우 의미판정까지 확인한 뒤 G2/G3를 판정하며 최종8·웹은 그전 보류.'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command=f'temporal_rate_rank_learning {mode}',jobs=[dict(component='K',stage=mode,session_id=session)]if session else[],session_id=session,other_session_id=None,training_running=mode=='train',process_running=session is not None,next=nxt);write(GRAPH/'CURRENT_EXECUTION.json',e)
    sub=read(HERE/'SUBGRAPH.json');sub.update(at=stamp(),status=status)
    for n in sub['nodes']:
        if n['id']=='C4':n.update(status='SPLICE_GAIN_PRESERVED_ONE_FAST_BOUNDARY_REWORK',evidence=['stability_rank_blend/RESULTS.json','G2_REFERENCE_LINEAGE_REVIEW.json','temporal_rate_rank_learning/PREPARATION_FROZEN.json'])
    write(HERE/'SUBGRAPH.json',sub)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['last_updated']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=mode=='train',progress=summary,remaining=nxt,latest_evidence=str(k/'PREPARATION_FROZEN.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 연결 개선 유지·속도변형 학습'
    if title not in t:
        for head,body in[('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(head,head+'\n\n'+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/G2_REFERENCE_LINEAGE_REVIEW.json`.\n',1)
        p.write_text(t,'utf-8')
    handoff=f'''# 최신 인수인계 — 2026-11-13 새벽 G2 자율 진행

{summary}

{nxt}

현재 K 단계={mode}, session={session}. 실제 세션/완료파일을 확인해 중복 실행하지 않는다. K prepare 최초 ModuleNotFoundError(splice_core)는 encoder 전 발생했으며 FIRST_FAILURE 보존. launch_encoder.py가 J 경로만 명시해 재개, frozen code는 그대로다. 입력완료 파일은 PREPARATION_INPUTS_COMPLETE.json. root train.py/evaluate.py 검토 후 단계별 동결·실행한다.

R 손 회전대표54/54·회전324/324 개선, 짧은변화672창 중630/672/630은 미해결. A각도잔차는대표53/54·회전318/324로 미채택. R/A/P/H/N/P/S/B/M/J의 완료학습·추론 재실행 금지. J 합성구간 증거13/18→18/18, 역순4/9→9/9이며 SB동일가중도 개선 유지. J단독 fast33ms 겹침3seed; SB기존local_full수준8/9로복구하나 최초REAL11seed43손실은남는다. baseline 명칭이최초G2기준과다름을 G2_REFERENCE_LINEAGE_REVIEW에기록.

J 변경구간 의미37새요청 및 SB8새요청 완료. 정상/speed의 의미상태 그대로, 손전체미관측72shapeunknown·completion전부None. SB저장독립검산은absence agent가수행. K준비=palmp_fixture agent,trainer/eval정적검토=motion agent,root실행·조율. 같은Codex검산은독립사람/KSL정답아님.

주요경로H=ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113. runtime=ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. G2 REWORK/G3이후보류. originalG2은 midpoint/occurrence/extent비교구조를구분하고 소급 약한baseline교체금지. 최종8·사용자/멘토자료·카메라·다운로드·Git·외부전송·운영교체금지.100단어인코더/C·T3F보존. 사용자새벽계속진행승인유효,작업중단아님.
'''
    (root/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8');print(status,session)
if __name__=='__main__':main()
