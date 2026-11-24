from common import *
def main():
    mode=sys.argv[1];session=int(sys.argv[2])if len(sys.argv)>2 else None
    status='ACTIVE_G2_FULL_BAG_'+mode.upper();k=HERE/'temporal_rate_rank_learning';l=HERE/'full_bag_rate_rank_learning'
    summary='공개 속도변형 학습 K와 고정1:1결합을 모두 비교·별도검산했다. 합성 순서 시험의 실제 구간18/18·역순9/9 개선은 유지되지만, K단독은 정상1건·결합은빠른정상1건의약33ms경계겹침이남아미채택이다. 그 경계변화가새손모양/방향오답경고로이어진사례는없었다. G2보류를유지하며원인에맞춰학습예제표집을보완한다.'
    nxt='K FIT6실패는다른행동을잘못찾은것보다full-extent경계선택실패이고,승리음성의학습노출0~3회가확인됐다. L은K와동일J부모·동일1200단계·동일source선택일정에서,각선택source의유효속도모두및저장된음성최대512개를손실에포함한다. 자료/정답/가중비중은그대로. 고정L단독과원local_full0.5+L0.5를같은82조건3seed로1회비교하고새구간의미도검증한다. 완료실험재실행·G2기준완화·개인자료·최종8·웹교체금지.'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command=f'full_bag_rate_rank_learning {mode}',jobs=[dict(component='L',stage=mode,session_id=session)]if session else[],session_id=session,other_session_id=None,training_running=mode=='train',process_running=session is not None,next=nxt);write(GRAPH/'CURRENT_EXECUTION.json',e)
    sub=read(HERE/'SUBGRAPH.json');sub.update(at=stamp(),status=status)
    for n in sub['nodes']:
        if n['id']=='C4':n.update(status='K_GAIN_RETAINED_EXTENT_BOUNDARY_REWORK_L_'+mode.upper(),evidence=['temporal_rate_rank_learning/evaluation/INDEPENDENT_SAVED_VERIFICATION.json','temporal_rate_rank_learning/changed_span_shadow/VERIFICATION.json','temporal_rate_rank_learning/FIT_FAILURE_CAUSES.json'])
    write(HERE/'SUBGRAPH.json',sub)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['last_updated']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=mode=='train',progress=summary,remaining=nxt,latest_evidence=str(k/'FIT_FAILURE_CAUSES.json'));write(OPS/'STATE.json',state)
    graph=read(GRAPH/'TASK_GRAPH.json');g2=next(n for n in graph['nodes']if n['id']=='G2')
    eid='K_RATE_AND_FIXED_BLEND_REWORK'
    if not any(a.get('experiment_id')==eid for a in g2['attempts']):
        paths=[k/'TRAINING_SAVED_VERIFICATION.json',k/'evaluation/INDEPENDENT_SAVED_VERIFICATION.json',k/'changed_span_shadow/VERIFICATION.json',HERE/'G2_REFERENCE_LINEAGE_REVIEW.json']
        g2['attempts'].append(dict(at=stamp(),experiment_id=eid,status='REWORK',checks=dict(normal_preserved=False,false_support_reduced=True,not_abstention_only=True,three_seeds=True),evidence=[dict(path=str(p),sha256=sha(p))for p in paths],note='Both fixed K conditions preserve18/18 reordered witnesses and9/9 reverse relationships, but each loses one previously supported normal/speed extent separation. No new shape/direction warning. Extent overlap is not itself a KSL error. Prior G2 reference lineage retained; no gate waiver.'))
        g2['status']='REWORK';write(GRAPH/'TASK_GRAPH.json',graph)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 어려운 경계 예제 노출 보강'
    if title not in t:
        for head,body in[('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(head,head+'\n\n'+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/temporal_rate_rank_learning/FIT_FAILURE_CAUSES.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 새벽 자율 작업 계속

{summary}

{nxt}

L현재단계={mode}, session={session}. 실제 세션/완료파일 확인 후 재개, 중복실행 금지. motion agent=L/train.py, palm agent=K source선택일정 재생/준비, absence agent=독립코드검토, root=L/evaluate.py 및 실행. L부모는Kfinal이아니라K가시작했던J3checkpoint로동결예정. 이 선택으로추가학습량교란을없앤다. available rate동시기여는균등rate표집기댓값이고,전체음성은기존저장bag최대512개이지모든격자음성아님. 두변경의묶음효과이며각각의독립효과주장X.

K prepare98043/train19298/eval69222/changedsemantic모두exit0,재실행X. K준비검산4106/학습11464/평가7681/의미8574 항목PASS는산술검사수이다. FIT전체460bag의최상위fullpositive458/460·460/460·456/460. 6FIT실패는필수witness전부포함/몸100%/다른행동negative0이나경계띠밖;negative노출3/1/3/0/1/2. rate1normal/splice동일positive780개씩logit차최대7.63e-6이므로branchscore불일치가주원인이라고주장X.

구간경계비중첩/관측행동순서/요구완료는서로다름. ORDER_EXTENT_COMPLETION_DESIGN_REVIEW/RELATION_REUSE_DESIGN_REVIEW근거참조. 최초G2의midpoint/occurrence와currentbaseline을구분. 현재약33ms겹침을오답정답으로단정하거나문턱완화로통과시키지않음. K두조건정상/속도카드상태변화0,누락/가림완료승격0,손전체미관측shape144항목unknown유지. completion은여전히None.

기존R손회전대표54/54·회전324/324개선이나짧은변화미해결,추가A미채택. J합성구간13/18→18/18/역순4/9→9/9,SB와K개선유지. R/A/PD/H/N/P/S/B/M/J/K 완료학습·평가재실행금지. allFIT손모델/최종8/운영웹미실행. 공통100/C·T3F불변. 같은Codex별도검산≠독립사람/KSL전문가정답.

H=research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113. Python=ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. G0/G1PASS,G2REWORK,G3이후보류. 사용자/멘토자료·촬영·다운로드·Git·외부전송·운영기본교체금지. 사용자새벽계속진행승인유효,중단상태아님.
''','utf-8');print(status,session)
if __name__=='__main__':main()
