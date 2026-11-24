from common import *

def main():
    T=HERE/'typed_shape_transition_review';B=HERE/'body_progression_learning';M=HERE/'monotone_net_direction'
    p=T/'ROOT_VISUAL_REVIEW.json'
    if not p.exists():
        specs=[('04','02'),('07','02'),('03','02'),('06','04')]
        paths=[T/f'NIA_SL_WORD0739_REAL{person}_F'/f'page_{page}.jpg' for person,page in specs]
        write(p,dict(at=stamp(),reviewer='root Codex, separate visual sample inspection; not independent human/KSL expert',
            viewed=[dict(path=str(q),sha256=sha(q)) for q in paths],
            judgment='Agree open/flexing/closed distinction, conservative ambiguous boundaries and closed-hand forearm raising. No forcing image-blur boundaries or fully occluded fingers.',
            limitations='Root sampled24 panels; agent reports206 frames. Neither is human-expert ground truth. Six FIT people of one expression cannot establish other-expression generalization.',
            review=str(T/'REVIEW_COMPLETE.json'),review_sha256=sha(T/'REVIEW_COMPLETE.json'),approved_for_research_input_quality_audit=True,model_effectiveness_established=False))
    summary='몸 움직임 존재를 손정보와 분리한 후보3개 학습·local/static/full 비교를 완료했다. 실제 유지와 합성정지 오탐은0이며 정상/속도 지지는 보존했지만 짧은 부분진행 일부를 놓치는 교환관계가 남아 미채택이다. 변위부호를 지키는 방향부품도 정상 지지 감소로 미채택. 공개FIT6명206프레임 손모양변화/팔움직임 정답을 직접RGB확인했고 루트가24개 패널을 별도표본검수했다.'
    nxt='B 저장결과의 별도검산을 마무리하고, 새 직접RGB정답을 기존 RTMW 관측 및 다른표현 손모양정답과 연결한다. 손이닫히는변화와 닫힌손의팔이동을 분리하는 범용 학습계약을 검토한 뒤 관측가능성/수행자분리 검증을 통과하면 새시간별판정부를학습·비교한다. 기존completed 재실행·최종8·사용자·웹교체금지.'
    status='ACTIVE_G2_TYPED_SHAPE_OBSERVATION_AND_BODY_VERIFICATION'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,active_unit=str(HERE),next=nxt,current_command='saved body evaluation verification and FIT hand-transition training design',jobs=[],session_id=None,other_session_id=None,training_running=False,process_running=False,last_completed=str(B/'evaluation/COMPARISON.json'),subgraph=str(HERE/'SUBGRAPH.json'),reviewer='root Codex plus delegated Codex reviews; no independent human truth');write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='A2':n.update(status='RGB_REVIEW_COMPLETE_BODY_COMPARED',evidence=['body_progression_learning/INPUT_TRAINING_REVIEW.json','body_progression_learning/evaluation/COMPARISON.json','monotone_net_direction/VERIFICATION.json','typed_shape_transition_review/ROOT_VISUAL_REVIEW.json'])
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=False,progress=summary,remaining=nxt,active_unit=str(HERE),latest_evidence=str(T/'ROOT_VISUAL_REVIEW.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 몸 진행 비교 완료·손모양 전환 직접 정답'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\n모든 B/M prepare/train/eval명령은완료. B evaluate_local85686/full27809도exit0. B local/static/full COMPARISON 존재; 중복실행금지. motion에이전트B저장평가검산중, absence에이전트다른FIT정답연결/typedshape학습설계중.\n\nH/paired_static_progression_learning=S(부분probe4287/4356), body_progression_learning=B(부분probe4210/4356), progression_presence_learning=P(4338/4356). 모두부분진행단위·반복view/seed이고사람수아님. B합성정지probe0/6516씩3seed, M5416검산PASS이나정상지지손실. web미교체. 손관측정보삭제는bodyhead입력에만적용하며손모양판정기능삭제아님.\n\n원본/C/T3-F/공통100/운영웹보존, 사용자·멘토·최종8미접근. 같은Codex검산≠독립사람정답. G2 REWORK 후 G3~G6통과전완료선언금지. 사용자새벽자율지속승인유효; 연구중단아님.\n','utf-8')
    (HERE/'PROGRESS_KO.txt').write_text(summary+'\n\n다음: '+nxt+'\n','utf-8')
    print(status)

if __name__=='__main__':main()
