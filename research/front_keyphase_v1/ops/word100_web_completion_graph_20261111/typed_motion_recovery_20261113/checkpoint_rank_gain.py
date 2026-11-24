from common import *

def main():
    assert (HERE/'splice_rank_learning/evaluation/COMPARISON.json').exists()
    assert (HERE/'hand_articulation_learning/FINAL_BRANCH_DECISION.json').exists()
    summary='손 전체 회전 보완 R의 대표 개선은 유지된다. 각도·길이비를 추가한 A는 18분할 검산 후 짧은 변화 문제를 해결하지 못하고 일부 유지가 보류돼 미채택이다. J는 공개 FIT 연결대조를 추가 학습해 합성 순서 시험의 실제 동작 구간 포착13/18→18/18, 역순 확인4/9→9/9로 개선했다. 그러나 빠른 정상1사례의 구간이33ms 겹쳐 2초기값에서 순서 근거가 줄어 아직 G2 REWORK이다. 단순 분포 재계산도 이 문제를 해결하지 못했다. 웹·TCN·공통100은 불변.'
    nxt='완료 A/J 학습·평가 재실행 금지. 동일가중 old+J 순위 대조는 새학습 없이 저장점수로 한조건만 검산한다. 공개 FIT의 시간재표본화 학습 공백을 확인해 .8/1.2 고정 변형의 정답 보존 가능성부터 감사한다. 중요한 정답 시점이 사라지면 전체동작 양성을 강제하지 않는다. J가 바꾼 구간의 기존 의미 판정은 새37고유요청만 검토준비한다. 최종8·웹·개인촬영 접근 없이 그래프 G2원인보완을 계속한다.'
    status='ACTIVE_G2_SPLICE_GAIN_FAST_BOUNDARY_REWORK'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command='saved-score stability diagnostic / FIT time-sampling contract / changed-span semantic preparation',jobs=[],session_id=None,other_session_id=None,training_running=False,process_running=False,next=nxt);write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='A3':n.update(status='ROTATION_GAIN_ARTICULATION_ADDITION_NOT_ADOPTED',evidence=['hand_rotation_invariant_learning/saved_evaluation/PAIRED_COMPARISON.json','hand_articulation_learning/FINAL_BRANCH_DECISION.json'])
        if n['id']=='C4':n.update(status='SPLICE_18_OF_18_GAIN_FAST_BOUNDARY_REWORK',evidence=['splice_rank_learning/evaluation/COMPARISON.json','splice_rank_learning/evaluation/PTS_ORDER_AUDIT.json'])
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['last_updated']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=False,progress=summary,remaining=nxt,latest_evidence=str(HERE/'splice_rank_learning/evaluation/PTS_ORDER_AUDIT.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 연결 구간 개선·빠른 정상 경계 보완'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    p=root/'NEW_CHAT_HANDOFF.md';old=p.read_text('utf-8-sig');at=old.find('실행 완료:');history=old[at:]if at>=0 else old
    p.write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\nA train49809/eval27196/rotation모두완료. A대표53/54,회전318/324(6낮은확신),부분630/672/629이며추가학습X. J train9191/eval6841완료. 새72FIT묶음+기존178재생,다른family4/5/7출력보존. Jnormal/speed075끝점공유는0us이며수학검산상정상순서유지. speed125REAL11close source68..81은기존부터offset띠77..80을1넘었고newraise source80..88은onset80..82안에있어33ms겹침. DEV주석을수정하지않음. 조건부후보질량144행검산완료,fastmixed여전히남음. allFIT손모델/최종8/운영웹미실행.\n\n현재agent: absence=고정old/J logit평균 대조, palm_fixture=FIT시간증강라벨감사, motion=새구간 의미shadow실행기준비. 사용자지속승인유효,실험재실행/카메라/비공개/Git/외부전송금지.\n\n'+history,'utf-8')
    print(status)

if __name__=='__main__':main()
