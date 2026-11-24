from common import *

def main():
    a=HERE/'hand_articulation_learning'
    assert (a/'TRAINING_CONTRACT.json').exists()
    done=(a/'TRAINING_COMPLETE.json').exists()
    progress=read(a/'TRAINING_PROGRESS.json') if (a/'TRAINING_PROGRESS.json').exists() else {'done':0,'total':18}
    status='ACTIVE_G2_ARTICULATION_AND_SPLICE_REPAIR'
    summary='손 전체 회전과 손가락 펴기를 혼동하던 대표 문제는 개선됐다. 남은 짧은 굽힘 전환은 원본에서 관측 정보가 존재함을 확인하고, 기존 입력·가중치를 복사한 모델에 손가락 각도·길이비 24값을 0잔차로 추가해 같은 사람 제외 조건으로 미세조정 중이다. 별도로 실제 움직임이 있는 서로 다른 구간을 잘못 묶는 5사례의 합성 연결 경계 문제를 확인해 FIT 연결 대조72묶음을 준비·검산했다. 웹은 미교체, G2 REWORK.'
    nxt='A 학습 exec49809 완료를 확인하고 새 모델만 저장 비교·회전 대조·별도검산한다. J FIT 연결대조 학습기를 검토해 새 3seed 고정학습과 기존82조건 비교를 한다. 팔 기준축 신뢰성은 FIT 근거로 분리검토하며 임의 보간·DEV 문턱 선택은 하지 않는다. G2 통과 전 최종8·웹 교체 금지.'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command='hand_articulation_learning/train.py',session_id=None if done else 49809,training_running=not done,process_running=not done,jobs=[] if done else [dict(unit='hand_articulation_learning',exec_session=49809,completed_folds=progress['done'],total=18)],next=nxt);write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='A3':n.update(status='ARTICULATION_FINETUNE_COMPLETED' if done else 'ARTICULATION_FINETUNE_RUNNING',evidence=['hand_articulation_learning/FROZEN.json','hand_articulation_learning/TRAINING_CONTRACT.json'])
    if not any(n['id']=='C4' for n in s['nodes']):s['nodes'].append(dict(id='C4',title='실제 한 행동과 합성 연결 구간 구분',status='FIT_INPUTS_VERIFIED_TRAINER_REVIEW',evidence=['splice_rank_learning/FROZEN.json','splice_rank_learning/VERIFICATION.json']))
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['last_updated']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=not done,progress=summary,remaining=nxt,latest_evidence=str(a/'TRAINING_CONTRACT.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 짧은 손 변화와 잘못 연결된 구간 분리 보완'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    p=root/'NEW_CHAT_HANDOFF.md';old=p.read_text('utf-8-sig');at=old.find('실행 완료:');history=old[at:] if at>=0 else old
    p.write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\nA18분할×300step, 부모R와 동일 사람/정답/샘플링이며 새 가중치는0으로 초기화했다. 기존63입력·기존모델보존. J72묶음은 합성불연속 연결을 단일행동 전체구간 선택의 음성으로만 사용하며 손모양·진행·방향 음성으로 쓰지 않는다. 공통100 인코더와 모든기존178FIT묶음을 유지한다. RD FIT 수학감사600PASS: 중앙기울기는끝점잡음에강하나짧은역행을숨길수있어전체판정으로미채택.\n\n'+history,'utf-8')
    print(status,progress)

if __name__=='__main__':main()
