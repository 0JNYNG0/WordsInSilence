from common import *
def main():
    status='ACTIVE_G2_TYPED_MOTION_AND_NULL_SUPERVISION'
    summary='학습 반전 입력의 실제 시간문맥 불일치를 확인하고 교정한 고정 비교를 완료했다. 어깨폭 투영과 문맥 교정을 함께 쓰면 정상/속도/역방향/복합의 방향 판단은 개선됐으나 손모양 단독 변경의 오경고가 남아 미채택. 동일 FIT 구간 손정보제거 21쌍에서는 오판 재현이 없어 단일 원인으로 단정하지 않았다. 몸 움직임 방향을 손정보와 분리한 구조 비교와 연속 유지 구간의 부재 감독 보강을 이어간다.'
    nxt='G2의 방향판정 분리·반전대칭 후보를 고정 검증하고, 직접 연속 확인한 공개 FIT 유지 구간으로 억지 후보 선택 여부를 먼저 확인한다. 필요시 null 후보를 학습하되 정상 동작/속도/교차/중요손모양/방향/순서/누락을 함께 유지해야 통합 단계로 간다. 최종8·웹·기존 C/T3-F·공통100·사용자자료는 유지.'
    write(HERE/'SUBGRAPH.json',dict(at=stamp(),status=status,nodes=[dict(id='A',title='측정 대상·학습/실행 입력 계약',status='AUDITED',evidence=['INPUT_CONTRACT_AUDIT.json','VERIFICATION.json']),dict(id='B',title='방향 문맥 분리와 물리적 반전 대칭',status='IN_PROGRESS',evidence=['context_isolation_audit/RESULTS.json','direction_isolation_learning/FROZEN.json']),dict(id='C',title='실제 유지 구간 확인·적합 후보 없음 학습',status='IN_PROGRESS',evidence=['absence_review/REVIEW_COMPLETE.json']),dict(id='D',title='정상·복합오류 통합 회귀',status='PENDING')],loop='실패 보존→구간/관측/판정 원인 분리→FIT 근거 보완→새 조건 동결→새 비교. 완료 실험 반복 금지.',no_word_specific_rules=True))
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(status=status,at=stamp(),active_unit=str(HERE),next=nxt,current_command='direction_isolation_learning/train.py',session_id=64664,training_running=True,process_running=True);write(GRAPH/'CURRENT_EXECUTION.json',e)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(status=status,at=stamp(),progress=summary,remaining=nxt,active_unit=str(HERE),latest_evidence=str(HERE/'VERIFICATION.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig')
    for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
        t=t.replace(heading,heading+'\n\n### 2026-11-13 반전 입력 계약 교정·유지 구간 주석 보강\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/VERIFICATION.json`.\n',1)
    p.write_text(t,'utf-8');p=root/'NEW_CHAT_HANDOFF.md'
    p.write_text('# 최신 인수인계 — 2026-11-13 G2 야간 재개\n\n'+summary+'\n\n'+nxt+'\n\n현재 세부 그래프: '+str(HERE/'SUBGRAPH.json')+'\n현재 실행 기록: '+str(GRAPH/'CURRENT_EXECUTION.json')+' (실제 세션/완료파일 먼저 확인; 중복 실행 금지).\n\n이미 완료: prepare.py2406항목, train.py2조건×3seed, evaluate.py1회, verify.py10338검사PASS. 전체 G2는REWORK. geometry·정답유지/정상context동일, FIT역방향915항목context차이확인. 원래비교/최초오류보존. 별도Codex검산≠독립사람정답.\n\n발표자료는 output/pdf/final_check_latency_20261112 보존. 공개속도비교 완료35쌍5.3923→2.5734초는현재RTMW처리시점비교이며과거9초전체경로와다름. 1회queuefull은미해결. 현재연구는카메라/학습외자료접근없음. 최종8미평가; C·T3-F/공통100/웹교체없음.\n','utf-8')
    print('Checkpoint updated')
if __name__=='__main__':main()
