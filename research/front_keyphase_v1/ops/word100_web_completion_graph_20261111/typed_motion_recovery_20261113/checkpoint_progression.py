from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=int);ap.add_argument('--command',default='progression_presence_learning/train.py');args=ap.parse_args()
    summary='공개32영상의 연속 유지 구간을 확인해 적합 후보 없음(null)을 학습했다. 저장 비교에서 유지 구간의 억지 선택은 FIT210/216→0/216, 이미 노출된 수행자분리 probe72/72→0/72로 줄었으나 기존 정답extent6개 손실과 실제 부분진행 억제가 확인되어 웹 미채택이다. 방향 문맥 분리·반전대칭 후보도 정상 회귀로 미채택. 두 행동이 경계 한 시각만 공유할 때 겹침으로 계산하던 산술을 별도 감사로 정정했다(동작 성능 개선과 구분).'
    nxt='G2 원인에 맞춰 기존 extent 순위 모델은 동결하고, 직접 확인한 진행 표본과 유지 구간으로 별도 진행존재 판정부를 학습·검증한다. 짧은 실제 움직임을 완료로 승격하거나 누락으로 숨기지 않고, 모든 관측적격 후보의 진행 근거와 기존선택extent를 따로 보존한다. family2 부재 정답 없음은 원래출력 유지. 전체 정상/속도/교차/손모양/방향/순서/누락 회귀 통과 전 G3·최종8·웹 교체 없음.'
    status='ACTIVE_G2_PROGRESSION_VS_EXTENT_SEPARATION'
    write(HERE/'SUBGRAPH.json',dict(at=stamp(),status=status,nodes=[
        dict(id='A',title='측정 대상·학습/실행 입력 계약',status='AUDITED',evidence=['VERIFICATION.json']),
        dict(id='B',title='방향 문맥 분리와 반전 대칭',status='COMPARED_NOT_ADOPTED',evidence=['direction_isolation_learning/VERIFICATION.json']),
        dict(id='C',title='연속 유지 정답·후보 없음 학습',status='HOLD_GAIN_PARTIAL_REGRESSION',evidence=['null_candidate_learning/VERIFICATION.json','null_candidate_learning/local_bag_evaluation/INDEPENDENT_ARITHMETIC_VERIFICATION.json']),
        dict(id='C2',title='진행 존재와 전체 구간 선택 분리',status='IN_PROGRESS',evidence=['progression_presence_learning/train.py']),
        dict(id='D',title='정상·복합오류 통합 회귀',status='PENDING')],loop='실패 보존→원인 분리→공개 FIT 정답/계약 확인→새 조건 동결→새 비교. 완료 실험 재실행 금지.',no_word_specific_rules=True))
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,active_unit=str(HERE/'progression_presence_learning'),next=nxt,current_command=args.command,session_id=args.session,training_running=bool(args.session and 'train.py'in args.command),process_running=bool(args.session));write(GRAPH/'CURRENT_EXECUTION.json',e)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(status=status,at=stamp(),progress=summary,remaining=nxt,active_unit=str(HERE/'progression_presence_learning'),latest_evidence=str(HERE/'null_candidate_learning/local_bag_evaluation/INDEPENDENT_ARITHMETIC_VERIFICATION.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 유지 구간 학습 결과·부분진행 분리 보완'
    if title not in t:
        for h,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 G2 야간 진행\n\n'+summary+'\n\n'+nxt+'\n\n현재 실행: '+str(GRAPH/'CURRENT_EXECUTION.json')+'; 실제 세션/완료파일 먼저 확인하고 중복 실행 금지.\n\n완료보존: H 입력2406, 반전교정2조건×3seed, direction isolation2조건×3seed, null3seed,82시나리오×3seed 비교. 독립검산10338/7600/7367 PASS, local bag858행·부분17469행 검산. 같은 Codex 별도검산은 독립 사람정답 아님. probe08/09는 이전 노출된 개발자료, 신규 최종평가 아님.\n\nC·T3-F/공통100/웹 유지. 사용자·멘토자료와 최종8 접근 없음. 공개 원본읽기전용. 발표자료 output/pdf/final_check_latency_20261112 보존. 현재RTMW 촬영시점 비교35쌍5.3923→2.5734초는 과거9초전체경로비교가 아님; queuefull1회 원인은 후속G6.\n','utf-8')
    print(status,args.session)

if __name__=='__main__':main()
