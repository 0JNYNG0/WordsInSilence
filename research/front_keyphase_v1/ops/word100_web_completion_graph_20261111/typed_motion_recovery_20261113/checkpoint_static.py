from common import *
import argparse

def main():
    a=argparse.ArgumentParser();a.add_argument('--session',type=int);a.add_argument('--other',type=int);a.add_argument('--command',default='paired_static_progression_learning/train.py');args=a.parse_args()
    summary='진행 존재와 전체 구간 선택을 분리한 후보를 학습·검산했다. 공개 FIT의 부분진행13113/13113, 노출 probe4338/4356 후보를 지지하며 실제 유지 오탐0, 기존에 맞던 구간505/505 지지를 보존했다. 중복 view/reverse/seed를 포함한 후보 단위이며 사람수나 최종정확도가 아니다. 전체 정상·속도108card 지지 보존, 31541검산PASS. 그러나 같은자세를정지시킨 FIT 합성입력72개에서 seed별44/42/38개 진행오탐이 있어, 실제시간변화와 자세를 더 분리해야 한다. 웹은 아직 미교체.'
    nxt='paired_static_progression_learning: 사전고정 모든 full/partial 후보의 시작·중간·끝 자세를 같은길이·dt로 고정한 대조를 추가학습한다. 관측되지 않은 anchor는정지negative에서제외. 실제 부분진행·유지 정답과 oldextent/공통100은 유지한다. 동일학습정지 통과를신규평가로부르지않고 수행자분리 정지·실제동작회귀·전체82공개시나리오를별도검산한다. G2의 실제범위·방향·순서·누락은아직남음; G3/최종8/웹교체는전체gate후.'
    status='ACTIVE_G2_PAIRED_STATIC_PROGRESSION'
    s=read(HERE/'SUBGRAPH.json');s['at']=stamp();s['status']=status
    for n in s['nodes']:
        if n['id']=='C2':n.update(status='NARROW_COMPONENT_VERIFIED_STATIC_FAILURE',evidence=['progression_presence_learning/VERIFICATION.json','progression_presence_learning/local_evaluation/INDEPENDENT_ARITHMETIC_VERIFICATION.json','progression_presence_learning/static_contract_audit/RESULTS.json'])
    if not any(n['id']=='C3'for n in s['nodes']):s['nodes'].insert(-1,dict(id='C3',title='같은 자세와 실제 변화 구분',status='IN_PROGRESS',evidence=['paired_static_progression_learning/PREPARATION_CONTRACT.json']))
    write(HERE/'SUBGRAPH.json',s)
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,active_unit=str(HERE/'paired_static_progression_learning'),next=nxt,current_command=args.command,session_id=args.session,other_session_id=args.other,training_running=bool(args.session and'train.py'in args.command),process_running=bool(args.session or args.other));write(GRAPH/'CURRENT_EXECUTION.json',e)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,progress=summary,remaining=nxt,active_unit=str(HERE/'paired_static_progression_learning'),latest_evidence=str(HERE/'progression_presence_learning/VERIFICATION.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 진행 존재 분리 개선·동일자세 정지 대조'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 G2 야간 진행\n\n'+summary+'\n\n'+nxt+'\n\n실행 기록: '+str(GRAPH/'CURRENT_EXECUTION.json')+' — 실제 세션·완료파일 확인, 완료실험 재실행 금지.\n\n이전 완료: 반전교정2조건×3seed, 방향분리2조건×3seed 미채택; null3seed 유지오탐감소했으나부분진행억제로미채택. 진행존재분리3seed는좁은component개선확인, 동일자세정지실패보완중. 공동경계한시각의겹침산술7관계정정은동작성능개선과구분. 원본/최초오류/기존결과보존.\n\nC·T3-F/공통100/웹/최종8유지, 사용자·멘토자료접근없음. 별도Codex검산≠독립사람정답. 후보중복수≠사람수,probe08/09이전노출≠신규최종평가. 발표자료보존(output/pdf/final_check_latency_20261112); queuefull1회는G6후속.\n','utf-8')
    print(status,args.session)

if __name__=='__main__':main()
