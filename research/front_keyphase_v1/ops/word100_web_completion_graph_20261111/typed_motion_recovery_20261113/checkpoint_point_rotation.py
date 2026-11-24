from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='PREPARING_POINT_ROTATION_CONTRACTS');ap.add_argument('--session',type=int);ap.add_argument('--command',default='input contracts and independent saved arithmetic');ap.add_argument('--secondary-session',type=int);ap.add_argument('--secondary-command');a=ap.parse_args()
    summary=('손 상태·전환의 수행자 제외 18회 학습과 저장 결과 계산을 완료했다. 대표 54조건 중 51조건을 통과했으나 REAL05의 닫힌 손으로 팔 세우기를 세 초기값 모두 손 펴기로 오판했다. 기존 모델은 미교체이며 손 전체 회전과 굽힘을 분리하는 입력 보완을 준비한다. 별도 FIT 측정 감사에서 설명은 손바닥 이동인데 최근 계산은 몸 손목을 사용한 불일치를 확인했다. 같은 후보의 부분 진행 부호 충돌은 손목 8건, 손바닥 0건이지만 이는 학습 성능이 아닌 관측점 계약 검토다. 짧은 진행과 유지 변위는 여전히 겹친다.')
    nxt=('손바닥 기준 방향 학습 입력 계약과 손 전체 회전 불변 손모양 입력을 각각 검토한다. 기존 주먹 합성은 MCP도 바꿨으므로 새 말단 손가락만 바꾸는 대조를 별도 준비해 손모양 변경이 팔 이동 판정에 끼치는 영향을 분리한다. 근거를 고정한 뒤 필요한 학습·고정 비교·별도 검산을 이어간다. 기존 완료 평가 재실행 및 G2 통과 전 웹 교체 금지.')
    jobs=[dict(name=a.command,session_id=a.session)] if a.session else []
    if a.secondary_session:jobs.append(dict(name=a.secondary_command,session_id=a.secondary_session))
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=a.stage,current_command=a.command,jobs=jobs,session_id=a.session,other_session_id=None,training_running=bool(jobs),process_running=bool(jobs),next=nxt,last_completed=str(HERE/'hand_transition_learning/saved_evaluation/RESULTS.json'));write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=a.stage)
    for n in s['nodes']:
        if n['id']=='A3':n.update(status='REWORK_ROTATION_VS_FLEXION',evidence=['hand_transition_learning/saved_evaluation/RESULTS.json'])
    if not any(n['id']=='A4' for n in s['nodes']):s['nodes'].append(dict(id='A4',title='손바닥 이동과 손 전체 회전의 관측 계약',status='CONTRACT_REVIEW',evidence=['partial_direction_contract/POINT_CONTRACT_REVIEW.txt','palm_preserving_controls/REVIEW_KO.txt']))
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=a.stage,training_running=bool(jobs),progress=summary,remaining=nxt,latest_evidence=str(HERE/'hand_transition_learning/saved_evaluation/RESULTS.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 손바닥 이동·손 전체 회전 분리'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\n현재 단계: '+a.stage+'\n실행 세션: '+str(jobs)+'\n\nU hand_transition_learning train18회 완료(session54288 exit0), saved_evaluation 완료. 54조건은6원본×3종×3seed로 독립54영상 아님. 손모양 정답11표현37클립1014셀이나 전환 정답은0739오른손6원본뿐. 후보 선택에 사용된 진단이며 독립 최종 평가 아님. absence 에이전트 별도 산술·원인검토, motion 에이전트 FIT 손바닥 방향계약, palm_fixture 에이전트 말단만 바꾸는 합성대조 준비. root는 새 helper 검토 및 toy12개 PASS.\n\nH/N/P/S/B/M 이전 학습·평가 완료/재실행 금지. B 정상·속도108card/기존정확구간505/505 유지와 정지오판0은 확인했으나 부분진행4210/4356, 복합구간 손실로 미채택. U 최초 import nn 오류 보존. 모든 새 작업은 분리 폴더. C/T3-F/공통100/운영웹/원본 보존. 사용자·멘토·최종8·카메라·다운로드·Git·외부전송 없음. 별도 Codex 검산은 독립 사람/KSL전문가 정답 검증이 아님. 사용자 새벽 자율지속 승인 유효.\n','utf-8')
    print(a.stage,jobs)

if __name__=='__main__':main()
