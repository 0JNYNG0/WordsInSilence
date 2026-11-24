from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--body-prep',type=int);ap.add_argument('--runtime-prep',type=int);ap.add_argument('--direction-train',type=int);args=ap.parse_args()
    summary='진행존재/전체구간 분리와 정지 대조 보완은 완료·검산했다. 다만 몸 움직임을 그대로 두고 손정보만 제거한 FIT24장면에서 진행판정이24→6/6/0으로 줄어 손정보에 의존하는 입력경로를 확인했다. 따라서 손모양검사와 몸움직임존재 판정부를 역할별로 분리해 추가학습을 준비중이다. 별도로 axis2 FIT의 실제신체기준 변위부호와 정답이 일치함을 확인해, 양의변위를반대방향이라부르지않는 최소단조방향부품을학습중이다.'
    nxt='body_progression_learning 공개286bag(body190기존캐시+96hold)와190pairedstatic,82runtimebodycache 준비를완료하면3seed1200step학습및동일local/static/full비교. monotone_net_direction은3seed800step후axis2만기존savedgeometry로비교·검산. typed_shape_transition_review는6FIT206프레임RGB전수확인중: 손이닫히는변화와이미닫힌손의이동을분리. 기존출력/보호/최종8유지,completed실험rerun금지.'
    status='ACTIVE_G2_TYPED_BODY_PROGRESS_AND_NET_DIRECTION'
    jobs=[dict(name=n,session_id=s)for n,s in [('body_prepare',args.body_prep),('runtime_body_prepare',args.runtime_prep),('net_direction_train',args.direction_train)]if s]
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,active_unit=str(HERE),next=nxt,current_command='body-only input preparation + monotone net-direction; RGB FIT review in agent',session_id=None,other_session_id=None,jobs=jobs,training_running=bool(args.direction_train),process_running=bool(jobs));write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='A2':n.update(title='몸 진행·손모양변화·방향을 역할별로 분리',status='CONTRACT_AUDITED_PREPARING',evidence=['paired_static_progression_learning/hand_information_contract/RESULTS.json','axis2_sign_audit/RESULTS.json'])
        if n['id']=='C3':n['evidence']=['paired_static_progression_learning/VERIFICATION.json','paired_static_progression_learning/static_evaluation/COMPARISON.json']
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,progress=summary,remaining=nxt,active_unit=str(HERE),latest_evidence=str(HERE/'paired_static_progression_learning/hand_information_contract/RESULTS.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 몸 진행·손정보 분리와 순방향 측정 계약'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 새벽 자율 G2 보완\n\n'+summary+'\n\n'+nxt+'\n\n실제 진행명령·session: '+str(GRAPH/'CURRENT_EXECUTION.json')+'의jobs를먼저확인. 학습/준비/eval완료파일존재시중복실행금지. 에이전트motion은B코드작성과검토,absence는FIT6원본206프레임손모양전환주석중.\n\n이미완료: H반전2조건×3seed, 방향분리2조건×3seed, Nnull3seed, P진행분리3seed, S정지짝3seed 및full/local/static평가. S검산9821PASS,정상속도108card/기존정답extent505보존,probe부분4287/4356(직전4338),정지probe0/44/0 per6516. 합성순서실패5/18은모두접합부교차:실제연속사용자오류성능과동일시금지. composition은saved결과조합만이며미통과.\n\n원본·기존C/T3-F·공통100·웹보존. 사용자/멘토/최종8미접근. 새HTML없음. 단어특례금지; typedfamily2새absence정답없어원본유지. 같은Codex검산≠독립사람정답. 연구는중단/완료아니며밤작업계속승인유효.\n','utf-8')
    print(status,jobs)

if __name__=='__main__':main()
