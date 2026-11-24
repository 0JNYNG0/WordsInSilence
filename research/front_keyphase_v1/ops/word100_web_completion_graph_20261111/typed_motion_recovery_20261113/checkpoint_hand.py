from common import *
import argparse
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=int);args=ap.parse_args();U=HERE/'hand_transition_learning'
    summary='몸 진행 분리(B)는 정상·속도108card와 정지대조를 유지했으나 부분진행·복합구간의 손실로 웹 미채택(저장검산9779PASS). 새 직접RGB206프레임과 기존11표현의 부분속성 정답을 연결해37공개FIT클립·1014속성셀을 준비했다. 한손로컬63입력의 작은Conv+GRU 부품을 수행자02..07을 한명씩 뺀6분할×3초기값으로 고정학습 중이다. 기존 공통100/손모양/TCN/운영웹은 수정하지 않는다.'
    nxt='hand_transition_learning train.py session54288 실제 완료를 확인하고 완료18개 모델을 재학습하지 않는다. 새206프레임 손상태·닫힘/열림/유지와 닫힌손으로팔세움 대조를 사람별로 검산한다. 다른표현은부분속성 범위만주장한다. 병행partial_direction_contract는FIT팔축회전/관측오류RGB검토이며 새방향학습전근거확인. G2전체gate전G3/최종8/웹교체금지.'
    status='ACTIVE_G2_HAND_TRANSITION_LOPO_TRAINING';jobs=[dict(name='hand_transition_lopo_train',session_id=args.session)] if args.session else []
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command='hand_transition_learning/train.py',jobs=jobs,session_id=args.session,other_session_id=None,training_running=bool(jobs),process_running=bool(jobs),next=nxt,last_completed=str(U/'INPUTS_COMPLETE.json'));write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    if not any(n['id']=='A3'for n in s['nodes']):s['nodes'].append(dict(id='A3',title='손 상태 변화와 팔 이동 구분 학습',status='TRAINING_LOPO',evidence=['hand_transition_learning/FROZEN.json','typed_shape_transition_review/HAND_TRAINING_INDEPENDENT_REVIEW.json']))
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=bool(jobs),progress=summary,remaining=nxt,latest_evidence=str(U/'INPUTS_COMPLETE.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 손모양 전환 부품·수행자 제외 검증'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\n새hand분기는22,087parameters이며새부품지도학습이지기존100전체미세조정이아니다.37클립11표현은손모양부분속성,닫힘전환정답은0739오른손6원본뿐.6fold02..07은기존FIT노출진단이며독립최종평가아님. 5/6fold평가클립은각1개. oldexplicit434셀,새206+기존합1014셀,4976real/reversewindow와2796동일자세대조. 내부unknown1182window는전후변화근거일뿐가린정확한경로라벨아님. 최초importnn오류는FIRST_FAILURE보존후수정.\n\nB/M/P/S/N/H이전학습평가모두완료,재실행금지. M5416/Binput8065/Boutput9779PASS. motion에이전트FIT투영부호8충돌의원본RGB확인중. absence는hand훈련계약검토완료,추후저장결과검산.\n\nC/T3-F/공통100/운영웹보존,사용자/멘토/최종8미접근. 별도Codex검산≠사람전문가정답. 사용자새벽자율지속승인유효;작업중단아님.\n','utf-8')
    print(status,jobs)
if __name__=='__main__':main()
