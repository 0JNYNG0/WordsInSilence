from common import *

def main():
    S=HERE/'paired_static_progression_learning';status='ACTIVE_G2_TYPED_ACTION_EVIDENCE_REVIEW'
    summary='밤 작업에서 반전 입력 문맥 불일치와 경계 한 시각 공유의 순서 산술을 교정·검산했다. 진행 존재를 전체구간 선택과 분리하고 실제 유지+동일자세 정지 대조로 학습했다. 정상/속도108card와 기존정답extent505/505 지지는 보존. FIT정지오탐44/42/38→0은재학습진단이며신규성능아님. 노출probe의정지0/44/0 per6516, 부분진행4287/4356(직전4338/4356)으로잔여교환관계가있어웹미채택. 몸이움직였음·요구행동임·완료함은분리해야한다.'
    nxt='G2 typed action 정답/입력 계약을 점검한다: 손목이동·손모양변화·팔동작을 혼동하지 않는 직접근거를 기존공개FIT에서 확인하고, 몸움직임존재가 손정보오류 때문에 가려지는지 같은몸/다른손정보 대조로 확인한다. 합성블록순서 실패5/18은모두접합부교차였으므로 실제연속오류와구분보고. 편집위치만으로불연속정답을강제학습하지않고 정상빠른전이·교차·가림복귀와함께감사. 확인된원인만새학습계약으로보완하고 전체gate 전G3/최종8/웹교체금지.'
    (HERE/'PROGRESS_KO.txt').write_text('현재: 웹 피드백 완성을 위한 G2 보완 중\n\n'+summary+'\n\n이번에 확인한 개선\n- 가만히 유지한 자세에서도 동작을 억지로 고르던 문제를 줄였다.\n- 짧은 진행과 전체 동작 구간을 별도로 가르쳐, 실제로 조금 움직인 사례를 전부 누락으로 숨기는 문제를 줄였다.\n- 기존 정상 구간을 유지하면서 같은 자세의 정지 입력을 구분하는 학습 근거를 추가했다.\n\n아직 남은 것\n- 중요한 동작이 정확히 어느 구간인지, 해당 설명의 그 행동인지 구분이 완성되지 않았다.\n- 손모양 오류가 있는 동작도 비교에 남겨야 하는데 일부 후보는 연결불확실해진다.\n- 손모양만 바꾼 장면에 방향 오류를 덧붙이는 경우가 남는다.\n- 동작 일부 수행과 전체 완료/누락은 별도 검증이 필요하다.\n- 합성 편집 오류 시험은 실제 사람의 연속 오류 수행 성능과 같지 않다.\n\n'+nxt+'\n\n수치 주의: 후보창·뷰·반전·3seed는 같은 영상에서 반복된 판단이며 독립 사람 수가 아니다. probe08/09도 상류 모델에 노출된 개발 진단이다. 별도 Codex 검산은 독립 한국수어 전문가 정답이 아니다. 현재웹·C/T3-F·공통100원본·사용자/멘토자료·최종8은 유지했다.\n','utf-8')
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='C3':n.update(status='COMPARED_STATIC_GAIN_PARTIAL_TRADEOFF',evidence=['paired_static_progression_learning/VERIFICATION.json','paired_static_progression_learning/static_evaluation/COMPARISON.json'])
        if n['id']=='D':n.update(status='COMBINED_REWORK',evidence=['composition_audit/RESULTS.json','paired_static_progression_learning/FIXTURE_SPLICE_REVIEW.json'])
    if not any(n['id']=='A2'for n in s['nodes']):s['nodes'].append(dict(id='A2',title='몸 움직임·손모양변화·편집접합 근거 구분',status='IN_PROGRESS'))
    for n in s['nodes']:
        if n['id']=='A2':n['title']='몸 움직임·손모양변화·편집접합 근거 구분'
    write(HERE/'SUBGRAPH.json',s)
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,active_unit=str(HERE),next=nxt,current_command='Typed observation and source-label review; no model job active',session_id=None,other_session_id=None,training_running=False,process_running=False);write(GRAPH/'CURRENT_EXECUTION.json',e)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,progress=summary,remaining=nxt,active_unit=str(HERE),latest_evidence=str(S/'VERIFICATION.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 정지 대조 보완·복합 비교·행동 근거 재검토'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/PROGRESS_KO.txt`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 G2 야간 진행\n\n'+summary+'\n\n'+nxt+'\n\n현재명령: '+str(GRAPH/'CURRENT_EXECUTION.json')+'; 새학습프로세스없고에이전트검토중. 모든train/evaluate완료파일과실제세션확인후재개,완료재실행금지.\n\n완료: H 반전2조건×3seed, direction_isolation2조건×3seed, N null3seed, P progression3seed, S paired-static3seed. 각82시나리오×3seed와localdiagnostic저장, S검산9821PASS. composition은저장값조합만; 모델재추론없음. S정지prepare190bags24543counterfactuals(미관측FIT8anchor제외), 실제source32clips. 원래보호477/family2출력유지.\n\n기존C·T3-F/웹/공통100보존, 사용자·멘토/최종8미접근. 발표자료보존. 새학습/웹통합의근거는Subgraph와검토계약을따름. 연구중단이나완료가아니며사용자새벽자율진행승인유효.\n','utf-8')
    print(status)

if __name__=='__main__':main()
