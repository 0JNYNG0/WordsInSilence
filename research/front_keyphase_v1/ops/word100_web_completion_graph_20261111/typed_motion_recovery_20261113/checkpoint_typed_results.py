from common import *

def main():
    R=HERE/'hand_rotation_invariant_learning';P=HERE/'palm_direction_learning'
    assert (R/'TRAINING_COMPLETE.json').exists() and (P/'TRAINING_COMPLETE.json').exists()
    pair=read(R/'saved_evaluation/PAIRED_COMPARISON.json')
    assert pair['new_canonical_pass']==54 and pair['canonical_total']==54
    summary=('손 전체 회전을 분리한 손모양 부품은 대표 비교51/54→54/54, 새로운 회전 대조245/324→324/324로 개선됐다. 그러나 REAL06의 굽힘 끝 한 프레임을 공유하는 짧은 구간에서 일부 초기값이 변화를 놓쳐 전체 교체하지 않았다. 기존 손모양 판정은 보존한다. 손바닥 방향 부품은 역방향·복합대조 각각6/9→9/9 반대 방향 근거를 찾았으나 느린 정상 동작의 지지9/9→6/9로 손실이 있어 미채택이다. REAL13의 끝 전완 축 길이 급변으로 순변위가 상쇄되는 현상을 추가 원본 확인 중이다. G2 REWORK이며 웹은 변경하지 않았다.')
    nxt=('새 학습·비교는 모두 완료, 재실행 금지. 손의 짧은 변화 실패가 관측 표현인지 시간 학습 문제인지 원본74프레임 전후와 저장 출력을 비교한다. REAL13의 팔 기준축 변화도 원본과 대조해 관측 오류·실제 동작을 구분한다. 동시에 FIT 편집 경계에 걸친 후보를 한 행동으로 잘못 연결하는 학습 정답 문제를 검토한다. 수정한 관측·구간 연결을 합쳐 G2 고정 회귀를 통과한 뒤에만 웹/최종8 단계로 진행한다.')
    status='ACTIVE_G2_TYPED_GAINS_WITH_SHORT_CHANGE_AND_ENDPOINT_REWORK'
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status=status,current_command='saved arithmetic and RGB/association contract review',jobs=[],session_id=None,other_session_id=None,training_running=False,process_running=False,next=nxt,last_completed=str(R/'rotation_controls/RESULTS.json'));write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status=status)
    for n in s['nodes']:
        if n['id']=='A3':n.update(status='REPRESENTATIVE_ROTATION_GAIN_PARTIAL_REWORK',evidence=['hand_rotation_invariant_learning/saved_evaluation/PAIRED_COMPARISON.json','hand_rotation_invariant_learning/rotation_controls/RESULTS.json'])
        if n['id']=='A4':n.update(status='DIRECTION_GAIN_NORMAL_SPEED_ENDPOINT_REWORK',evidence=['palm_direction_learning/evaluation/RESULTS.json','palm_direction_learning/VERIFICATION.json'])
    write(HERE/'SUBGRAPH.json',s)
    state=read(OPS/'STATE.json');state['updated_at']=stamp();state['next_single_action']=nxt;state['word100_web_completion_graph_20261111'].update(at=stamp(),status=status,training_running=False,progress=summary,remaining=nxt,latest_evidence=str(R/'saved_evaluation/PAIRED_COMPARISON.json'));write(OPS/'STATE.json',state)
    root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 회전 혼동 개선·짧은 변화와 끝 관측 보완'
    if title not in t:
        for h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:t=t.replace(h,h+'\n\n'+title+'\n'+b+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/SUBGRAPH.json`.\n',1)
        p.write_text(t,'utf-8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 새벽 G2 자율 작업\n\n'+summary+'\n\n'+nxt+'\n\n실행 완료: hand_rotation_invariant_learning train71330, evaluate_saved97643, rotation_controls 모두 exit0. palm_direction_learning train48495/evaluate exit0. R 18LOPO 모델, P 3seed 모델을 다시 학습하지 않는다. 새 모형은 사용자/멘토/최종8 미접근, 기존 공통100/C/T3-F/운영웹 불변.\n\n대표54=6원본×3조건×3seed, 회전324=대표54×6합성각도이며 독립 영상 수가 아니다. 원래 U 학습완료와 결과 보존. R 속성셀 총866/seed 중853/846/840맞음(원래812/808/817); 일부 개별 퇴보도 존재. 부분 전환630/672/630 per672로 원래653/644/647과 비교, REAL06 frame74공유42중첩창 실패를 별도 원인 검토. 새 allFIT hand 모델 아직 없음.\n\nP는 직접 부분1082에 진행 여부만 학습, 방향 정답 복제 없음. 저장검산6088PASS/보호477, 그러나 정상속도 회귀로 미채택. 검산 보고서 int64직렬화 최초오류 보존 후 저장 산술만 복구. REAL13[55,70]끝전완길이.2035→.0623, palm unitdelta-.01008; arm-fraction은더나빠 대체근거없음. 원본 확인 없이 오류로 단정하거나 문턱을 조정하지 않는다.\n\n새 palm_preserving_controls 24개는 수치 불변대조일 뿐 닫힌손 정답으로 사용 금지. 500 frame-side중493적용7보류,432기하비교 동일, 별도검산462PASS. root는 CONTROL_OVERVIEW와 REAL02axis불일치/RGB REAL05주먹 유지 사진을 직접 확인했다. body_action_binding_review는 공통rank 5실패가 합성연결경계에 걸친다는 근거로 FIT provenance 보강과 새 몸 신경망을 비교검토 중.\n\n모든 과거 H/N/P/S/B/M 실험 완료/재실행 금지. currentagents absence=R검산/REAL06, motion=P검산/REAL13, palm_fixture=FIT몸 관측/연결 정답 설계. 별도Codex검산≠독립 사람/KSL전문가 정답. 연구 지속 승인 유효이며 중단 아님. 원본읽기전용·촬영/다운로드/Git/외부전송 금지.\n','utf-8')
    print(status)

if __name__=='__main__':main()
